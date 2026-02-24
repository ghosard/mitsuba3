#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <drjit/math.h> // 需要包含 math 头文件以使用 log, cos, sqrt

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class UniformHemispherePathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    // 定义成员变量存储噪声参数
    Spectrum m_bg_color;
    Float m_bg_sigma;

    UniformHemispherePathIntegrator(const Properties &props) : Base(props) {
        // 1. 读取噪声强度 (Sigma)
        // C++ 中使用 get<float>("名字", 默认值)
        // 注意：Properties 里存的是标量(float)，读取后赋值给 JIT 变量(Float)会自动转换
        float sigma_scalar = props.get<float>("sigma", 0.1f);
        m_bg_sigma = sigma_scalar;

        // 2. 读取底色 (Background Color)
        if constexpr (is_rgb_v<Spectrum>) {
            // 如果是 RGB 模式，XML 中的 <rgb> 标签会被解析为 mitsuba::Color<float, 3>
            // 我们需要显式指定读取这个类型
            using ScalarColor3f = mitsuba::Color<float, 3>;
            
            // 读取标量颜色
            ScalarColor3f c = props.get<ScalarColor3f>("bg_color", ScalarColor3f(0.5f));
            
            // 转换为 JIT Spectrum
            m_bg_color = Spectrum(c);
        } else {
            // 如果是单色/光谱模式，通常 XML 里写的是 <float> 或 <spectrum>
            // 简单起见，这里假设它是一个 float 值
            float c = props.get<float>("bg_color", 0.5f);
            m_bg_color = Spectrum(c);
        }
    }

    std::pair<Spectrum, Bool> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Bool active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        if (unlikely(m_max_depth == 0))
            return { 0.f, false };

        Ray3f ray                     = Ray3f(ray_);
        Spectrum throughput           = 1.f;
        Spectrum result               = 0.f;
        Float eta                     = 1.f;
        PreliminaryIntersection3f pi  = dr::zeros<PreliminaryIntersection3f>();
        UInt32 depth                  = 0;

        // 修改：初始 valid_ray 设为 false，我们会在击中光源或击中背景时将其置为 true
        Mask valid_ray = false; 

        struct LoopState {
            Ray3f ray;
            PreliminaryIntersection3f pi;
            Spectrum throughput;
            Spectrum result;
            Float eta;
            UInt32 depth;
            Mask valid_ray;
            Bool active;
            Sampler* sampler;

            DRJIT_STRUCT(LoopState, ray, pi, throughput, result, eta, depth,
                         valid_ray, active, sampler)
        } ls = {
            ray, pi, throughput, result, eta, depth, valid_ray, active, sampler
        };

        ls.pi = scene->ray_intersect_preliminary(ls.ray, true, false, 0, 0, ls.active);

        // ---------------------- Hide area emitters 逻辑 (保持不变) ----------------------
        if (m_hide_emitters && dr::any_or<true>(ls.depth == 0u)) {
            Mask skip_emitters = ls.pi.is_valid() && (ls.pi.shape->emitter() != nullptr) && ls.active;
            if (dr::any_or<true>(skip_emitters)) {
                SurfaceInteraction3f si = ls.pi.compute_surface_interaction(
                    ls.ray, +RayFlags::Minimal, skip_emitters);
                Ray3f ray = si.spawn_ray(ls.ray.d);
                PreliminaryIntersection3f pi_after_skip =
                    Base::skip_area_emitters(scene, ray, true, skip_emitters);
                dr::masked(ls.pi, skip_emitters) = pi_after_skip;
            }
        }

        dr::tie(ls) = dr::while_loop(
            dr::make_tuple(ls),
            [](const LoopState &ls) { return ls.active; },
            [this, scene](LoopState &ls) {
                
                // 计算 Surface Interaction
                SurfaceInteraction3f si = ls.pi.compute_surface_interaction(ls.ray, +RayFlags::All);
                
                // 击中 Mask
                Mask is_hit = si.is_valid();
                // 未击中 Mask (且当前路径仍然活跃)
                Mask is_miss = !is_hit && ls.active;

                // =========================================================
                // 新增模块：未击中时生成高斯噪声背景
                // =========================================================
                if (dr::any_or<true>(is_miss)) {
                    // Box-Muller 变换辅助 lambda
                    auto generate_noise_channel = [&](Mask mask) -> Float {
                        Float u1 = ls.sampler->next_1d(mask);
                        Float u2 = ls.sampler->next_1d(mask);
                        u1 = dr::maximum(u1, 1e-6f); // 避免 log(0)
                        return dr::sqrt(-2.f * dr::log(u1)) * dr::cos(2.f * dr::Pi<Float> * u2);
                    };

                    Spectrum noise_val;
                    if constexpr (is_rgb_v<Spectrum>) {
                        // RGB 模式：生成三个独立的噪声
                        Float r = generate_noise_channel(is_miss);
                        Float g = generate_noise_channel(is_miss);
                        Float b = generate_noise_channel(is_miss);
                        noise_val = Spectrum(r, g, b);
                    } else {
                        // 单色模式
                        noise_val = Spectrum(generate_noise_channel(is_miss));
                    }

                    // 计算背景颜色: Mean + Sigma * Noise
                    Spectrum bg = m_bg_color + Spectrum(m_bg_sigma) * noise_val;

                    // 累加到结果中 (Result += Throughput * Background)
                    // 注意使用 masked assignment，只更新 miss 的部分
                    dr::masked(ls.result, is_miss) = spec_fma(ls.throughput, bg, ls.result);
                    
                    // 标记这些光线为有效（因为我们要输出背景色，而不是纯黑）
                    ls.valid_ray |= is_miss;
                }
                // =========================================================

                // ---------------------- 直接发光 (Geometry Emitters) ----------------------
                // 只有击中物体(is_hit)才计算这部分
                EmitterPtr emitter = si.emitter(scene);
                Mask has_emitter = is_hit && (emitter != nullptr);
                
                if (dr::any_or<true>(has_emitter)) {
                    Mask use_emission = !m_hide_emitters || (ls.depth > 0u);
                    use_emission &= has_emitter; // 确保确实有光源

                    if (dr::any_or<true>(use_emission)) {
                        Spectrum Le = emitter->eval(si, true);
                        dr::masked(ls.result, use_emission) = spec_fma(ls.throughput, Le, ls.result);
                        ls.valid_ray |= use_emission;
                    }
                }

                // ---------------------- 决定是否继续追踪 ----------------------
                // 只有击中了物体，且没达到最大深度，才继续
                Bool active_next = (ls.depth + 1 < m_max_depth) && is_hit;

                if (dr::none_or<false>(active_next)) {
                    ls.active = active_next;
                    return;
                }

                // ---------------------- BSDF 采样 & 均匀半球逻辑 (保持原样) ----------------------
                BSDFPtr bsdf = si.bsdf(ls.ray);
                Point2f sample = ls.sampler->next_2d(active_next);
                Vector3f wo = warp::square_to_uniform_hemisphere(sample);

                Float pdf = warp::square_to_uniform_hemisphere_pdf(wo);
                Float cos_theta = wo.z();

                Mask valid_dir = (pdf > 0.f) && (cos_theta > 0.f);

                Spectrum bsdf_val = bsdf->eval(BSDFContext{}, si, wo, active_next);
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                Spectrum bsdf_weight = dr::zeros<Spectrum>();
                Mask update_mask = active_next && valid_dir;
                
                if (dr::any_or<true>(update_mask)) {
                    Float inv_pdf = dr::rcp(pdf);
                    Spectrum w = bsdf_val * (cos_theta * inv_pdf);
                    dr::masked(bsdf_weight, update_mask) = w;
                }

                ls.throughput *= bsdf_weight;
                ls.eta *= 1.f;
                
                // RR (Russian Roulette)
                dr::masked(ls.depth, is_hit) += 1; // 只有 hit 才增加深度

                Float throughput_max = dr::max(unpolarized_spectrum(ls.throughput));
                Float rr_prob = dr::minimum(throughput_max * dr::square(ls.eta), .95f);
                Mask rr_active = ls.depth >= m_rr_depth;
                Mask rr_continue = ls.sampler->next_1d(active_next) < rr_prob;
                
                dr::masked(ls.throughput, rr_active && active_next) *= dr::rcp(dr::detach(rr_prob));
                
                ls.active = active_next && (!rr_active || rr_continue) && (throughput_max != 0.f);

                // 生成新光线
                Vector3f wo_world = si.to_world(wo);
                ls.ray = si.spawn_ray(wo_world);
                ls.pi = scene->ray_intersect_preliminary(ls.ray, false, jit_flag(JitFlag::LoopRecord), 0, 0, ls.active);
            });

        return {
            dr::select(ls.valid_ray, ls.result, 0.f),
            ls.valid_ray
        };
    }

    std::string to_string() const override {
        return tfm::format("UniformHemispherePathIntegrator[\n"
                           "  max_depth = %u,\n"
                           "  rr_depth = %u,\n"
                           "  bg_mean = %s,\n"
                           "  bg_sigma = %f\n"
                           "]",
                           m_max_depth, m_rr_depth, m_bg_color, m_bg_sigma);
    }

    /// 辅助函数
    Spectrum spec_fma(const Spectrum &a, const Spectrum &b, const Spectrum &c) const {
        if constexpr (is_polarized_v<Spectrum>) return a * b + c;
        else return dr::fmadd(a, b, c);
    }

    MI_DECLARE_CLASS(UniformHemispherePathIntegrator)
};

MI_EXPORT_PLUGIN(UniformHemispherePathIntegrator)
NAMESPACE_END(mitsuba)
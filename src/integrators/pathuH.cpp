#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>          // 均匀半球采样
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * 非重要性采样的简单路径追踪器：
 * - 从相机发射光线
 * - 每次击中表面：
 *   - 如果是光源： result += throughput * L_e
 *   - 然后在局部法线半球上做均匀采样，方向 wo
 *   - throughput *= bsdf(wo) * cosθ / pdf_uniform
 *   - 用该方向生成下一条 ray
 * - 没有 NEE，没有 MIS
 */

template <typename Float, typename Spectrum>
class UniformHemispherePathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    UniformHemispherePathIntegrator(const Properties &props) : Base(props) { }

    std::pair<Spectrum, Bool> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Bool active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        if (unlikely(m_max_depth == 0))
            return { 0.f, false };

        // --------------------- 初始状态 ----------------------

        Ray3f ray                     = Ray3f(ray_);
        Spectrum throughput           = 1.f;
        Spectrum result               = 0.f;
        Float eta                     = 1.f;    // 这里只用在 RR 里，保持 1 即可
        PreliminaryIntersection3f pi  = dr::zeros<PreliminaryIntersection3f>();
        UInt32 depth                  = 0;

        // 如果有 environment emitter，就先认为“有有效光线”
        Mask valid_ray = !m_hide_emitters && (scene->environment() != nullptr);

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
            ray,
            pi,
            throughput,
            result,
            eta,
            depth,
            valid_ray,
            active,
            sampler
        };

        // 首跳一般是 coherent 的，不做重排
        ls.pi = scene->ray_intersect_preliminary(ls.ray,
                                                 /* coherent = */ true,
                                                 /* reorder = */ false,
                                                 /* reorder_hint = */ 0,
                                                 /* reorder_hint_bits = */ 0,
                                                 ls.active);

        // ---------------------- Hide area emitters ----------------------
        if (m_hide_emitters && dr::any_or<true>(ls.depth == 0u)) {
            Mask skip_emitters = ls.pi.is_valid() &&
                                 (ls.pi.shape->emitter() != nullptr) &&
                                 ls.active;

            if (dr::any_or<true>(skip_emitters)) {
                SurfaceInteraction3f si = ls.pi.compute_surface_interaction(
                    ls.ray, +RayFlags::Minimal, skip_emitters);
                Ray3f ray = si.spawn_ray(ls.ray.d);
                PreliminaryIntersection3f pi_after_skip =
                    Base::skip_area_emitters(scene, ray, true, skip_emitters);
                dr::masked(ls.pi, skip_emitters) = pi_after_skip;
            }
        }

        // ---------------------- 主循环：随机行走 ----------------------

        dr::tie(ls) = dr::while_loop(
            dr::make_tuple(ls),
            [](const LoopState &ls) { return ls.active; },
            [this, scene](LoopState &ls) {

                // 填充完整交点信息
                SurfaceInteraction3f si =
                    ls.pi.compute_surface_interaction(ls.ray, +RayFlags::All);

                // ---------------------- 直接发光（真正打到 emitter） ----------------------

                EmitterPtr emitter = si.emitter(scene);
                if (dr::any_or<true>(emitter != nullptr)) {
                    // hide_emitters：如果希望隐藏直接可见光源，可以在 depth == 0 时忽略
                    Mask use_emission = !m_hide_emitters || (ls.depth > 0u);

                    if (dr::any_or<true>(use_emission)) {
                        Spectrum Le = emitter->eval(si, true);
                        ls.result = spec_fma(ls.throughput, Le, ls.result);
                        ls.valid_ray |= use_emission;
                    }
                }

                // ---------------------- 是否继续追踪？ ----------------------

                Bool active_next = (ls.depth + 1 < m_max_depth) && si.is_valid();

                if (dr::none_or<false>(active_next)) {
                    ls.active = active_next;
                    // 如果这是一个有效的光源命中（且没有 hide_emitters），标记 valid_ray
                    ls.valid_ray |= (emitter != nullptr) && !m_hide_emitters;
                    return; // scalar 情况会在这里提前退出
                }

                // ---------------------- 取得 BSDF ----------------------

                BSDFPtr bsdf = si.bsdf(ls.ray);

                // ---------------------- 均匀半球采样方向 ----------------------

                Point2f sample = ls.sampler->next_2d();
                Vector3f wo = warp::square_to_uniform_hemisphere(sample); // 局部坐标系下

                // 半球均匀 pdf = 1 / (2π)，这里用 Mitsuba 内置 warp 函数
                Float pdf = warp::square_to_uniform_hemisphere_pdf(wo);
                Float cos_theta = wo.z();  // 半球局部坐标中，z 就是 cosθ

                Mask valid_dir = (pdf > 0.f) && (cos_theta > 0.f);

                // ---------------------- 评估 BSDF，并用 cosθ / pdf 更新 throughput ----------------------

                Spectrum bsdf_val = bsdf->eval(BSDFContext{}, si, wo, active_next);
                // 转到世界坐标的 Mueller / Stokes 基底
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                Spectrum bsdf_weight = dr::zeros<Spectrum>();
                if (dr::any_or<true>(valid_dir)) {
                    Float inv_pdf = dr::rcp(pdf);
                    Spectrum w = bsdf_val * (cos_theta * inv_pdf);
                    dr::masked(bsdf_weight, valid_dir) = w;
                }

                // 更新 throughput
                ls.throughput *= bsdf_weight;

                // 简单起见，这里不处理折射，eta 保持 1
                ls.eta *= 1.f;

                // 如果当前顶点是有效的，并且这次采样方向有效，则认为有有效路径
                ls.valid_ray |= ls.active && si.is_valid() && valid_dir;

                // ---------------------- Russian roulette 停止条件 ----------------------

                dr::masked(ls.depth, si.is_valid()) += 1;

                Float throughput_max = dr::max(unpolarized_spectrum(ls.throughput));
                Float rr_prob = dr::minimum(throughput_max * dr::square(ls.eta), .95f);

                Mask rr_active   = ls.depth >= m_rr_depth;
                Mask rr_continue = ls.sampler->next_1d() < rr_prob;

                // 对于可导版本，需要 detach rr_prob；这里保持 Mitsuba 原来的写法
                ls.throughput[rr_active] *= dr::rcp(dr::detach(rr_prob));

                ls.active = active_next && (!rr_active || rr_continue) &&
                            (throughput_max != 0.f);

                // ---------------------- 生成下一条 ray 并求交 ----------------------

                // 从局部方向 wo 转回世界空间
                Vector3f wo_world = si.to_world(wo);
                ls.ray = si.spawn_ray(wo_world);

                ls.pi = scene->ray_intersect_preliminary(ls.ray,
                                                         /* coherent = */ false,
                                                         /* reorder = */ jit_flag(JitFlag::LoopRecord),
                                                         /* reorder_hint = */ 0,
                                                         /* reorder_hint_bits = */ 0,
                                                         ls.active);
            });

        return {
            /* spec  = */ dr::select(ls.valid_ray, ls.result, 0.f),
            /* valid = */ ls.valid_ray
        };
    }

    std::string to_string() const override {
        return tfm::format("UniformHemispherePathIntegrator[\n"
                           "  max_depth = %u,\n"
                           "  rr_depth = %u\n"
                           "]",
                           m_max_depth, m_rr_depth);
    }

    /// 和原始 path 一样：极化模式下做 Mueller 乘法，否则做 fused-madd
    Spectrum spec_fma(const Spectrum &a, const Spectrum &b,
                      const Spectrum &c) const {
        if constexpr (is_polarized_v<Spectrum>)
            return a * b + c;
        else
            return dr::fmadd(a, b, c);
    }

    MI_DECLARE_CLASS(UniformHemispherePathIntegrator)
};

// 插件名你可以改成自己喜欢的，比如 "path_uhemi"
MI_EXPORT_PLUGIN(UniformHemispherePathIntegrator)
NAMESPACE_END(mitsuba)
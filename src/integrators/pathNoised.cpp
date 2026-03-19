#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/core/warp.h>
#include <drjit/math.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-path:

Path tracer (:monosp:`path`)
----------------------------

.. pluginparameters::

 * - max_depth
   - |int|
   - Specifies the longest path depth in the generated output image (where -1
     corresponds to :math:`\infty`). A value of 1 will only render directly
     visible light sources. 2 will lead to single-bounce (direct-only)
     illumination, and so on. (Default: -1)

 * - rr_depth
   - |int|
   - Specifies the path depth, at which the implementation will begin to use
     the *russian roulette* path termination criterion. For example, if set to
     1, then path generation may randomly cease after encountering directly
     visible surfaces. (Default: 5)

 * - hide_emitters
   - |bool|
   - Hide directly visible emitters. (Default: no, i.e. |false|)

This integrator implements a basic path tracer and is a **good default choice**
when there is no strong reason to prefer another method.

To use the path tracer appropriately, it is instructive to know roughly how
it works: its main operation is to trace many light paths using *random walks*
starting from the sensor. A single random walk is shown below, which entails
casting a ray associated with a pixel in the output image and searching for
the first visible intersection. A new direction is then chosen at the intersection,
and the ray-casting step repeats over and over again (until one of several
stopping criteria applies).

.. image:: ../../resources/data/docs/images/integrator/integrator_path_figure.png
    :width: 95%
    :align: center

At every intersection, the path tracer tries to create a connection to
the light source in an attempt to find a *complete* path along which
light can flow from the emitter to the sensor. This of course only works
when there is no occluding object between the intersection and the emitter.

This directly translates into a category of scenes where a path tracer can be
expected to produce reasonable results: this is the case when the emitters are
easily "accessible" by the contents of the scene. For instance, an interior
scene that is lit by an area light will be considerably harder to render when
this area light is inside a glass enclosure (which effectively counts as an
occluder).

Like the :ref:`direct <integrator-direct>` plugin, the path tracer internally
relies on multiple importance sampling to combine BSDF and emitter samples. The
main difference in comparison to the former plugin is that it considers light
paths of arbitrary length to compute both direct and indirect illumination.

.. note:: This integrator does not handle participating media

.. tabs::
    .. code-tab::  xml
        :name: path-integrator

        <integrator type="path">
            <integer name="max_depth" value="8"/>
        </integrator>

    .. code-tab:: python

        'type': 'path',
        'max_depth': 8

 */

template <typename Float, typename Spectrum>
class PathNoisedIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)
    Spectrum m_bg_color;
    Float m_bg_sigma;
    bool m_noised_emitter;
    bool m_use_uniform_sampling;

    PathNoisedIntegrator(const Properties &props) : Base(props) {
        // 读取噪声与背景设置
        m_bg_sigma = props.get<float>("sigma", 0.1f);
        m_use_uniform_sampling = props.get<bool>("use_uniform_sampling", false);
        m_noised_emitter = props.get<bool>("noised_emitter", false);

        if constexpr (is_rgb_v<Spectrum>) {
            using ScalarColor3f = mitsuba::Color<float, 3>;
            m_bg_color = Spectrum(props.get<ScalarColor3f>("bg_color", ScalarColor3f(0.0f)));
        } else {
            m_bg_color = Spectrum(props.get<float>("bg_color", 0.0f));
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

        // --------------------- Configure loop state ----------------------

        Ray3f ray                     = Ray3f(ray_);
        Spectrum throughput           = 1.f;
        Spectrum result               = 0.f;
        Float eta                     = 1.f;
        PreliminaryIntersection3f pi  = dr::zeros<PreliminaryIntersection3f>();
        UInt32 depth                  = 0;

        // If m_hide_emitters == false, the environment emitter will be visible
        Mask valid_ray = !m_hide_emitters && (scene->environment() != nullptr);

        // Variables caching information from the previous bounce
        Interaction3f prev_si         = dr::zeros<Interaction3f>();
        Float         prev_bsdf_pdf   = 1.f;
        Bool          prev_bsdf_delta = true;
        BSDFContext   bsdf_ctx;

        /* Set up a Dr.Jit loop. This optimizes away to a normal loop in scalar
           mode, and it generates either a megakernel (default) or
           wavefront-style renderer in JIT variants. This can be controlled by
           passing the '-W' command line flag to the mitsuba binary or
           enabling/disabling the JitFlag.LoopRecord bit in Dr.Jit.
        */
        struct LoopState {
            Ray3f ray;
            PreliminaryIntersection3f pi;
            Spectrum throughput;
            Spectrum result;
            Float eta;
            UInt32 depth;
            Mask valid_ray;
            Interaction3f prev_si;
            Float prev_bsdf_pdf;
            Bool prev_bsdf_delta;
            Bool active;
            Sampler* sampler;

            DRJIT_STRUCT(LoopState, ray, pi, throughput, result, eta, depth, \
                valid_ray, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                active, sampler)
        } ls = {
            ray,
            pi,
            throughput,
            result,
            eta,
            depth,
            valid_ray,
            prev_si,
            prev_bsdf_pdf,
            prev_bsdf_delta,
            active,
            sampler
        };

        // First bounce is usually coherent - don't reorder threads
        ls.pi = scene->ray_intersect_preliminary(ls.ray,
                                                 /* coherent = */ true,
                                                 /* reorder = */ false,
                                                 /* reorder_hint = */ 0,
                                                 /* reorder_hint_bits = */ 0,
                                                 ls.active);

        // ---------------------- Hide area emitters ----------------------

        /* dr::any_or() checks for active entries in the provided boolean
           array. JIT/Megakernel modes can't do this test efficiently as
           each Monte Carlo sample runs independently. In this case,
           dr::any_or<..>() returns the template argument (true) which means
           that the 'if' statement is always conservatively taken. */

        if (m_hide_emitters && dr::any_or<true>(ls.depth == 0u)) {
            // Did we hit an area emitter? If so, skip all area emitters along this ray
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

        dr::tie(ls) = dr::while_loop(dr::make_tuple(ls),
            [](const LoopState& ls) { return ls.active; },
            [this, scene, bsdf_ctx](LoopState& ls) {

            /* dr::while_loop implicitly masks all code in the loop using the
               'active' flag, so there is no need to pass it to every function */

            // Fill out all information of the interaction
            SurfaceInteraction3f si =
                ls.pi.compute_surface_interaction(ls.ray, +RayFlags::All);
            
            auto generate_noise_channel = [&](Mask mask) -> Float {
            Float u1 = dr::maximum(ls.sampler->next_1d(mask), 1e-6f);
            Float u2 = ls.sampler->next_1d(mask);
            return dr::sqrt(-2.f * dr::log(u1)) * dr::cos(2.f * dr::Pi<Float> * u2);
            };

            Spectrum noise_val;
            if constexpr (is_rgb_v<Spectrum>) {
                noise_val = Spectrum(generate_noise_channel(ls.active),
                                        generate_noise_channel(ls.active),
                                        generate_noise_channel(ls.active));
            } else {
                noise_val = Spectrum(generate_noise_channel(ls.active));
            }
            
            // ------------------ New: add noise --------------------------
            Mask is_miss = !si.is_valid() && ls.active;
            if (dr::any_or<true>(is_miss)) {

                Spectrum bg_addition = m_bg_color + Spectrum(m_bg_sigma) * noise_val;
                
                // 将噪声与底色累加到当前路径结果中
                dr::masked(ls.result, is_miss) = spec_fma(ls.throughput, bg_addition, ls.result);
                ls.valid_ray |= is_miss;
            }

            // ---------------------- Direct emission ----------------------

            if (dr::any_or<true>(si.emitter(scene) != nullptr)) {
                DirectionSample3f ds(scene, si, ls.prev_si);
                Float em_pdf = 0.f;

                if (dr::any_or<true>(!ls.prev_bsdf_delta))
                    em_pdf = scene->pdf_emitter_direction(ls.prev_si, ds,
                                                          !ls.prev_bsdf_delta);

                // Compute MIS weight for emitter sample from previous bounce
                Float mis_bsdf = mis_weight(ls.prev_bsdf_pdf, em_pdf);
                
                Spectrum em_val = ds.emitter->eval(si, ls.prev_bsdf_pdf > 0.f);
                if (m_noised_emitter) {
                    Spectrum em_noise = noise_val * m_bg_sigma;
                    em_val += em_noise; 
                }
                // Accumulate, being careful with polarization (see spec_fma)
                ls.result = spec_fma(
                    ls.throughput,
                    em_val * mis_bsdf,
                    ls.result);
            }

            // Continue tracing the path at this point?
            Bool active_next = (ls.depth + 1 < m_max_depth) && si.is_valid();

            if (dr::none_or<false>(active_next)) {
                ls.active = active_next;
                ls.valid_ray |= (si.emitter(scene) != nullptr) && !m_hide_emitters;
                return; // early exit for scalar mode
            }

            BSDFPtr bsdf = si.bsdf(ls.ray);

            // ---------------------- Emitter sampling ----------------------

            // Perform emitter sampling?
            Mask active_em = active_next && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            DirectionSample3f ds = dr::zeros<DirectionSample3f>();
            Spectrum em_weight = dr::zeros<Spectrum>();
            Vector3f wo = dr::zeros<Vector3f>();

            if (dr::any_or<true>(active_em)) {
                // Sample the emitter
                std::tie(ds, em_weight) = scene->sample_emitter_direction(
                    si, ls.sampler->next_2d(), true, active_em);
                active_em &= (ds.pdf != 0.f);

                if (m_noised_emitter) {
                    // 注意：em_weight 已经是 (Radiance / pdf) 的结果了。
                    // 我们直接在蒙特卡洛估计量上加噪声，防止噪声被极端微小的 pdf 放大成无穷大。
                    Spectrum em_noise = noise_val * m_bg_sigma;
                    dr::masked(em_weight, active_em) += em_noise;
                }

                /* Given the detached emitter sample, recompute its contribution
                   with AD to enable light source optimization. */
                if (dr::grad_enabled(si.p)) {
                    ds.d = dr::normalize(ds.p - si.p);
                    Spectrum em_val = scene->eval_emitter_direction(si, ds, active_em);
                    em_weight = dr::select(ds.pdf != 0, em_val / ds.pdf, 0);
                }

                wo = si.to_local(ds.d);
            }

            // ------ Evaluate BSDF * cos(theta) and sample direction -------
            Float sample_1 = ls.sampler->next_1d();
            Point2f sample_2 = ls.sampler->next_2d();

            // 初始化，防止 dr::masked 报错
            Spectrum bsdf_val = dr::zeros<Spectrum>();
            Float bsdf_pdf = 0.f; 
            BSDFSample3f bsdf_sample = dr::zeros<BSDFSample3f>();
            Spectrum bsdf_weight = dr::zeros<Spectrum>();

            if (m_use_uniform_sampling) {
                // 评估光源方向的 BSDF，获取原始的 pdf_em
                auto [val_em, pdf_em] = bsdf->eval_pdf(bsdf_ctx, si, wo, active_next);
                bsdf_val = val_em;
                
                Bool has_delta = has_flag(bsdf->flags(), BSDFFlags::Delta);
                Mask use_uniform       = active_next && !has_delta;
                Mask use_delta_fallback = active_next && has_delta;

                // [核心修正 A：修复 MIS 权重失衡导致的偏蓝]
                // 既然我们强制把 BSDF 策略换成了均匀采样，那么“从该策略生成 wo”的真实概率就是 1/(2*PI)。
                // 我们必须向 MIS 权重公式提供这个真实的 PDF，否则能量不守恒导致严重偏色。
                Float uni_pdf_for_mis = warp::square_to_uniform_hemisphere_pdf(wo); 
                // 只有入射和出射在同一半球（反射）时，PDF才有效
                uni_pdf_for_mis = dr::select(si.wi.z() * wo.z() > 0.f, uni_pdf_for_mis, 0.f);
                // 替换掉原来的 pdf_em
                bsdf_pdf = dr::select(use_uniform, uni_pdf_for_mis, pdf_em);

                // --- 分支 A: 可以进行均匀半球采样的材质 ---
                if (dr::any_or<true>(use_uniform)) {
                    Vector3f local_wo = warp::square_to_uniform_hemisphere(sample_2);
                    
                    // [核心修正 B：修复背面变黑/丢失间接光的问题]
                    // Mitsuba 的 twosided 材质允许光线打在背面 (si.wi.z < 0)。
                    // 均匀半球采样默认产生 z > 0 的方向。如果打在背面，我们需要强制把生成的方向翻转到下半球。
                    local_wo.z() = dr::select(si.wi.z() < 0.f, -local_wo.z(), local_wo.z());

                    // 翻转后，不管在哪个半球，概率密度依然是 1/(2*PI)
                    Float uni_pdf = warp::square_to_uniform_hemisphere_pdf(local_wo); 
                    
                    dr::masked(bsdf_sample.wo, use_uniform) = local_wo;
                    dr::masked(bsdf_sample.pdf, use_uniform) = uni_pdf;
                    dr::masked(bsdf_sample.eta, use_uniform) = 1.f;
                    dr::masked(bsdf_sample.sampled_type, use_uniform) = +BSDFFlags::Smooth; 
                    
                    Spectrum bsdf_val_next = bsdf->eval(bsdf_ctx, si, local_wo, use_uniform);
                    dr::masked(bsdf_weight, use_uniform) = dr::select(uni_pdf > 0.f, bsdf_val_next / uni_pdf, 0.f);
                }

                // --- 分支 B: 纯镜面/玻璃材质 (Delta) 的保护机制 ---
                if (dr::any_or<true>(use_delta_fallback)) {
                    auto [delta_val, delta_pdf, delta_sample, delta_weight] = 
                        bsdf->eval_pdf_sample(bsdf_ctx, si, wo, sample_1, sample_2, use_delta_fallback);
                    
                    dr::masked(bsdf_sample, use_delta_fallback) = delta_sample;
                    dr::masked(bsdf_weight, use_delta_fallback) = delta_weight;
                }

            } else {
                // [原版] 重点采样 (Importance Sampling)
                std::tie(bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight)
                    = bsdf->eval_pdf_sample(bsdf_ctx, si, wo, sample_1, sample_2, active_next);
            }


            // --------------- Emitter sampling contribution ----------------

            if (dr::any_or<true>(active_em)) {
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                // Compute the MIS weight
                Float mis_em =
                    dr::select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                // Accumulate, being careful with polarization (see spec_fma)
                ls.result[active_em] = spec_fma(
                    ls.throughput, bsdf_val * em_weight * mis_em, ls.result);
            }

            // ---------------------- BSDF sampling ----------------------

            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);

            ls.ray = si.spawn_ray(si.to_world(bsdf_sample.wo));

            /* When the path tracer is differentiated, we must be careful that
               the generated Monte Carlo samples are detached (i.e. don't track
               derivatives) to avoid bias resulting from the combination of moving
               samples and discontinuous visibility. We need to re-evaluate the
               BSDF differentiably with the detached sample in that case. */
            if (dr::grad_enabled(ls.ray)) {
                ls.ray = dr::detach<true>(ls.ray);

                // Recompute 'wo' to propagate derivatives to cosine term
                Vector3f wo_2 = si.to_local(ls.ray.d);
                auto [bsdf_val_2, bsdf_pdf_2] = bsdf->eval_pdf(bsdf_ctx, si, wo_2, ls.active);
                bsdf_weight[bsdf_pdf_2 > 0.f] = bsdf_val_2 / dr::detach(bsdf_pdf_2);
            }

            // ------ Update loop variables based on current interaction ------

            ls.throughput *= bsdf_weight;
            ls.eta *= bsdf_sample.eta;
            ls.valid_ray |= ls.active && si.is_valid() &&
                         !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);

            // Information about the current vertex needed by the next iteration
            ls.prev_si = Interaction3f(si);
            ls.prev_bsdf_pdf = bsdf_sample.pdf;
            ls.prev_bsdf_delta = has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta);

            // -------------------- Stopping criterion ---------------------

            dr::masked(ls.depth, si.is_valid()) += 1;

            Float throughput_max = dr::max(unpolarized_spectrum(ls.throughput));

            Float rr_prob = dr::minimum(throughput_max * dr::square(ls.eta), .95f);
            Mask rr_active = ls.depth >= m_rr_depth,
                 rr_continue = ls.sampler->next_1d() < rr_prob;

            /* Differentiable variants of the renderer require the russian
               roulette sampling weight to be detached to avoid bias. This is a
               no-op in non-differentiable variants. */
            ls.throughput[rr_active] *= dr::rcp(dr::detach(rr_prob));

            ls.active = active_next && (!rr_active || rr_continue) &&
                        (throughput_max != 0.f);

            // Reorder threads based on the shape they hit
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

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("PathNoisedIntegrator[\n"
            "  max_depth = %u,\n"
            "  rr_depth = %u\n"
            "]", m_max_depth, m_rr_depth);
    }

    /// Compute a multiple importance sampling weight using the power heuristic
    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::detach<true>(dr::select(dr::isfinite(w), w, 0.f));
    }

    /**
     * \brief Perform a Mueller matrix multiplication in polarized modes, and a
     * fused multiply-add otherwise.
     */
    Spectrum spec_fma(const Spectrum &a, const Spectrum &b,
                      const Spectrum &c) const {
        if constexpr (is_polarized_v<Spectrum>)
            return a * b + c;
        else
            return dr::fmadd(a, b, c);
    }

    MI_DECLARE_CLASS(PathNoisedIntegrator)
};

MI_EXPORT_PLUGIN(PathNoisedIntegrator)
NAMESPACE_END(mitsuba)

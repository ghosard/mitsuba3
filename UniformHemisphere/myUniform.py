# uniform_hemisphere_integrator.py
# 最小示例：自定义一个只做一次均匀半球采样的 integrator，
# 用在你现在的 cbox.xml 上。

import mitsuba as mi
import drjit as dr

# 用你现在在用的变体
mi.set_variant("cuda_ad_rgb")   # 或者 "llvm_ad_rgb" 看你环境

class UniformHemisphere(mi.SamplingIntegrator):
    def __init__(self, props=mi.Properties()):
        # 这里 props 主要是为了和 mitsuba 的接口兼容，
        # 没有额外参数就直接传进去即可
        mi.SamplingIntegrator.__init__(self, props)
        # 可以留一个 max_depth 接口，后面你要多 bounce 再扩展
        self.max_depth = 1

    def sample(self,
               scene: mi.Scene,
               sampler,
               ray: mi.RayDifferential3f,
               medium: mi.Medium = None,
               active: bool = True):

        # 确保用的是 Ray3f（对 cuda_ad_rgb 来说是对的）
        ray = mi.Ray3f(ray)

        # 和普通 integrator 一样，先求一次主光线和场景的交点
        si = scene.ray_intersect(ray, active)
        active = active & si.is_valid()

        # 初始化输出 radiance
        L = mi.Color3f(0.0)

        emitter_primary = si.emitter(scene, active)
        null_emitter    = type(emitter_primary)()
        hit_primary     = active & (emitter_primary != null_emitter)

        # 如果主光线什么都没打到，直接返回黑色
        # 注意不能用 Python 的 if(active)，所以用 dr.select
        L += dr.select(
            hit_primary,
            mi.Emitter.eval_vec(emitter_primary, si, hit_primary),
            mi.Color3f(0.0)
        )

        # 对于 hit 的像素，我们做一次均匀半球采样
        # -------------------------------------------------
        # 从采样器拿一个 2D 随机数
        u = sampler.next_2d(active)

        # 在局部坐标系下做均匀半球采样（完全不考虑 BSDF）
        wo_local = mi.warp.square_to_uniform_hemisphere(u)
        # 对应的 cos(theta)，局部坐标里的 z 分量就是法线方向
        cos_theta = wo_local.z

        # 把局部方向变成世界坐标
        wo_world = si.to_world(wo_local)

        # 从交点发出第二条 ray（避免自交，用 spawn_ray）
        ray2 = si.spawn_ray(wo_world)

        # 再和场景求交，看这条方向上有没有打到东西
        si2 = scene.ray_intersect(ray2, active)
        hit2 = active & si2.is_valid()

        # 这里为了示例，把“只要打到任意表面”都当成恒定白色光源 Li = 1
        # 真正做 path tracing 的时候，你可以在这里用 si2.emitter(...) 去 eval 光源
        Li = mi.Color3f(dr.select(hit2, 1.0, 0.0))

        # 均匀半球的 pdf = 1 / (2π)
        pdf = 1.0 / (2.0 * dr.pi)

        # 最简单的 MC 估计：L = Li * cos(theta) / pdf
        # 这里完全忽略了 BSDF，只是展示“方向来自均匀半球采样”的框架
        contrib = Li * (cos_theta / pdf)

        # 把无效像素清零
        L = dr.select(active, contrib, mi.Color3f(0.0))

        # 返回值：颜色, 有效 mask, AOV 列表（这里暂时不输出 AOV）
        return L, active, []

    def to_string(self) -> str:
        return "UniformHemisphere[]"


# 注册成一个 integrator 插件（可选，但推荐）
mi.register_integrator("uniform_hemi",
                       lambda props: UniformHemisphere(props))


if __name__ == "__main__":
    # 直接用你现在的 cbox 场景
    scene = mi.load_file("./cbox.xml")

    # 取第一个 sensor
    sensor = scene.sensors()[0]

    # 创建我们自定义的 integrator 实例
    integrator = UniformHemisphere()

    # 用自定义 integrator 渲染，注意要把 integrator 作为参数传给 render
    image = mi.render(scene,
                      sensor=sensor,
                    #   integrator=integrator,
                      spp=1)   # 这里可以调你要的采样数

    # 存图
    mi.util.write_bitmap("cbox_uniform_hemi.png", image)
    print("Done: cbox_uniform_hemi.png")

# uniform_hemisphere_integrator.py
# 最小示例：自定义一个只做一次均匀半球采样的 integrator，
# 用在你现在的 cbox.xml 上。

import mitsuba as mi
import drjit as dr

# 用你现在在用的变体
mi.set_variant("cuda_ad_rgb")   # 或者 "llvm_ad_rgb" 看你环境

if __name__ == "__main__":
    # 直接用你现在的 cbox 场景
    scene = mi.load_file("./scenes/cbox.xml")

    # 取第一个 sensor
    sensor = scene.sensors()[0]

    # 创建我们自定义的 integrator 实例

    # 用自定义 integrator 渲染，注意要把 integrator 作为参数传给 render
    image = mi.render(scene,
                      sensor=sensor,
                    #   integrator=integrator,
                      spp=512)   # 这里可以调你要的采样数

    # 存图
    mi.util.write_bitmap("cbox_uniform_hemi.png", image)
    print("Done: cbox_uniform_hemi.png")

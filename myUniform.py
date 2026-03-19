import mitsuba as mi
from diffusers.utils import make_image_grid
import numpy as np
from PIL import Image

# 初始化变体
mi.set_variant("cuda_ad_rgb")

if __name__ == "__main__":
    # 加载基础场景
    scene = mi.load_file("scenes/teapot/teapot/scene_v3.xml")
    sensor = scene.sensors()[0]

    # 1. 动态创建两个 Integrator 实例
    # 你可以在这里直接传入你的自定义参数（例如 bg_color 和 sigma）
    int_uH = mi.load_dict({
        'type': 'pathuH',
        'max_depth': 6,
        # 'bg_color': [0.0, 0.0, 0.0], 
        # 'sigma': 1.0  
    })

    int_path = mi.load_dict({
        'type': 'path',
        'max_depth': 6,
    })
    
    int_uHN = mi.load_dict({
        'type': 'pathuHN',
        'max_depth': 6,
        'bg_color': [0.0, 0.0, 0.0],
        'sigma': 1.0
    })

    int_pathNoised = mi.load_dict({
        'type': 'pathNoised',
        'max_depth': 6,
        'bg_color': [0.0, 0.0, 0.0],
        'sigma': 1,
        'use_uniform_sampling': True,
        'noised_emitter': True,
    })

    # 用字典装起来方便遍历和打印
    integrators = {
        # "pathuH": int_uH,
        # "pathuHN": int_uHN,
        "path": int_path,
        "pathNoised": int_pathNoised,
    }

    all_images = []
    max_iter = 10
    # 2. 外层循环：遍历不同的 Integrator
    for name, integrator in integrators.items():
        print(f"Rendering with {name}...")
        
        # 内层循环：遍历不同的 SPP
        for i in range(max_iter):
            spp = 2 ** i
            print(f"  SPP: {spp}")
            
            # 将动态创建的 integrator 传给 render 函数
            image = mi.render(scene, sensor=sensor, integrator=integrator, spp=spp)
            
            # 转换和保存逻辑（用于可视化）
            bitmap = mi.util.convert_to_bitmap(image)
            
            # 注意：srgb_gamma=True 会做非线性提亮，适合人眼观察
            bitmap_rgba = bitmap.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)
            np_img = np.array(bitmap_rgba, copy=False)
            pil_image = Image.fromarray(np_img, mode='RGB')
            
            all_images.append(pil_image)

    # 3. 生成 2 行 5 列的网格
    grid = make_image_grid(all_images, rows= len(all_images) // max_iter, cols= max_iter)
    grid.save("cbox_integrator_comparison.jpg", quality=75)
    print("Done: saved to cbox_integrator_comparison.jpg")
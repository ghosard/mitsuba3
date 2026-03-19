import mitsuba as mi
import numpy as np
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET
from diffusers.utils import make_image_grid
import os
# 初始化变体
mi.set_variant("cuda_ad_rgb")
WIDTH = 512
HEIGHT = 512
CENTER_CROP = False
RESCALE_RESOLUTION = True   
OURPUT_DIR = "outputs"
def center_crop(img: Image.Image, crop_w=512, crop_h=512):
    """
    纯中心裁剪，不进行任何缩放。
    如果原图比裁剪尺寸小，边缘会超出范围（建议确保原图 > 512x512）。
    """
    w, h = img.size
    if w == crop_w and h == crop_h:
        return img
    if w < crop_w or h < crop_h:
        min_size = min(w, h)
        img = img.resize((min_size, min_size), resample=Image.Resampling.LANCZOS)
    
    # 使用 // 进行整数除法，PIL crop 需要整数坐标
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    right = (w + crop_w) // 2
    bottom = (h + crop_h) // 2
    
    return img.crop((left, top, right, bottom))

def load_scene_with_custom_res(xml_path_str, target_width, target_height):
    """
    通过解析 XML 树动态修改分辨率，并安全加载场景
    绝对不会修改原文件，纯内存操作
    """
    tree = ET.parse(xml_path_str)
    root = tree.getroot()

    # 精准遍历并修改 <film> 节点下的 width 和 height
    for film in root.iter('film'):
        for child in film:
            if child.tag == 'integer':
                if child.get('name') == 'width':
                    child.set('value', str(target_width))
                elif child.get('name') == 'height':
                    child.set('value', str(target_height))

    xml_str = ET.tostring(root, encoding='unicode')

    original_cwd = os.getcwd()
    scene_dir = str(Path(xml_path_str).parent.absolute())
    
    try:
        # 切换到场景所在目录，让 load_string 能找到相对路径下的 obj 和纹理
        os.chdir(scene_dir)
        scene = mi.load_string(xml_str)
    finally:
        # 无论加载成功与否（哪怕中途报错），必定安全切回原工作目录
        os.chdir(original_cwd)
        
    return scene

if __name__ == "__main__":
    # 1. 定义需要遍历的场景路径列表 (你可以随意添加更多场景)
    scene_paths = [
        "scenes/teapot/teapot/scene_v3.xml",
        "scenes/dragon/dragon/scene_v3.xml",
        "scenes/dining-room/dining-room/scene_v3.xml"
        # "scenes/cbox/cbox.xml",  # 示例：添加其他场景
    ]

    # 定义所有的 Integrator
    int_uH = mi.load_dict({'type': 'pathuH', 'max_depth': 6})
    int_path = mi.load_dict({'type': 'path', 'max_depth': 6})
    int_uHN = mi.load_dict({'type': 'pathuHN', 'max_depth': 6, 'bg_color': [0.0, 0.0, 0.0], 'sigma': 0.2})
    int_pathNoised = mi.load_dict({
        'type': 'pathNoised',
        'max_depth': 6,
        'bg_color': [0.0, 0.0, 0.0],
        'sigma': 0.5,
        'use_uniform_sampling': True,
        'noised_emitter': True,
    })
    int_pathNoisedNoUniform = mi.load_dict({
        'type': 'pathNoised',
        'max_depth': 6,
        'bg_color': [0.0, 0.0, 0.0],
        'sigma': 0.5,
        'use_uniform_sampling': False,
        'noised_emitter': True,
    })
    int_pathNoisedNoUniformNoEmitter = mi.load_dict({
        'type': 'pathNoised',
        'max_depth': 6,
        'bg_color': [0.0, 0.0, 0.0],
        'sigma': 0.5,
        'use_uniform_sampling': False,
        'noised_emitter': False,
    })

    # 用字典装起来方便遍历
    integrators = {
        # "pathuH": int_uH,
        # "pathuHN": int_uHN,
        # "path": int_path,
        # "pathNoised": int_pathNoised,
        # "pathNoisedNoUniform": int_pathNoisedNoUniform,
        "pathNoisedNoUniformNoEmitter": int_pathNoisedNoUniformNoEmitter,
    }

    max_iter = 10
    spp_list = [1,2,3,4,5,6,7,8,16,32,64,128,256,512,1024, 2048, 4096]
    # 2. 最外层循环：遍历 scene 列表
    for scene_path_str in scene_paths:
        print(f"\n========== Loading Scene: {scene_path_str} ==========")
        
        # 提取 scene_baseName (例如 "scenes/teapot/teapot/scene_v3.xml" 会提取出 "scene_v3")
        scene_baseName = f"{Path(scene_path_str).parent.name}_{Path(scene_path_str).stem}" 
        
        try:
            # 直接使用我们封装好的函数
            if RESCALE_RESOLUTION:
                scene = load_scene_with_custom_res(scene_path_str, WIDTH, HEIGHT)
            else:
                scene = mi.load_file(scene_path_str)
            sensor = scene.sensors()[0]
        except Exception as e:
            print(f"[Error] Failed to load scene {scene_path_str}. Error: {e}")
            continue
        all_integrator_images = []
        # 3. 中层循环：遍历不同的 Integrator (Tracer)
        for tracer_name, integrator in integrators.items():
            print(f"  Rendering with Tracer: {tracer_name}...")
            
            # 构建输出目录路径：output/scene_baseName/tracer_name/
            # exist_ok=True 保证如果文件夹已存在不会报错，parents=True 保证自动创建多级父目录
            if integrators.__len__() > 1:
                output_dir = Path(OURPUT_DIR) / scene_baseName / tracer_name
            else:
                output_dir = Path(OURPUT_DIR) / scene_baseName
            output_dir.mkdir(parents=True, exist_ok=True)
            total_images = []
            # 4. 内层循环：遍历不同的 SPP
            for spp in spp_list:
                print(f"    SPP: {spp}")
                
                # 执行渲染
                image = mi.render(scene, sensor=sensor, integrator=integrator, spp=spp)
                
                # 转换格式
                bitmap = mi.util.convert_to_bitmap(image)
                bitmap_rgba = bitmap.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)
                np_img = np.array(bitmap_rgba, copy=False)
                pil_image = Image.fromarray(np_img, mode='RGB')
                if CENTER_CROP:
                    pil_image = center_crop(pil_image)
                # 拼接完整的文件保存路径并保存
                save_path = output_dir / f"spp_{spp}.jpg"
                total_images.append(pil_image)
                all_integrator_images.append(pil_image)
                pil_image.save(save_path, quality=75)
            make_image_grid(total_images, rows=1, cols=len(spp_list)).save(output_dir / f"{tracer_name}_merged.jpg", quality=75)
        make_image_grid(all_integrator_images, rows=len(integrators), cols=len(spp_list)).save( Path(OURPUT_DIR) / scene_baseName / f"{scene_baseName}_merged.jpg", quality=75)
    print("\nAll rendering tasks completed!")
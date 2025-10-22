import os
from PIL import Image
from tqdm import tqdm

# 原图像目录
input_dir = "/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/dataset/coco2017/train2017_default_size"
# 输出图像目录
output_dir = "/hpc2ssd/JH_DATA/spooler/yuxuanzhao/lijungang/wujie/LLaDA-V/dataset/coco2017/train2017"
# 新的图像大小
target_size = (384, 384)

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 支持的图像扩展名
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# 遍历处理图像
for filename in tqdm(os.listdir(input_dir), desc="Resizing images"):
    ext = os.path.splitext(filename)[1].lower()
    if ext in image_extensions:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            with Image.open(input_path) as img:
                resized_img = img.resize(target_size, Image.BICUBIC)
                resized_img.save(output_path)
        except Exception as e:
            print(f"❌ 处理失败: {filename} - 错误: {e}")

print("✅ 所有图片已成功resize为 384x384。")

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

# ========== 路径配置（直接写全路径） ==========
img_dir = "/xx/xx/xx/xxr/xx/VOC2007/JPEGImages"
ann_dir = "/xx/xx/xx/xxr/xx/VOC2007/Annotations"

# ========== 阈值设定 ==========
DENSITY_THRESHOLD = 10        # 鱼体数量 >= 10 记为高密度
BLUR_THRESHOLD = 100          # Laplacian 方差 <= 100 判定模糊
BRIGHTNESS_THRESHOLD = 50     # 平均亮度 <= 50/255 判定过暗

# ========== 获取所有样本 ==========
image_ids = [f[:-4] for f in os.listdir(ann_dir) if f.endswith(".xml")]

# 统计变量
high_density_count = 0
low_quality_count = 0

# 存每张图片的统计信息
records = []

for img_id in tqdm(image_ids):
    # === 读取标注，统计鱼的数量 ===
    ann_path = os.path.join(ann_dir, img_id + ".xml")
    tree = ET.parse(ann_path)
    objs = tree.findall("object")
    num_fish = len(objs)

    high_density = num_fish >= DENSITY_THRESHOLD
    if high_density:
        high_density_count += 1

    # === 读取图像，计算清晰度 & 亮度 ===
    img_path = os.path.join(img_dir, img_id + ".jpg")
    img = cv2.imread(img_path)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 清晰度：Laplacian 方差
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 亮度：灰度均值
    brightness = np.mean(gray)

    low_quality = (fm <= BLUR_THRESHOLD) or (brightness <= BRIGHTNESS_THRESHOLD)
    if low_quality:
        low_quality_count += 1

    records.append({
        "image_id": img_id,
        "num_fish": num_fish,
        "laplacian_var": fm,
        "brightness": brightness,
        "is_high_density": high_density,
        "is_low_quality": low_quality
    })

# ========== 打印结果 ==========
total = len(image_ids)
print("总样本数:", total)
print("高密度鱼群样本数:", high_density_count,
      f"({high_density_count/total*100:.2f}%)")
print("低质量图像样本数:", low_quality_count,
      f"({low_quality_count/total*100:.2f}%)")

# ========== 保存到 CSV ==========
df = pd.DataFrame(records)
out_path = "/xx/xx/xx/xxr/xx/dataset_complexity_stats.csv"
df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"详细结果已保存到 {out_path}")

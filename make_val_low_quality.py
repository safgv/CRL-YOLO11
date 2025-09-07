import pandas as pd
import os

# === 路径配置 ===
CSV_PATH = "/xx/xx/xx/xxr/xx/dataset_complexity_stats.csv"  
OUT_PATH = "/xx/xx/xx/xxr/xx/val_low_quality.txt"

def main():
    # 1. 读取CSV
    df = pd.read_csv(CSV_PATH)
    if "image_id" not in df.columns or "is_low_quality" not in df.columns:
        raise ValueError("CSV 必须包含 'image_id' 和 'is_low_quality' 列")

    # 2. 筛选低质量子集
    low_quality_ids = df[df["is_low_quality"] == True]["image_id"].astype(str).tolist()
    print(f"[INFO] 低质量子集样本数: {len(low_quality_ids)}")

    # 3. 确保输出目录存在
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # 4. 写入 val_low_quality.txt
    with open(OUT_PATH, "w") as f:
        for iid in low_quality_ids:
            f.write(f"{iid}\n")

    print(f"[INFO] 已生成: {OUT_PATH}")

if __name__ == "__main__":
    main()

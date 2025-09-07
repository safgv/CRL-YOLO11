import os
import yaml
import glob
import pandas as pd
from pathlib import Path

# === 配置：你的数据配置与统计CSV ===
YAML_PATH = "/T2013054/hbq/Datasets/Fish.yaml"  # 你的数据集yaml
CSV_PATH  = "dataset_complexity_stats.csv"      # 包含 image_id 和 is_low_quality 的CSV
OUT_TXT   = "/xx/xx/xx/xxr/xx/val_low_quality.txt"  # 生成的可用val列表（写路径，不是ID）

# 可接受的图像扩展名
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_stem2path_from_dir(img_dir):
    """从目录扫描所有图片，构建 stem->path 的映射"""
    img_dir = Path(img_dir)
    stem2path = {}
    for ext in IMG_EXTS:
        for p in img_dir.rglob(f"*{ext}"):
            stem2path[p.stem] = str(p.resolve())
    return stem2path

def build_stem2path_from_txt(txt_path, root=None):
    """从已有的txt列表构建 stem->path 的映射"""
    stem2path = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            # 若有root，允许把相对路径拼到root上
            full = Path(root, p).resolve() if root and not os.path.isabs(p) else Path(p).resolve()
            if full.suffix.lower() in IMG_EXTS and full.exists():
                stem2path[full.stem] = str(full)
            else:
                # 也可能是目录或不带后缀的路径，尝试匹配常见后缀
                if not full.suffix and full.exists() and full.is_dir():
                    for ext in IMG_EXTS:
                        for fp in Path(full).rglob(f"*{ext}"):
                            stem2path[fp.stem] = str(fp.resolve())
                else:
                    # 若是相对路径且不存在，尝试不同扩展名
                    stem = full.stem
                    base = full.with_suffix("")
                    for ext in IMG_EXTS:
                        cand = Path(str(base) + ext)
                        if cand.exists():
                            stem2path[cand.stem] = str(cand.resolve())
                            break
    return stem2path

def main():
    data = load_yaml(YAML_PATH)
    root = data.get("path", None)
    val_entry = data.get("val", None)
    if val_entry is None:
        raise FileNotFoundError(f"YAML {YAML_PATH} 中未找到 'val' 字段，请检查。")

    # 读取CSV，筛低质量样本的 image_id（不带后缀）
    df = pd.read_csv(CSV_PATH)
    if "image_id" not in df.columns or "is_low_quality" not in df.columns:
        raise ValueError("CSV 必须包含 'image_id' 和 'is_low_quality' 列。")
    ids = df[df["is_low_quality"] == True]["image_id"].astype(str).tolist()
    ids_set = set(ids)
    print(f"[INFO] 低质量ID数: {len(ids_set)}")

    # 构建 stem->path 映射
    stem2path = {}
    val_path = Path(root, val_entry).resolve() if root and not os.path.isabs(str(val_entry)) else Path(val_entry).resolve()

    if val_path.is_dir():
        print(f"[INFO] val 是目录：{val_path}")
        stem2path = build_stem2path_from_dir(val_path)
    elif val_path.is_file():
        print(f"[INFO] val 是txt列表：{val_path}")
        stem2path = build_stem2path_from_txt(val_path, root=root)
    else:
        # 有的yaml会把 train/val 写成相对路径（相对 root），上面已处理；兜底再试试 root 下拼接
        maybe_dir = Path(root, val_entry) if root else Path(val_entry)
        if maybe_dir.exists() and maybe_dir.is_dir():
            print(f"[INFO] val 解析为目录：{maybe_dir}")
            stem2path = build_stem2path_from_dir(maybe_dir)
        elif maybe_dir.exists() and maybe_dir.is_file():
            print(f"[INFO] val 解析为txt：{maybe_dir}")
            stem2path = build_stem2path_from_txt(maybe_dir, root=root)
        else:
            raise FileNotFoundError(f"无法解析 val 路径：{val_entry}（解析后：{val_path}）")

    # 根据 ID 匹配路径
    hit, miss = 0, 0
    lines = []
    for iid in ids:
        p = stem2path.get(iid, None)
        if p and os.path.exists(p):
            lines.append(p)
            hit += 1
        else:
            # 允许 jpg/png 试探（某些情况下 stem 映射缺失）
            found = False
            for ext in IMG_EXTS:
                # 如果 val 是目录，可以直接拼
                if val_path.is_dir():
                    cand = val_path / f"{iid}{ext}"
                    if cand.exists():
                        lines.append(str(cand.resolve()))
                        hit += 1
                        found = True
                        break
                # 若 val 是txt，里头已经是绝对路径；再全局搜也行（开销大，不建议）
            if not found:
                miss += 1

    os.makedirs(Path(OUT_TXT).parent, exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

    print(f"[INFO] 写入 {OUT_TXT}")
    print(f"[INFO] 匹配成功 {hit} / 未匹配 {miss}")
    if miss > 0:
        print("[WARN] 有未匹配到的ID，请确认这些ID是否确实属于 val 集，或检查图像后缀/放置目录。")

if __name__ == "__main__":
    main()

# yolo_val_low_quality.py
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from pathlib import Path
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

def get_weight_size(path: str) -> str:
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    # ===================== 必改区（按需修改） =====================
    # 1) 模型权重（你的CRL-YOLO11）
    model_path = '/xx/xx/xx/xxr/xx/runs/train/xx/weights/best.pt'
    # 2) 数据根目录（来自 Fish.yaml 的 path）
    dataset_root = '/T2013054/hbq/Datasets/Fish'
    # 3) 低质量子集的图片列表（我们由 CSV 生成的 val_low_quality.txt）
    val_lowq_txt = '/xx/xx/xx/xxr/xx/val_low_quality.txt'
    # 4) 结果保存路径与名称
    project_dir = '/xx/xx/xx/xxr/xx/runs/val'
    run_name = 'CRL-YOLO11-lowq'  # 跑baseline时可改成 'yolov11n-lowq'
    # 5) 验证设置
    imgsz = 640
    batch = 16
    # ===========================================================

    # 基本检查
    assert os.path.isfile(model_path), f"权重不存在: {model_path}"
    assert os.path.isdir(dataset_root), f"数据根目录不存在: {dataset_root}"
    assert os.path.isfile(val_lowq_txt), f"低质量子集txt不存在: {val_lowq_txt}"

    # 加载模型
    model = YOLO(model_path)

    # 用 data 覆盖 val（指向低质量子集），无需改 Fish.yaml
    data_cfg = {
        'path': dataset_root,          # 与 Fish.yaml 的 path 对齐
        'val':  val_lowq_txt           # 低质量子集图片绝对路径列表
        # 其余如 'names'、'nc' 会从模型/训练权重中解析，不必写死
    }

    # 运行验证
    result = model.val(
        data=data_cfg,
        imgsz=imgsz,
        batch=batch,
        project=project_dir,
        name=run_name,
        verbose=True
    )

    # 仅目标检测任务下生成表格
    if model.task == 'detect':
        # 速度统计（毫秒/图）
        preprocess_time_per_image = result.speed['preprocess']  # ms
        inference_time_per_image  = result.speed['inference']   # ms
        postprocess_time_per_image= result.speed['postprocess'] # ms
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image

        # 模型信息
        n_layers, n_params, n_grad, flops = model_info(model.model)

        # 类别与逐类指标
        model_names = list(result.names.values())
        length = len(model_names)

        # ========== 打印模型信息表 ==========
        print('\n' + '='*25 + ' 模型信息 ' + '='*25)
        t1 = PrettyTable()
        t1.field_names = ["GFLOPs", "Parameters", "前处理(s)", "推理(s)", "后处理(s)",
                          "FPS(总)", "FPS(推理)", "权重大小"]
        t1.add_row([
            f'{flops:.1f}',
            f'{n_params:,}',
            f'{preprocess_time_per_image / 1000:.6f}',
            f'{inference_time_per_image  / 1000:.6f}',
            f'{postprocess_time_per_image/ 1000:.6f}',
            f'{1000 / all_time_per_image:.2f}' if all_time_per_image > 0 else 'N/A',
            f'{1000 / inference_time_per_image:.2f}' if inference_time_per_image > 0 else 'N/A',
            f'{get_weight_size(model_path)}MB'
        ])
        print(t1)

        # ========== 打印指标结果表（低质量子集） ==========
        print('\n' + '='*25 + ' 指标结果 (低质量子集) ' + '='*25)
        t2 = PrettyTable()
        t2.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]

        # 逐类
        for idx in range(length):
            t2.add_row([
                model_names[idx],
                f"{result.box.p[idx]:.4f}",
                f"{result.box.r[idx]:.4f}",
                f"{result.box.f1[idx]:.4f}",
                f"{result.box.ap50[idx]:.4f}",
                f"{result.box.all_ap[idx, 5]:.4f}",  # 0:50 1:55 ... 5:75 ... 9:95
                f"{result.box.ap[idx]:.4f}"
            ])

        # 平均
        t2.add_row([
            "all(平均)",
            f"{result.results_dict.get('metrics/precision(B)', np.nan):.4f}",
            f"{result.results_dict.get('metrics/recall(B)',    np.nan):.4f}",
            f"{np.nanmean(result.box.f1[:length]):.4f}",
            f"{result.results_dict.get('metrics/mAP50(B)',     np.nan):.4f}",
            f"{np.nanmean(result.box.all_ap[:length, 5]):.4f}",
            f"{result.results_dict.get('metrics/mAP50-95(B)',  np.nan):.4f}"
        ])
        print(t2)

        # 保存表格到文件
        save_dir = Path(result.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / 'paper_data.txt'
        with open(save_file, 'w+', encoding='utf-8') as f:
            f.write(str(t1))
            f.write('\n\n')
            f.write(str(t2))
        print(f"\n[INFO] 结果已保存到 {save_file}")

    else:
        print("[WARN] 当前任务非检测任务，未生成检测类指标表。")


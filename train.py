import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 训练参数官方详解链接：https://docs.ultralytics.com/zh/modes/train/#how-do-i-train-an-object-detection-model-using-ultralytics-yolo11

if __name__ == '__main__':
    model = YOLO('xxx/xxx/ultralytics/ultralytics/cfg/models/11/yolo11n.yaml')
    model.train(data='/xxx/xxx/ultralytics/ultralytics/cfg/datasets/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=8,
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                # fraction=0.2,
                project='/xxx/xxx/ultralytics/runs/train',
                name='xxx',
                )
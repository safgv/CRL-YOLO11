import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 验证参数官方详解链接：https://docs.ultralytics.com/zh/modes/val/

if __name__ == '__main__':
    model = YOLO('/xxx/xxx/ultralytics/runs/train/xxx/weights/best.pt')
    model.val(data='/xxx/xxx/ultralytics/ultralytics/cfg/datasets/URPC2019.yaml',
              split='val',
              imgsz=640,
              batch=1,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='/xxx/xxx/ultralytics/runs/val',
              name='xxx',
              )
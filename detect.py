import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 推理参数官方详解链接：https://docs.ultralytics.com/zh/modes/predict/

if __name__ == '__main__':
    model = YOLO('/xxx/xxx/ultralytics/runs/train/xxx/weights/best.pt') # select your model.pt path
    model.predict(source='/xxx/xxx/ultralytics/ultralytics/assets/bus.jpg',
                  imgsz=640,
                  project='/xxx/xxx/ultralytics/detect',
                  name='xxx',
                  save=True,
                  line_width=2,
                  conf=0.2,
                  iou=0.6,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )
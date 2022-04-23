# YOLOv5 ğŸš€ by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import sys
from pathlib import Path
import numpy as np
import torch
import cv2
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.general import check_img_size,   non_max_suppression, \
     scale_coords, set_logging,  save_one_box
from utils.torch_utils import select_device
from utils.augmentations import letterbox

class TDDetector:
    def __init__(self):
        ###Init Parameters
    
        self.imgsz=640  # inference size (pixels)
        self.conf_thres=0.6  # confidence threshold
        self.iou_thres=0.2  # NMS IOU threshold
        self.max_det=2  # maximum detections per image
        self.device='' # cuda device, i.e. 0 or 0,1,2,3 or cpu
        # Initialize
        set_logging()
        self.device = select_device(self.device)
        self.device.type != 'cpu'  # half precision only supported on CUDA
    
        print(Path(__file__).parents[0])
        self.model = attempt_load(weights=Path(__file__).parents[0]/'weights/TD.pt',  map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
    
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(16, 16))
        
    
    
        # self.model(torch.zeros(1, 3, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

    @torch.no_grad()

    def detect(self,source,night=False  # file/dir/
            ):

        ##Load Data
        
        im0 = source  # BGR
        if night:
            gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
            pre=im0.copy()
            eq = self.clahe.apply(gray)
            im0 = np.stack((eq,)*3, axis=-1)
        
        else:
            pre=im0
        
        # Padded resize
        img = letterbox(im0, self.imgsz, self.stride, auto=True)[0]
    
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        
    
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to FP32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
    
        # Inference
        pred = self.model(img, augment=False, visualize=False)[0]
    
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=self.max_det)
        # Process predictions
        up=[]
        down=[]
        for i, det in enumerate(pred):  # detections per image
    
            im0_copy =  im0.copy()
    
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0_copy.shape).round()
    
                for *xyxy, conf, cls in reversed(det):
                    print(conf)
                    c = int(cls)  # integer class
                    if self.names[c] == 'h':
                        up=save_one_box(xyxy, im0_copy, file='', BGR=True,save=False)
                    elif self.names[c] == 'd':
                        down=save_one_box(xyxy, im0_copy, file='', BGR=True,save=False)
                    else:
                        continue
                    
                    
                            
        return up, down
    

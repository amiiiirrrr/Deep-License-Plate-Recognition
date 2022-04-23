# YOLOv5 ğŸš€ by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import sys
from pathlib import Path
import os
import numpy as np
import torch
import cv2
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.general import check_img_size,   non_max_suppression, \
     scale_coords, set_logging,  save_one_box, xyxy2xywh
from utils.torch_utils import select_device
from utils.augmentations import letterbox

class PLDetector:
    def __init__(self,point1,point2,point3,point4,point5,point6):
        ###Init Parameters
    
        self.imgsz=[640,640]  # inference size (pixels)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.device='' # cuda device, i.e. 0 or 0,1,2,3 or cpu
        # Initialize
        set_logging()
        self.device = select_device(self.device)
        self.device.type != 'cpu'  # half precision only supported on CUDA
    
        print(Path(__file__).parents[0])
        self.model = attempt_load(weights=Path(__file__).parents[0]/'weights/yolov5l.pt',  map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
    
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(16, 16))
        
    
    
        self.model(torch.zeros(1, 3, *self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.LPmodel = attempt_load(Path(__file__).parents[0]/'weights/LP.pt', map_location=self.device)  # load FP32 model
        
        self.LPnames = self.LPmodel.module.names if hasattr(self.LPmodel, 'module') else self.LPmodel.names  # get class names
        self.point1=point1
        self.point2=point2
        self.point3=point3
        self.point4=point4
        self.point5=point5
        self.point6=point6
    @torch.no_grad()
    
    def geteq(self,point1,point2):
        points = [point1,point2]
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords,np.ones(len(x_coords))]).T
        m, c = np.linalg.lstsq(A, y_coords,rcond=None)[0]
        return m,c


    def detect(self,source='img_12mp',night=False  # file/dir/
            ):

        ##Load Data
        im0 = cv2.imread(source)  # BGR
        
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
        
        
        
        count_found=0
        count_notfound=0
        #for saving LPs
        LP = []
        lp_line=[]
        tlbr_lp = []
    
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to FP32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
    
        # Inference for Car
        pred = self.model(img, augment=False, visualize=False)[0]
    
        # NMS for Car
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=self.max_det)
    
        # Process predictions
        for i, det in enumerate(pred):  # detections per image
    
            im0_copy =  im0.copy()
    
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0_copy.shape).round()
    
                for *xyxy, conf, cls in reversed(det):
    
                    c = int(cls)  # integer class
                    if self.names[c] not in ['car', 'bus', 'truck','train']:
                        continue
                    
                    crop=save_one_box(xyxy, im0_copy, file='', BGR=True,save=False)
                    cropc = crop.copy()  # for save_crop
                    crop = letterbox(crop, 640, stride=32, auto=True)[0]
                    
                    crop = crop.transpose((2, 0, 1))  # HWC to CHW, BGR to RGB[::-1]
                    crop = np.ascontiguousarray(crop)
                    
                    crop = torch.from_numpy(crop).to(self.device)
                    crop = crop.float()  # uint8 to 32
                    crop = crop / 255.0  # 0 - 255 to 0.0 - 1.0
                    if len(crop.shape) == 3:
                        crop = crop[None]  # expand for batch dim
                    
                    # Inference for LP
                    pred2 = self.LPmodel(crop, augment=False, visualize=False)[0]
                    
                    # NMS for LP
                    pred2 = non_max_suppression(pred2, self.conf_thres, self.iou_thres, None, False, max_det=self.max_det)
                    
                    for j, LPdet in enumerate(pred2):  # detections per image
                        if not len(LPdet):
                            count_notfound+=1
                            save_one_box(xyxy, im0_copy, file='Saved_data/Not_Founded/'  +str(count_notfound)+'.jpg', BGR=True)
    
                        if len(LPdet):
                            
                            # Rescale boxes from img_size to im0 size
                            LPdet[:, :4] = scale_coords(crop.shape[2:], LPdet[:, :4], cropc.shape).round()

                            # Write results
                            for *xyxy_LP, conf, cls in reversed(LPdet):
                                c = int(cls)  # integer class
                                count_found+=1
                                
                                extracted=save_one_box(xyxy_LP, cropc, file='Saved_data/' + self.LPnames[c] +"/"+ str(count_found)+'.jpg', BGR=True)
                                if night:
                                    denoise=cv2.medianBlur(extracted,3)
                                    gaussian_3 = cv2.GaussianBlur(denoise, (0, 0), 3.0)
                                    unsharp_image = cv2.addWeighted(denoise, 9.0, gaussian_3, -8.0, 0, denoise)
                                    extracted=cv2.medianBlur(unsharp_image,3)
                                LP.append(extracted)
                                #  bounding box coordinate of LP
                                cx,cy,w,h = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) ).view(-1).tolist()
                                tl = np.array([cx - w/2., cy - h/2.]) # top left coordinate of vehicle
                                br = np.array([cx + w/2., cy + h/2.]) # buttom right coordinate of vehicle
                                cx,cy,w,h = (xyxy2xywh(torch.tensor(xyxy_LP).view(1, 4)) ).view(-1).tolist()
                                tl_lp = np.array([cx - w/2., cy - h/2.]) # top left coordinate of LP
                                br_lp = np.array([cx + w/2., cy + h/2.]) # buttom right coordinate of LP
                                extracted_shape = np.array(extracted.shape[:2][::-1])
                                tl_lp = tl + tl_lp
                                br_lp = br + br_lp - extracted_shape
                                tlbr_np = np.append(tl_lp,br_lp)
                                tlbr_lp.append(torch.Tensor(tlbr_np.tolist()))
                                cx,cy,w,h = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) ).view(-1).tolist()
                                x=cx+w//2
                                y=cy+h//2
                                m,c=self.geteq(self.point1,self.point2)
                                tempy=m*x+c
                                if tempy<y:
                                    line=1
                                else:
                                    m,c=self.geteq(self.point3,self.point4)
                                    tempy=m*x+c
                                    if tempy<y:
                                        line=2
                                    else:
                                        m,c=self.geteq(self.point5,self.point6)
                                        tempy=m*x+c
                                        if tempy<y:
                                            line=3
                                        else:
                                            line=4
                                lp_line.append(line)
 
        return LP, tlbr_lp ,lp_line
    

"""
run.py is written for test Plate Recognition model
"""
import tensorflow as tf
import cv2

from tools.draw_lp import draw_result
from ocrPlate.ocr_up.src_up.infer_app import inferOcr as inferOcr_up
from ocrPlate.ocr_down.src_down.infer_app import inferOcr as inferOcr_down
from ocrPlate.deep_ocr_up.infer_app import inferOcr as deep_inferOcr_up
from ocrPlate.deep_ocr_down.infer_app import inferOcr as deep_inferOcr_down
from yolov5_textDetection.text_detector import TDDetector
from plateDetection_yolov5.inference import PLDetector
import os
import glob
import time

__author__ = "Ateam"
__project__ = "Ateam, Palte Recognition"
__license__ = "Public Domain"
__version__ = "1.0.0"
__email__ = "azmusavi19@gmail.com"
__status__ = "Production"

class ModelRrunner:
    """
    A class to test the model
    """
    def __init__(self):
        """
        The initialize method 
        :param args: argparser
        :return: None
        """
        model_down = tf.Graph()
        with model_down.as_default():
            self.inferOcr_down_obj = deep_inferOcr_down()
        model_up = tf.Graph()
        with model_up.as_default():
            self.inferOcr_up_obj = deep_inferOcr_up()
        self.newTD=TDDetector()
        self.plateDet_obj = PLDetector()
        # self.textDet_obj = TextDetector()
        
    def run(self, img_path):
        """
        A method to run the training and testing the model
        :param img_path: str
        :return: None
        """
        
        detected_plates, tlbr_lp = self.plateDet_obj.detect(source=img_path, night=False)
        lp_text = []
        i = 0
        for plate in detected_plates:
            cv2.imwrite("plate" + str(i) + ".jpg", plate)
            recognized_up = ",,,,,"
            recognized_down = ",,,,,"
            # up_text, down_text = self.textDet_obj.detect_text(plate)
            up_text,down_text = self.newTD.detect(plate)
            # print(up_text)
            if len(up_text) > 0:
             recognized_up, conf_up = self.inferOcr_up_obj.infer_up(up_text)
             if conf_up < 0.085:
                    recognized_up = ",,,,,"
             print(f'{recognized_up} , {conf_up}')
             cv2.imwrite("plateup" + str(i) + ".jpg", up_text)
            if len(down_text) > 0:
                recognized_down, conf_down = self.inferOcr_down_obj.infer_down(down_text)
                print(f'{recognized_down} , {conf_down}')
                if conf_down < 0.085:
                    recognized_down = ",,,,,"
                cv2.imwrite("platedown" + str(i) + ".jpg", down_text)
            lp_text.append([recognized_up + ' ' + recognized_down])
            i += 1
            

        print(lp_text)
        draw_result(tlbr_lp, lp_text, img_path, line_thickness=3)

    def lp_extract(self, img_path, output_path):
        """
        A method to extract licence plate
        :param img_path: str
        :param output_path: str
        :return: None
        """
        current_path = os.getcwd()
        save_path = current_path + output_path
        images_path = glob.glob(img_path)
        for img in images_path:
            detected_plates, tlbr_lp = self.plateDet_obj.detect(source=img, night=False)
            lp = 1
            for plate in detected_plates:
                cv2.imwrite(save_path + 'lp/' + str(lp) + '_' + img.split('/')[-1], plate)
                # up_text, down_text = self.textDet_obj.detect_text(plate)
                up_text,down_text = self.newTD.detect(plate)
                if len(up_text) > 0:
                    cv2.imwrite(save_path + 'up/' + str(lp) + '_' + img.split('/')[-1], up_text)
                if len(down_text) > 0:
                    cv2.imwrite(save_path + 'down/' + str(lp) + '_' + img.split('/')[-1], down_text)
                lp += 1

    def run_directory(self, img_path, output_path):
        """
        A method to run in a directory
        :param img_path: str
        :param output_path: str
        :return: None
        """
        current_path = os.getcwd()
        save_path = current_path + output_path
        images_path = glob.glob(img_path)
        count = 0
        total_t = []
        for img in images_path:
            print(img)
            t0 = time.time()
            ave_t = 0
            detected_plates, tlbr_lp = self.plateDet_obj.detect(source=img, night=False)
            dt = time.time() - t0
            ave_t += dt 
            lp_text = []
            for plate in detected_plates:
                count += 1
                recognized_up = ",,,,,"
                recognized_down = ",,,,,"
                # up_text, down_text = self.textDet_obj.detect_text(plate)
                t0 = time.time()
                up_text,down_text = self.newTD.detect(plate)
                dt = time.time() - t0
                ave_t += dt
                # cv2.imwrite(f"Saved_data/Final_result/plate_{img.split('/')[-1].split('.')[0]}_" + str(count) + ".png", plate)
                if len(up_text) > 0:
                    cv2.imwrite("Saved_data/Final_result/up/plateup" + str(count) + ".png", up_text)
                    t0 = time.time()
                    recognized_up, conf_up = self.inferOcr_up_obj.infer_up(up_text)
                    dt = time.time() - t0
                    ave_t += dt
                    if conf_up < 0.05:
                        recognized_up = ",,,,,"
                if len(down_text) > 0:
                    cv2.imwrite("Saved_data/Final_result/down/platedown" + str(count) + ".png", down_text)
                    t0 = time.time()
                    recognized_down, conf_down = self.inferOcr_down_obj.infer_down(down_text)
                    dt = time.time() - t0
                    ave_t += dt
                    if conf_down < 0.05:
                        recognized_down = ",,,,,"
            
                lp_text.append([recognized_up + ' ' + recognized_down])
            print("Time per Image: ", ave_t)
            total_t.append(ave_t)
        print("average time", sum(total_t)/len(total_t))
            # draw_result(tlbr_lp, lp_text, img, line_thickness=3)

    def run_video(self, vid_path, output_path):
        """
        A method to run on a video
        :param vid_path: str
        :param output_path: str
        :return: None
        """
        file_name = vid_path.split('/')[-1].split('.')[0]
        print(file_name)
        os.makedirs(f'saved_video/{file_name}', exist_ok=True)
        vidcap = cv2.VideoCapture(vid_path)

        success,image = vidcap.read()
        if not success:
            print("capture failed!")
        count = 0
        while success:
            detected_plates, tlbr_lp = self.plateDet_obj.detect(source=image, night=False)
            lp_text = []
            i = 0
            for plate in detected_plates:
                cv2.imwrite(f'saved_video/{file_name}/{file_name}_{count}_' + str(i) + ".jpg", plate)
                recognized_up = ",,,,,"
                recognized_down = ",,,,,"
                up_text, down_text = self.textDet_obj.detect_text(plate)
                up_text,down_text = self.newTD.detect(plate)
                # print(up_text)
                if len(up_text) > 0:
                 recognized_up, conf_up = self.inferOcr_up_obj.infer_up(up_text)
                 if conf_up < 0.5:
                        recognized_up = ",,,,,"
                 print(f'{recognized_up} , {conf_up}')
                 cv2.imwrite("Saved_data/Final_result/up/plateup" + str(count) + ".jpg", up_text)
                if len(down_text) > 0:
                    recognized_down, conf_down = self.inferOcr_down_obj.infer_down(down_text)
                    print(f'{recognized_down} , {conf_down}')
                    if conf_down < 0.5:
                        recognized_down = ",,,,,"
                    cv2.imwrite("Saved_data/Final_result/down/platedown" + str(count) + ".jpg", down_text)
                lp_text.append([recognized_up + ' ' + recognized_down])
                print(count)
                i += 1
            # print(lp_text)
            if len(detected_plates) > 0:
                draw_result(tlbr_lp, lp_text, f'{file_name}_{count}.bmp', line_thickness=3, img_source=image)
            # cv2.imwrite(f"{file_name}/{file_name}_{count}.bmp", image)     # save frame as JPEG file      
            success,image = vidcap.read()
            # print('Read a new frame: ', success, count)
            count += 1

if __name__ == '__main__':
    image_path = 'test_images/30_near_300w_53.bmp'#daytime_test'
    images_path = '/home/fteam/Desktop/amir/plateRecognition/test_images/daytime_test/*.bmp'
    output_path = '/Saved_data/Final_result/'
    vid_path = 'test_video/30_near_off.mp4'

    run_model_obj = ModelRrunner()
    # run_model_obj.run(image_path)
    # run_model_obj.lp_extract(images_path, output_path)
    run_model_obj.run_directory(images_path, output_path)
    # run_model_obj.run_video(vid_path, output_path)


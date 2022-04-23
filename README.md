# Deep-License-Plate-Recognition
License Plate Recognition Using Yolov5 for plate detection, ResNet-BiLSTM-Attention or simpleHTR models for OCR

# Structure

## Plate detection using Yolov5

In this section, Plates will be detected by fine-tune Yolov5 object detection model. In order to train, provide your own license plate dataset and re-train yolov5 model by following below steps.

![alt text](https://github.com/amiiiirrrr/Deep-License-Plate-Recognition/blob/main/plateDetection_yolov5/dataset/images/test/006_1.jpg)

1. cd to plateDetection_yolov5
2. $ pip install -r requirements.txt
3. Install pytorch (If you want to use the NVIDIA GeForce RTX GPU with PyTorch, please check
the instructions at [pytorch installing guide.](https://pytorch.org/get-started/locally/))
4. For training, create a folder named “LP”. LP folder should have this format:

  |--LP

      |--images
      
            |--train
            
            |--test
            
      |--labels
      
            |--train
            
            |--test
            
            
5. python train.py
6. Your can see your training result under “runs/train” folder
7. Replace your “best.pt” weight with “LP.pt” under wights folder.

• models: the necessary files for YOLOv5 model for license plate detection and training. E.g., conv layers.

• utils: functions for summarize inference and train.

• weights: trained weight for car and plates.

• inference.py: contains an “detect” function for inference. Returns a list of detected license plate

• requairments.txt: requirements package for running

• train.py: file for training and configs

After finishing training you will got a fine-tuned model that will able to find pltaes in your dataset distribution. Use this model in pipeline.

## Detect Up and Down of detected Plate

After Detection of License plate we should seprate up text and down text of each plate so that having correct OCR. Because each license has two line of text. This sepration will help us having great output.

![1_daytime_0913_0102](https://user-images.githubusercontent.com/28767607/164892983-ab1aa83f-29d0-4e88-b259-fd5471b9c764.jpg)


![Capture](https://user-images.githubusercontent.com/28767607/164892988-1061dd8f-2e52-4235-8517-b1fe73c91832.PNG)


![Capture2](https://user-images.githubusercontent.com/28767607/164892993-07c2976c-a8dc-4a3a-a952-44d375e9d8d5.PNG)

Using yolov5_textDetection directory you can train the model on your own dataset to detect each line of text.

## OCR Plates 

After detection of each line of text, it's OCR's turn to recognize what has written in the picture. Two types of model are used in this section. ResNet-BiLSTM-Attention model and simpleHTR model (cnn + lstm + ctc layers). We get better output by the first model.

### OCR using ResNet-BiLSTM-Attention model

deep_ocr_down and deep_ocr_up directories are related to ResNet-BiLSTM-Attention model. For each line of text we train a model.

### OCR using simpleHTR model

ocr_down and ocr_up directories are related to simpleHTR model. For each line of text we train a model.

# References

https://github.com/ultralytics/yolov5

https://github.com/githubharald/SimpleHTR

https://github.com/clovaai/deep-text-recognition-benchmark



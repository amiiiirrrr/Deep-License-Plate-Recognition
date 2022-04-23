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

![alt text](C:\Users\amir\Desktop\1_d![1_daytime_0913_0102](https://user-images.githubusercontent.com/28767607/164891258-0cfadfd0-9dc1-4ebb-b1b4-316787b0c3b6.jpg)
aytime_0913_0102.jpg) 
---------------->>>>    
![alt text](C:\Users\amir\Desk![Capture](https://user-images.githubusercontent.com/28767607/164891272-1ec0cb5a-8df0-4731-bccc-023dc724abdf.PNG)

top\Capture.PNG) ![alt text](![Capture2](https://user-images.githubusercontent.com/28767607/164891293-aac62704-43d9-41c7-8c51-08d085d82e21.PNG)
)

## OCR Plates 

### OCR using ResNet-BiLSTM-Attention model

### OCR using simpleHTR model

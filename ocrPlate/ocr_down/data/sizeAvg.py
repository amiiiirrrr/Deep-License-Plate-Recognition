import glob
import cv2

# up_path = glob.glob('up/*.png')
down_path = glob.glob('data/*.png')

up_h = []
down_h = []
up_w =[]
down_w = []

# for path in up_path:
#     image = cv2.imread(path)
#     height, width, channels = image.shape
#     up_h.append(height)
#     up_w.append(width)
# print('####################')
# print(f'Height average of up is: {sum(up_h) / len(up_h)}')
# print(f'Width average of up is: {sum(up_w) / len(up_w)}')

for path in down_path:
    image = cv2.imread(path)
    height, width, channels = image.shape
    down_h.append(height)
    down_w.append(width)
print('####################')
print(f'Height average of down is: {sum(down_h) / len(down_h)}')
print(f'Width average of down is: {sum(down_w) / len(down_w)}')

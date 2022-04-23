"""
draw_lp.py is written for draw output
"""

import cv2
import os

__author__ = "Ateam"
__project__ = "Ateam, Palte Recognition"
__license__ = "Public Domain"
__version__ = "1.0.0"
__email__ = "azmusavi19@gmail.com"
__status__ = "Production"

def draw_result(tlbr_lp, lp_text, img_path='img', line_thickness=3, img_source=[]):
    current_path = os.getcwd()
    save_path = current_path + '/Saved_data/Final_result/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if len(img_source) > 0:
        img = img_source.copy()
    else: 
        img = cv2.imread(img_path)
    for ind, tlbr in enumerate(tlbr_lp):
        if not ",,,,," in lp_text[ind][0]:
            img = plot_one_box(tlbr, img, label=lp_text[ind][0], line_thickness=line_thickness)
    cv2.imwrite(save_path + img_path.split('/')[-1], img)
    print(f'Saved result path: {save_path}')
	
def plot_one_box(x, im, color=(200, 0, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle(im, c1, c2, color, thickness=3)#, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 2, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 2, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return im
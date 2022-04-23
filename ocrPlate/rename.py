import glob
import os
import cv2
import numpy as np

lp_kanji_dict = {"a": '滋', "b": '賀', "c": 'な', "d": 'に', "e": 'わ', "f": '堺', "g": '大', "h": '阪', "i": '和', "j": '泉', "k": '神',
                 "l": '戸', "m": '姫', "n": '路', "o": '奈', "p": '良', "q": '三', "r": '重', "s": '歌', "t": '山', "u": '富', "v": '士', 
                 "w": '浜', "x": '松', "y": '岡', "z": '崎', "A": '名', "B": '古', "C": '屋', "D": '河', "E": '尾', "F": '張', "G": '小', 
                 "H": '牧', "I": '豊', "J": '田', "K": '春', "L": '日', "M": '井', "N": '梨', "O": '諏', "P": '訪', "Q": '岐', "R": '阜',
                 "S": '京', "T": '都', "U": '市', "V": 'つ', "W": 'く', "X": 'ば', "Y": '横', "Z": '熊', "!": '谷', "@": '宮', "#": '川', 
                 "%": '越', "^": '宇', "&": '足', "*": '立'," _": '多', "+": '摩', "=": '品', "-": '練', "<": '馬', ">": '福',"?": '石', 
                 "|": '金', "[": '沢', "]": '高', "}": '知', "{": '徳', ":": '島', "`": '北', "~": '九', ",": '州', 
                 ";": '佐', ".": '世', "/": '保', "\\": '本'}#,"(":"読", ")":"み"}
kanji_list = []
def replace_kanji(name):
	s1 = name[0]
	s2 = name[1]
	s3 = name[2]
	numbers = "0123456789"
	numbers = list(numbers)
	print(s1,"---",s2,"---",s3)
	for key, val in lp_kanji_dict.items():
		if s1 == val:
			k1 = key
			kanji_list.append(k1)
		if s2 == val:
			k2 = key
			kanji_list.append(k2)
		if s3 == val:
			k3 = key
			kanji_list.append(k3)
		if s1 in numbers:
			k1 = s1
		if s2 in numbers:
			k2 = s2
		if s3 in numbers:
			k3 = s3
	# print(s1,"-->",k1)
	# print(s2,"-->",k2)
	name_list = list(name)
	name_list[0] = k1
	name_list[1] = k2
	name_list[2] = k3
	name = ''.join(name_list)
	return name

images = glob.glob('up/*.png')
images = os.listdir('up/')

img = images[0]

for img in images:
	print(img)
	im = cv2.imread('up/' + img)
	im_name = img.split('.')[0]
	im_name = replace_kanji(im_name)
	
	cv2.imwrite('upk/'+im_name+'.png',im)
print("-"*30)
final_new_menu = list(dict.fromkeys(kanji_list))
print(''.join(final_new_menu))

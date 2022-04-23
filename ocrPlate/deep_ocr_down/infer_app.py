import argparse
import json
from typing import Tuple, List
import cv2
import sys
import os
# appending current directory
sys.path.append('ocrPlate/deep_ocr_down/')
import string
import argparse
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from modules.utils import CTCLabelConverter, AttnLabelConverter
from modules.dataset import RawImg, AlignCollate
from modules.model import Model
from modules.config_down import BaseConfig
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class inferOcr:
    def __init__(self):

        self.dict = {'a': 'あ', 'b': 'い', 'c': 'う', '-': '-', 'e': 'え', 'f': 'お', 'g': 'か', 'h': 'き', 'i': 'く', 'j': 'け', 'k': 'こ', 'l': 'さ', 'm': 'し', 'n': 'す', 'o': 'せ', 'p': 'そ', 'q': 'た', 'r': 'ち', '.':'.', 't': 'つ', 'u': 'て', 'v': 'と', 'w': 'な', 'x': 'に', 'y': 'ぬ', 'z': 'ね', 'A': 'の', 'B': 'は', 'C': 'ひ', 'D': 'ふ', 'E': 'へ', 'F': 'ほ', 'G': 'ま', 'H': 'み', 'I': 'む', 'J': 'め', 'K': 'も', 'L': 'や', 'M': 'ゆ', 'N': 'よ', 'O': 'ら', 'P': 'り', 'Q': 'る', 'R': 'れ', 'S': 'ろ', 'T': 'わ', 'U': 'を', '0':'0', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9'}
        self.opt = BaseConfig().get_args()
        """ vocab / character number configuration """
        if self.opt.sensitive:
            self.opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
        cudnn.benchmark = True
        cudnn.deterministic = True
        self.opt.num_gpu = torch.cuda.device_count()

        """ model configuration """
        if 'CTC' in self.opt.Prediction:
            self.converter = CTCLabelConverter(self.opt.character)
        else:
            self.converter = AttnLabelConverter(self.opt.character)
        self.opt.num_class = len(self.converter.character)

        if self.opt.rgb:
            self.opt.input_channel = 3
        self.model = Model(self.opt)
        self.model = torch.nn.DataParallel(self.model).to(device)

        # load model
        print('loading OCR_Down pretrained model from %s' % self.opt.saved_model)
        self.model.load_state_dict(torch.load(os.getcwd() + self.opt.saved_model, map_location=device))
        self.model.eval()

    def infer_down(self, img) -> None:
        """
        Main function.
        """
        assert img is not None
        
        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        AlignCollate_demo = AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD)
        demo_data = RawImg(img, self.opt)
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=AlignCollate_demo, pin_memory=True)
        # infer text on test image
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in self.opt.Prediction:
                preds = self.model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = self.converter.decode(preds_index, preds_size)

            else:
                preds = self.model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            preds_str = preds_str[0]
            if 'Attn' in self.opt.Prediction:
                pred_EOS = preds_str.find('[s]')
                preds_str = preds_str[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = preds_max_prob[0][:pred_EOS]
            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            preds_str = preds_str.replace('d','-')
            preds_str = preds_str.replace('s','.')
            preds_str = self.dict[preds_str[0]] + preds_str[1:]
        return preds_str, confidence_score.item()

# if __name__ == '__main__':
#     ocr = inferOcr()
#     img = cv2.imread('test.png')
#     print(ocr.infer_down(img))
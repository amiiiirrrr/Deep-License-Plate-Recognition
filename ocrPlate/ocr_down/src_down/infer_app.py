import argparse
import json
from typing import Tuple, List
import cv2
import sys
# appending current directory
sys.path.append('ocrPlate/ocr_down/src_down/')
from dataloader_down import DataLoaderIAM, Batch
from methods_down import Model, DecoderType
from preprocess_down import Preprocessor
from configuration_down import BaseConfig


class inferOcr:
    def __init__(self):
        self.args = BaseConfig().get_args()
        self.preprocessor = Preprocessor(self.get_img_size(), dynamic_width=False, padding=0)
        # set chosen CTC decoder
        decoder_mapping = {'bestpath': DecoderType.BestPath,
                           'beamsearch': DecoderType.BeamSearch,
                           'wordbeamsearch': DecoderType.WordBeamSearch}
        decoder_type = decoder_mapping[self.args.decoder]
        self.model_down = Model(self.args, list(open(
            self.args.fn_char_list).read()), 
            decoder_type, must_restore=True, dump=self.args.dump)

    def get_img_height(self) -> int:
        """Fixed height for NN."""
        return self.args.img_height


    def get_img_size(self, line_mode: bool = False) -> Tuple[int, int]:
        """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
        if line_mode:
            return self.args.img_size, self.get_img_height()
        return self.args.img_size, self.get_img_height()


    def infer_down(self, img) -> None:
        """
        Main function.
        """
        # infer text on test image

        # img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        # im = cv2.imread(img_pth)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        assert img is not None
        img = self.preprocessor.process_img(img)

        batch = Batch([img], None, 1)
        recognized, probability = self.model_down.infer_batch(batch, True)
        # im = cv2.putText(im, recognized[0], (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
        #                0.5, (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.imwrite(save_path+recognized[0] + '.png',im)
        # print(f'pp: "{recognized[0]}"')
        # print(f'Probability: {probability[0]}')
        return recognized[0], probability[0]

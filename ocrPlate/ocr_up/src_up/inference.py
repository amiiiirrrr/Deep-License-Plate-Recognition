
from typing import Tuple
import cv2
from dataloader import Batch
from methods import Model, DecoderType
from preprocess import Preprocessor
from configuration import BaseConfig
import os

class inference:

    def __init__(self, args):

        self.args = args
        self.preprocessor = Preprocessor(self.get_img_size(), dynamic_width=False, padding=0)
        # set chosen CTC decoder
        decoder_mapping = {'bestpath': DecoderType.BestPath,
                           'beamsearch': DecoderType.BeamSearch,
                           'wordbeamsearch': DecoderType.WordBeamSearch}
        decoder_type = decoder_mapping[args.decoder]
        self.args.batch_size = 1
        self.model = Model(self.args, list(open(args.fn_char_list).read()), decoder_type, must_restore=True, dump=args.dump)

    def get_img_height(self) -> int:
        """Fixed height for NN."""
        return self.args.img_height

    def get_img_size(self, line_mode: bool = False) -> Tuple[int, int]:
        """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
        if line_mode:
            return self.args.img_size, self.get_img_height()
        return self.args.img_size, self.get_img_height()

    def infer(self) -> None:
        """Recognizes text in image provided by file path."""
        path = '/home/fteam/Desktop/amir/OCR_Japen/SimpleHTR_full/data/test/'
        save_path = '/home/fteam/Desktop/amir/OCR_Japen/SimpleHTR_full/data/test2/result/'
        images = os.listdir(path)
        f = lambda x: os.path.join(path, x)
        imagepaths = list(map(f, images))
        for img_pth in imagepaths:
            img = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)
            im = cv2.imread(img_pth)
            assert img is not None

            # preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
            preprocessor = Preprocessor(self.get_img_size(), dynamic_width=False, padding=0)
            img = preprocessor.process_img(img)

            batch = Batch([img], None, 1)
            recognized, probability = self.model.infer_batch(batch, True)
            im = cv2.putText(im, recognized[0], (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imwrite(save_path+recognized[0] + '.png',im)
            print(f'pp: "{recognized[0]}"')
            print(f'Probability: {probability[0]}')


def main():
    """Main function."""
    args = BaseConfig().get_args()
    inf_obj = inference(args)
    # infer text on test image
    inf_obj.infer()


if __name__ == '__main__':
    main()

"""
config.py is written for define config parameters
"""
import argparse
from path import Path

__author__ = "Amir Mousavi"
__project__ = "Part company, Repetitive News"
__credits__ = ["Amir Mousavi"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Amir Mousavi"
__email__ = "azmusavi19@gmail.com"
__status__ = "Production"
print('######################DOWN')
class BaseConfig:
    """
    This class set static paths and other configs.
    Args:
    argparse :
    The keys that users assign such as sentence, tagging_model and other statictics paths.
    Returns:
    The configuration dict specify text, statics paths and controller flags.
    """
    def __init__(self):

        self.parser = argparse.ArgumentParser()
        self.run()

    def run(self):
        """
        The run method is written to define config arguments
        :return: None
        """
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
        self.parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
        self.parser.add_argument('--saved_model', default='/ocrPlate/deep_ocr_down/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth', help="path to saved_model to evaluation")
        """ Data processing """
        self.parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
        self.parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
        self.parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
        self.parser.add_argument('--rgb', action='store_true', help='use rgb input')
        self.parser.add_argument('--character', type=str, default='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQSRTU0123456789', help='character label')
        self.parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
        self.parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
        """ Model Architecture """
        self.parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
        self.parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
        self.parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
        self.parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
        self.parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
        self.parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
        self.parser.add_argument('--output_channel', type=int, default=512,
                            help='the number of output channel of Feature extractor')
        self.parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')


    def get_args(self):
        """
        The get_args method is written to return config arguments
        :return: argparse
        """
        return self.parser.parse_args()

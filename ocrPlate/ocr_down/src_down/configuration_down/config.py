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
        self.parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='train')
        self.parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='beamsearch')
        self.parser.add_argument('--batch_size', help='Batch size.', 
                                    type=int, 
                                    default=2,
                                    # default=1
                                    )
        self.parser.add_argument('--data_dir', 
                                    help='Directory containing IAM dataset.', 
                                    type=Path, 
                                    default='../data/',
                                    # default='ocrPlate/ocr_down/data/'
                                    )
        self.parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
        self.parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', default=False)
        self.parser.add_argument('--img_file', 
                                    help='Image used for inference.', 
                                    type=Path, 
                                    default='../data/word.jpg',
                                    # default='ocrPlate/ocr_down/data/test/word.png'
                                    )
        self.parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
        self.parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')
        self.parser.add_argument('--fn_char_list', 
                                    default='../model_down/charList.txt',
                                    # default='ocrPlate/ocr_down/model_down/charList.txt'
                                    )
        self.parser.add_argument('--fn_summary', 
                                    default='../model_down/summary.json',
                                    # default='ocrPlate/ocr_down/model_down/summary.json'
                                    )
        self.parser.add_argument('--fn_corpus', 
                                    default='../data/corpus.txt',
                                    # default='ocrPlate/ocr_down/data/corpus.txt'
                                    )
        self.parser.add_argument('--img_height', default=100)
        self.parser.add_argument('--img_size', default=256)
        self.parser.add_argument('--num_hidden', default=256)
        self.parser.add_argument('--model_dir', 
                                    default='../model_down/',
                                    # default='ocrPlate/ocr_down/model_down/'
                                    )

    def get_args(self):
        """
        The get_args method is written to return config arguments
        :return: argparse
        """
        return self.parser.parse_args()

import argparse
import json
from typing import Tuple, List

import cv2
import editdistance
from path import Path

from dataloader_down import DataLoaderIAM, Batch
from methods_down import Model, DecoderType
from preprocess_down import Preprocessor
from configuration_down import BaseConfig

class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model_down/charList.txt'
    fn_summary = '../model_down/summary.json'
    fn_corpus = '../data/corpus.txt'

class trainingEval:
    def __init__(self, args):
        self.args = args

    def get_img_height(self) -> int:
        """Fixed height for NN."""
        return self.args.img_height


    def get_img_size(self, line_mode: bool = False) -> Tuple[int, int]:
        """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
        if line_mode:
            return self.args.img_size, self.get_img_height()
        return self.args.img_size, self.get_img_height()


    def write_summary(self, char_error_rates: List[float], word_accuracies: List[float]) -> None:
        """Writes training summary file for NN."""
        with open(FilePaths.fn_summary, 'w') as f:
            json.dump({'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)


    def train(self, model: Model,
              loader: DataLoaderIAM,
              line_mode: bool,
              early_stopping: int = 25) -> None:
        """Trains NN."""
        epoch = 0  # number of training epochs since start
        summary_char_error_rates = []
        summary_word_accuracies = []
        preprocessor = Preprocessor(self.get_img_size(line_mode), data_augmentation=False, line_mode=line_mode)
        best_char_error_rate = float('inf')  # best valdiation character error rate
        no_improvement_since = 0  # number of epochs no improvement of character error rate occurred
        # stop training after this number of epochs without improvement
        while True:
            epoch += 1
            print('Epoch:', epoch)

            # train
            print('Train NN')
            loader.train_set()
            while loader.has_next():
                iter_info = loader.get_iterator_info()
                batch = loader.get_next()
                batch = preprocessor.process_batch(batch)
                loss = model.train_batch(batch)
                print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')

            # validate
            char_error_rate, word_accuracy = self.validate(model, loader, line_mode)

            # write summary
            summary_char_error_rates.append(char_error_rate)
            summary_word_accuracies.append(word_accuracy)
            self.write_summary(summary_char_error_rates, summary_word_accuracies)

            # if best validation accuracy so far, save model parameters
            if char_error_rate < best_char_error_rate:
                print('Character error rate improved, save model')
                best_char_error_rate = char_error_rate
                no_improvement_since = 0
                model.save()
            else:
                print(f'Character error rate not improved, best so far: {char_error_rate * 100.0}%')
                no_improvement_since += 1

            # stop training if no more improvement in the last x epochs
            if no_improvement_since >= early_stopping:
                print(f'No more improvement since {early_stopping} epochs. Training stopped.')
                break


    def validate(self, model: Model, loader: DataLoaderIAM, line_mode: bool) -> Tuple[float, float]:
        """Validates NN."""
        print('Validate NN')
        loader.validation_set()
        preprocessor = Preprocessor(self.get_img_size(line_mode), line_mode=line_mode)
        num_char_err = 0
        num_char_total = 0
        num_word_ok = 0
        num_word_total = 0
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            print(f'Batch: {iter_info[0]} / {iter_info[1]}')
            batch = loader.get_next()
            batch = preprocessor.process_batch(batch)
            recognized, _ = model.infer_batch(batch)

            print('Ground truth -> Recognized')
            for i in range(len(recognized)):
                num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
                num_word_total += 1
                dist = editdistance.eval(recognized[i], batch.gt_texts[i])
                num_char_err += dist
                num_char_total += len(batch.gt_texts[i])
                print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gt_texts[i] + '"', '->',
                      '"' + recognized[i] + '"')

        # print validation result
        char_error_rate = num_char_err / num_char_total
        word_accuracy = num_word_ok / num_word_total
        print(f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.')
        return char_error_rate, word_accuracy


    def infer(self, model: Model, fn_img: Path) -> None:
        """Recognizes text in image provided by file path."""
        img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
        assert img is not None

        preprocessor = Preprocessor(self.get_img_size(), dynamic_width=True, padding=16)
        img = preprocessor.process_img(img)

        batch = Batch([img], None, 1)
        recognized, probability = model.infer_batch(batch, True)
        print(f'Recognized: "{recognized[0]}"')
        print(f'Probability: {probability[0]}')


def main():
    """Main function."""
    args = BaseConfig().get_args()
    trainingEval_obj = trainingEval(args)
    # set chosen CTC decoder
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping[args.decoder]

    # train or validate on IAM dataset
    if args.mode in ['train', 'validate']:
        # load training data, create TF model
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)
        char_list = loader.char_list

        # when in line mode, take care to have a whitespace in the char list
        if args.line_mode and ' ' not in char_list:
            char_list = [' '] + char_list

        # save characters of model for inference mode
        open(FilePaths.fn_char_list, 'w').write(''.join(char_list))

        # save words contained in dataset into file
        open(FilePaths.fn_corpus, 'w').write(' '.join(loader.train_words + loader.validation_words))

        # execute training or validation
        if args.mode == 'train':
            model = Model(args, char_list, decoder_type)
            trainingEval_obj.train(model, loader, line_mode=args.line_mode, early_stopping=args.early_stopping)
        elif args.mode == 'validate':
            model = Model(args, char_list, decoder_type, must_restore=True)
            trainingEval_obj.validate(model, loader, args.line_mode)

    # infer text on test image
    elif args.mode == 'infer':
        model = Model(args, list(open(FilePaths.fn_char_list).read()), decoder_type, must_restore=True, dump=args.dump)
        trainingEval_obj.infer(model, args.img_file)

if __name__ == '__main__':
    main()

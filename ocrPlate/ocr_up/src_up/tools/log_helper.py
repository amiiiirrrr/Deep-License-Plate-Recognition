#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=logging-fstring-interpolation

"""
log_helper.py is a file to write methods which use for better log
"""

import logging

__author__ = "Amir Mousavi"
__project__ = "Part company, question Matching"
__credits__ = ["Amir Mousavi"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Amir Mousavi"
__email__ = "azmusavi19@gmail.com"
__status__ = "Production"

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def count_parameters(input_model):
    """
    count_parameters method is written for calculate number of model's parameter
    :param input_model: model
    :return:
        num_parameters: number of model parameters
    """
    num_parameters = sum(p.numel() for p in input_model.parameters() if p.requires_grad)
    return num_parameters


def process_time(s_time, e_time):
    """
    process_time method is written for calculate time
    :param s_time: start time
    :param e_time: end time
    :return:
        elapsed_mins: Minutes of process
        elapsed_secs: Seconds of process
    """
    elapsed_time = e_time - s_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def model_result_log(train_log_dict, valid_log_dict, test_log_dict):
    """
    model_result_log method is writen for show model result on each epoch
    :param train_log_dict: dictionary of train data result
    :param valid_log_dict: dictionary of validation data result
    :param test_log_dict: dictionary of test data result
    """
    logging.info(f"\tTrain. Loss: {train_log_dict['loss']:.4f} | "
                 f"Train. Acc: {train_log_dict['acc'] * 100:.2f}% | ")

    logging.info(f"\t Val. Loss: {valid_log_dict['loss']:.4f} |  "
                 f"Val. Acc: {valid_log_dict['acc'] * 100:.2f}% | ")

    logging.info(f"\t Test. Loss: {test_log_dict['loss']:.4f} |  "
                 f"Test. Acc: {test_log_dict['acc'] * 100:.2f}% | ")

    # logging.info(f"\t Train. Precision: {train_log_dict['precision']}")
    # logging.info(f"\t Train. Recall: {train_log_dict['recall']}")
    # logging.info(f"\t Train. F1_Score: {train_log_dict['f-score']}")
    logging.info(f"\t Train. Total F1 score: {train_log_dict['total_fscore']}")

    # logging.info(f"\t Val. Precision: {valid_log_dict['precision']}")
    # logging.info(f"\t Val. Recall: {valid_log_dict['recall']}")
    # logging.info(f"\t Val. F1_Score: {valid_log_dict['f-score']}")
    logging.info(f"\t Val. Total F1 score: {valid_log_dict['total_fscore']}")

    # logging.info(f"\t Test. Precision: {test_log_dict['precision']}")
    # logging.info(f"\t Test. Recall: {test_log_dict['recall']}")
    # logging.info(f"\t Test. F1_Score: {test_log_dict['f-score']}")
    logging.info(f"\t Test. Total F1 score: {test_log_dict['total_fscore']}")

    logging.info("_____________________________________________________________________\n")


def model_result_save(log_file, train_log_dict, valid_log_dict, test_log_dict):
    """
    model_result_save method is writen for save model result on each epoch
    :param log_file: text log file
    :param train_log_dict: dictionary of train data result
    :param valid_log_dict: dictionary of validation data result
    :param test_log_dict: dictionary of test data result
    """
    # log_file.write(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n")
    log_file.write(f"\tTrain. Loss: {train_log_dict['loss']:.4f} | "
                   f"Train. Acc: {train_log_dict['acc'] * 100:.2f}% | ")

    log_file.write(f"\t Val. Loss: {valid_log_dict['loss']:.4f} |  "
                   f"Val. Acc: {valid_log_dict['acc'] * 100:.2f}% | ")

    log_file.write(f"\t Test. Loss: {test_log_dict['loss']:.4f} |  "
                   f"Test. Acc: {test_log_dict['acc'] * 100:.2f}%\n | ")

    # log_file.write(f"\t Train. Precision: {train_log_dict['precision']}\n")
    # log_file.write(f"\t Train. Recall: {train_log_dict['recall']}\n")
    log_file.write(f"\t Train. F1_Score: {train_log_dict['f-score']}\n")
    # log_file.write(f"\t Train. Total F1 score: {train_log_dict['total_fscore']}\n")

    # log_file.write(f"\t Val. Precision: {valid_log_dict['precision']}\n")
    # log_file.write(f"\t Val. Recall: {valid_log_dict['recall']}\n")
    log_file.write(f"\t Val. F1_Score: {valid_log_dict['f-score']}\n")
    # log_file.write(f"\t Val. Total F1 score: {valid_log_dict['total_fscore']}\n")

    # log_file.write(f"\t Test. Precision: {test_log_dict['precision']}\n")
    # log_file.write(f"\t Test. Recall: {test_log_dict['recall']}\n")
    log_file.write(f"\t Test. F1_Score: {test_log_dict['f-score']}\n")
    # log_file.write(f"\t Test. Total F1 score: {test_log_dict['total_fscore']}\n")

    log_file.write("____________________________________________________________\n")
    log_file.flush()


def epoch_time(start_time, end_time):
    """
    Returns elapsed_mins and elapsed_secs
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

import logging
from typing import Optional
import torch
import torch.nn as nn
from os.path import exists
from os import makedirs


class TrainLogger(object):

    def __init__(self, experiment_type, experiment_no):
        """
        Custom logger to log hyperparameters, results and save model checkpoints
        :param experiment_type:
        :param experiment_no:
        """
        if not exists("./logs"):
            makedirs("./logs")
        level = logging.INFO
        logger_name = f"logs/{experiment_type}_experiment_no_{experiment_no}.log"
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        format_string = "%(message)s"
        log_format = logging.Formatter(format_string)
        file_handler = logging.FileHandler(logger_name, mode = 'a')
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        self.logger = logger
        self._exp_type = experiment_type
        self._exp_no = experiment_no

    def log_hyperparameters(self, **kwargs) -> None:
        """
        Log hyperparameter values
        :param kwargs:
        :return:
        """
        for kw, value in zip(kwargs, kwargs.values()):
            self.logger.info(f"{kw} = {value}")

    def log_result(self, result: str, model: Optional[nn.Module] = None, epoch: Optional[int] = None) -> None:
        """
        Log train or validation result
        :param result: string of result
        :param model: Optional - if supplied model checkpoint will be saved
        :param epoch: Optional - required if model is supplied
        :return: nothing
        """
        self.logger.info(result)
        if model:
            output_dir = f"./saved_models/{self._exp_type}/{self._exp_no}/{epoch}"
            if not exists(output_dir):
                makedirs(output_dir)
            output_loc = f"{output_dir}/epoch_{epoch}"
            torch.save(model.state_dict(), output_loc)
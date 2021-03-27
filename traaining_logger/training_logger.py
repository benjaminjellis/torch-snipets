import logging


class TrainLogger(object):

    def __init__(self, experiment_type, experiment_no):
        level = logging.INFO
        logger_name = f"{experiment_type}_exp_no_{experiment_no}.log"
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        format_string = ("%(message)s")
        log_format = logging.Formatter(format_string)
        file_handler = logging.FileHandler(logger_name, mode = 'a')
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        self.logger = logger

    def log_hyperparameters(self, **kwargs):
        for kw, value in zip(kwargs, kwargs.values()):
            self.logger.info(f"{kw} = {value}")

    def log_result(self, result: str):
        self.logger.info(result)
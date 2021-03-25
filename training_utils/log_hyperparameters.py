import logging

def log_hyperparams(**kwargs):
    """
    log hyperparameter values
    :param kwargs: hyperparameters to log
    :return:
    """
    for kw, value in zip(kwargs, kwargs.values()):
        logging.info(f"{kw} = {value}")
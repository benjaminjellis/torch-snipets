import torch
import torch.nn as nn
from os.path import exists
from os import makedirs


def save_checkpoint(model: nn.Module, experiment_type: str, experiment_number: int, epoch: int) -> None:
    """
    save model checkpoint to disk
    :param model: nn.Module
    :param experiment_type: type of experiment
    :param experiment_number: uid of experiemnt
    :param epoch: epcoh of the train job
    :return: nothing
    """
    output_dir = f"./saved_models/{experiment_type}/{experiment_number}/{epoch}"
    if not exists(output_dir):
        makedirs(output_dir)
    output_loc = f"{output_dir}/epoch_{epoch}"
    torch.save(model.state_dict(), output_loc)
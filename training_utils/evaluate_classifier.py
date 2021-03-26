"""
Evaluation function for top-1, top-k, and f1 score
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import logging
from typing import Optional


def evaluate_model(val_dataloader: DataLoader, device: str, model: nn.Module, k: Optional[int] = 5) -> tuple:
    """
    Evaluate model on a data loader
    :param k:
    :param val_dataloader: validation dataloader
    :param device: cuda or cpu
    :param model: nn.Module
    :return: top-1 accuracy, top-5 accuracy, f1_score
    """
    predictions_stacked = []
    targets_stacked = []

    correct_top_1 = 0
    correct_top_5 = 0
    total = 0

    with torch.no_grad():
        model.eval()
        for data in val_dataloader:
            images, labels = (d.to(device) for d in data)
            outputs = model(images)
            preds_top_1 = torch.argmax(outputs, 1)
            preds_top_5 = torch.topk(outputs, k, dim = 1)[1]
            total += outputs.shape[0]
            correct_top_1 += (preds_top_1 == labels).sum().item()

            for i in range(preds_top_5.shape[0]):
                if labels[i] in preds_top_5[i]:
                    correct_top_5 += 1

            targets_stacked += labels.tolist()
            predictions_stacked += preds_top_1.tolist()

    top_1_acc = correct_top_1 / total * 100
    top_5_acc = correct_top_5 / total * 100
    val_f1_score = f1_score(targets_stacked, predictions_stacked, average = "micro")
    val_result = f"Val | top-1 acc: {top_1_acc} | top-5 acc: {top_5_acc} | f1 score: {val_f1_score}"
    print(val_result)
    logging.info(val_result)
    return top_1_acc, top_5_acc, val_f1_score
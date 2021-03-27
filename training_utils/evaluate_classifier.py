"""
Evaluation function for top-1, top-k, f1 score, precision and recall
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Optional


def evaluate_model(val_dataloader: DataLoader,
                   device: str,
                   model: nn.Module,
                   k: Optional[int] = None,
                   logger = None) -> tuple:
    """
    Evaluate model on a data loader
    :param val_dataloader: validation dataloader
    :param device: cuda or cpu
    :param model: nn.Module
    :param k: for top-k accuracy
    :param logger: logger
    :return: top-1 accuracy, top-k accuracy, f1_score, precision, recall
    """
    if not k:
        k = 5

    predictions_stacked = []
    targets_stacked = []

    correct_top_1 = 0
    correct_top_k = 0
    total = 0

    with torch.no_grad():
        model.eval()
        for data in val_dataloader:
            images, labels = (d.to(device) for d in data)
            outputs = model(images)
            preds_top_1 = torch.argmax(outputs, 1)
            preds_top_k = torch.topk(outputs, k, dim = 1)[1]
            total += outputs.shape[0]
            correct_top_1 += (preds_top_1 == labels).sum().item()

            for i in range(preds_top_k.shape[0]):
                if labels[i] in preds_top_k[i]:
                    correct_top_k += 1

            targets_stacked += labels.tolist()
            predictions_stacked += preds_top_1.tolist()

    top_1_acc = correct_top_1 / total * 100
    top_k_acc = correct_top_k / total * 100
    val_f1_score = f1_score(targets_stacked, predictions_stacked, average = "micro")
    val_precision = precision_score(targets_stacked, predictions_stacked, average = "micro")
    val_recall = recall_score(targets_stacked, predictions_stacked, average = "micro")
    val_result = f"Val | top-1 acc: {top_1_acc} | top-{k} acc: {top_k_acc} | f1 score: {val_f1_score} | precision {val_precision} | recall {val_recall}"
    print(val_result)
    if logger:
        logger.log_result(val_result)
    return top_1_acc, top_k_acc, val_f1_score, val_precision, val_recall
import torch


@torch.jit.script
def one_hot_label_smoothing(predictions: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.1):
    """
    One-hot encode a target and smooth labels
    :param predictions:
    :param targets:
    :param smoothing:
    :return:
    """
    n_class = predictions.shape[1]
    one_hot = torch.full_like(predictions, fill_value = smoothing / (n_class - 1))
    one_hot.scatter_(dim = 1, index = targets.unsqueeze(1), value = 1.0 - smoothing)
    return one_hot

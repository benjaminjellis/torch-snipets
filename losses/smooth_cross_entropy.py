import torch
import torch.nn.functional as F
from losses.label_smoothing import one_hot_label_smoothing


@torch.jit.script
def smooth_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.1):
    """
    Smooth Cross Entropy Loss
    :param predictions: float tensor of shape [N,C]
    :param targets: float tensor of shape [N]
    :param smoothing: smoothing val
    :return: float tensor of shape [1]
    """
    one_hot = one_hot_label_smoothing(predictions = predictions, targets = targets, smoothing = smoothing)
    log_prob = F.log_softmax(predictions, dim =1)
    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1).mean()

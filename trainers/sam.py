import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from optimisers import SAM

from torch.utils.data import DataLoader
from trainers import evaluate_classifier

from traaining_logger import TrainLogger
from tqdm import tqdm
from typing import Optional, Callable


def train(experiment_no: int,
          experiment_type: str,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          model: nn.Module,
          loss_func: Callable,
          lr: float,
          epochs: int,
          rho: float,
          params_to_train: Optional[int] = None,
          k: Optional[int] = None
          ) -> None:
    """
    Trainer for model using SAM + SGD with once cycle lr schedule on single gpu
    :param experiment_no: uid for saving outputs into log and saving model checkpoint
    :param experiment_type:
    :param val_dataloader: validation dataloader
    :param train_dataloader: train dataloader
    :param model:
    :param loss_func:
    :param lr: learning rate
    :param rho: rho param for SAM
    :param epochs: number of epochs to train for
    :param params_to_train: Optional - number of layers to train, if 1 - just the top layer
    :param k: Optional - top-k for evaluation (default is 5)
    :return: nothing
    """
    log = TrainLogger(experiment_type = experiment_type, experiment_no = experiment_no)

    # log hyperparams
    log.log_hyperparameters(lr = lr, rho = rho, epochs = epochs,
                            batch_size = train_dataloader.batch_size,
                            model_name = type(model).__name__,
                            params_to_train = params_to_train)

    torch.manual_seed(0)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    total_params = sum(p.numel() for p in model.parameters())

    if params_to_train:
        # set only the last params_to_train params to be trainable
        count = 0
        for param in model.parameters():
            if total_params - count > params_to_train + 1:
                param.requires_grad = False
            count += 1

    model = model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    base_optimizer = optim.SGD

    optimizer = SAM(trainable_params,
                    base_optimizer,
                    rho = rho,
                    lr = lr,
                    momentum = 0.9,
                    weight_decay = 1e-5)

    lr_sched = OneCycleLR(optimizer = optimizer,
                          max_lr = lr,
                          epochs = epochs,
                          steps_per_epoch = len(train_dataloader))

    log.log_hyperparameters(optimizer = type(optimizer).__name__,
                            lr_sched = type(lr_sched).__name__)

    for epoch in range(epochs):
        train_dataloader = tqdm(train_dataloader)
        running_loss = 0
        for i, data in enumerate(train_dataloader):
            model.train()
            images, labels = (d.to(device) for d in data)

            # first forward-backward step
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.mean().backward()
            optimizer.first_step(zero_grad = True)

            # second forward-backward step
            outputs2 = model(images)
            loss2 = loss_func(outputs2, labels)
            loss2.mean().backward()
            optimizer.second_step(zero_grad = True)

            running_loss = loss.mean() + loss2.mean()

            lr_sched.step()

        train_result = f"Train loss after {epoch + 1} epochs: {running_loss}"
        print(train_result)
        log.log_result(train_result, model = model, epoch = epoch)

        evaluate_classifier(val_dataloader = val_dataloader, device = device, model = model,
                            k = k, logger = log)

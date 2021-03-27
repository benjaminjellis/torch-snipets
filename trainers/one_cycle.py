import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
from traaining_logger import TrainLogger
from tqdm import tqdm
from trainers import evaluate_classifier
from typing import Optional, Callable


def train(experiment_no: int,
          experiment_type: str,
          model: nn.Module,
          optimizer: Callable,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          lr: float,
          epochs: int,
          params_to_train: Optional[int] = None,
          k: Optional[int] = 5) -> None:
    """
    Simple blueprint for a trainer using OneCyleLR
    :param experiment_no: used to save log and checkpoints
    :param experiment_type: short string to used to name output folders
    :param model: nn.Module - the model to train
    :param optimizer: optimiser
    :param val_dataloader: validation dataloader
    :param train_dataloader: train dataloader
    :param lr: learning rate
    :param epochs: number of epochs to train for
    :param params_to_train: number of layers to train, if 1 - just the top layer
    :param k: Optional - top-k for evaluation (default is 5)
    :return: nothing
    """
    log = TrainLogger(experiment_type = experiment_type, experiment_no = experiment_no)
    # log hyperparams
    log.log_hyperparameters(lr = lr,
                            epochs = epochs,
                            batch_size = train_dataloader.batch_size,
                            model = type(model).__name__,
                            params_to_train = params_to_train)

    torch.manual_seed(0)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    total_params = sum(p.numel() for p in model.parameters())
    # set only the last params_to_train params to be trainable
    if params_to_train:
        count = 0
        for param in model.parameters():
            if total_params - count > params_to_train + 1:
                param.requires_grad = False
            count += 1

    model = model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optimizer(trainable_params, lr = lr)
    criterion = nn.CrossEntropyLoss()
    lr_sched = lrs.OneCycleLR(optimizer = optimizer, max_lr = lr, epochs = epochs,
                              steps_per_epoch = len(train_dataloader))

    log.log_hyperparameters(optimizer = type(optimizer).__name__,
                            lr_sched = type(lr_sched).__name__)

    for epoch in range(epochs):
        train_dataloader = tqdm(train_dataloader)
        running_loss = 0
        for i, data in enumerate(train_dataloader):
            model.train()
            images, labels = (d.to(device) for d in data)

            optimizer.zero_grad()

            # first forward
            outputs = model(images)
            # loss
            loss = criterion(outputs, labels)
            # backward
            loss.backward()
            # optimise
            optimizer.step()
            # lr step
            lr_sched.step()

            running_loss += loss.item()

        train_result = f"Train loss after {epoch + 1} epochs: {running_loss}"
        print(train_result)
        log.log_result(train_result, model = model, epoch = epoch)

        evaluate_classifier(val_dataloader = val_dataloader, device = device, model = model, k = k, logger = log)
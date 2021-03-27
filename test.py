from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision.models import resnet18
from trainers import train_sam_one_cycle
from losses import smooth_cross_entropy


class DummyDataset(Dataset):

    def __len__(self):
        return 30

    def __getitem__(self, item):
        return torch.rand(3, 20, 20), 2


train_dataset = DummyDataset()
test_dataset = DummyDataset()

train_dataloader = DataLoader(dataset = train_dataset,
                              batch_size = 3,
                              )

test_dataloader = DataLoader(dataset = test_dataset,
                             batch_size = 3,
                             )

num_classes = 10

train_sam_one_cycle(experiment_no = 1,
                    experiment_type = "hotel-id",
                    train_dataloader = train_dataloader,
                    val_dataloader = test_dataloader,
                    model = nn.Sequential(resnet18(pretrained = True),
                                          nn.Linear(in_features =resnet18().fc.out_features,
                                                    out_features = 10)),
                    loss_func = smooth_cross_entropy,
                    lr = 0.016,
                    epochs = 1,
                    rho = 0.05,
                    k = 3
                    )

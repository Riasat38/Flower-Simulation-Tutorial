import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader, random_split


def prepare_dataset(num_partitions, batch_size, validation_split=0.1):

    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    data_path = "./data"

    trainset = MNIST(data_path, train=True, download=True, transform=transform)
    testset = MNIST(data_path, train=False, download=True, transform=transform)

    # Split into train/val (single partitioned dataset; caller can shard later)
    num_train = int((1.0 - validation_split) * len(trainset))
    num_val = len(trainset) - num_train
    train_subset, val_subset = random_split(
        trainset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(2023),
    )

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def preload_mnist_to_ram(train=True):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST(
        root="./data", train=train, download=True, transform=transform
    )

    # Transformar todo de una vez
    loader = DataLoader(dataset, batch_size=len(dataset), num_workers=0)
    images, labels = next(iter(loader))

    return torch.utils.data.TensorDataset(images, labels)  # tensores en RAM


def get_mnist_dataloader(train=True, batch_size=5000):
    tensor_dataset = preload_mnist_to_ram(train=train)
    return DataLoader(
        tensor_dataset, batch_size=batch_size, shuffle=train, num_workers=0
    )


def get_distributed_mnist_dataloader(train=True, batch_size=5000):
    tensor_dataset = preload_mnist_to_ram(train=train)
    sampler = DistributedSampler(tensor_dataset)

    return DataLoader(
        tensor_dataset, batch_size=batch_size, sampler=sampler, num_workers=0
    )

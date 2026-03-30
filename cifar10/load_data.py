import matplotlib.pyplot as plt
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler


def preload_cifar10_to_ram(train=True, gray=False, batch_size=5000, normalize=True):
    if gray:
        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
                if normalize
                else transforms.Normalize(mean=(0,), std=(1,)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)
                    if normalize
                    else transforms.Normalize(mean=(0,), std=(1,)),
                ),
            ]
        )

    dataset = datasets.CIFAR10(
        root="./data", train=train, download=True, transform=transform
    )

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    images, labels = next(iter(loader))

    return torch.utils.data.TensorDataset(images, labels)  # tensores en RAM


def get_cifar10_dataloader(train=True, gray=False, batch_size=5000, normalize=True):
    tensor_dataset = preload_cifar10_to_ram(
        train=train, gray=gray, batch_size=batch_size, normalize=normalize
    )
    return DataLoader(
        tensor_dataset, batch_size=batch_size, shuffle=train, num_workers=0
    )


def get_distributed_cifar10_dataloader(
    train=True, gray=True, batch_size=5000, normalize=True
):
    tensor_dataset = preload_cifar10_to_ram(
        train=train, gray=gray, batch_size=batch_size, normalize=normalize
    )
    sampler = DistributedSampler(tensor_dataset)

    return DataLoader(
        tensor_dataset, batch_size=batch_size, sampler=sampler, num_workers=0
    )


def plot_images(gray=True, size=10):
    dataloader = get_cifar10_dataloader(
        gray=gray, train=True, batch_size=size, normalize=False
    )

    images, labels = next(iter(dataloader))
    fig, axes = plt.subplots(1, len(images), figsize=(12, 3))

    for i, (image, label) in enumerate(zip(images, labels)):
        if gray:
            axes[i].imshow(image.squeeze(), cmap="gray")
        else:
            axes[i].imshow(image.permute(1, 2, 0))

        axes[i].set_title(label)
        axes[i].axis("off")

    plt.show()

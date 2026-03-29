from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def load_mnist_sampler(batch_size: int = 64, num_workers: int = 0):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    sampler = DistributedSampler(dataset, shuffle=True)  # lee rank y world_size solo

    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )


def load_mnist_test_sampler(batch_size: int = 64, num_workers: int = 0):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

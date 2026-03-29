import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def load_cifar10_sampler(
    gray=True, train=True, batch_size: int = 64, num_workers: int = 0, normalize=True
):
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
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
                )
                if normalize
                else transforms.Normalize(mean=(0,), std=(1,)),
            ]
        )

    dataset = datasets.CIFAR10(
        root="./data", train=train, download=True, transform=transform
    )

    sampler = DistributedSampler(dataset, shuffle=train)  # lee rank y world_size solo

    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )


def plot_images(gray=False, size=10):
    dataloader = load_cifar10_sampler(
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

import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary

from cifar10.model import Cifar10Model
from evaluates import evaluate_classification
from trainers import train_ddp_grad_avarage
from utils import plot_confusion_matrix, plot_grid

from .load_sampler import load_cifar10_sampler, plot_images


def setup(rank: int, world_size: int, master_addr: str, master_port: str):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(
        backend="gloo" if not torch.cuda.is_available() else "nccl",
        rank=rank,
        world_size=world_size,
    )


def train(
    rank: int = 0,
    world_size: int = 1,
    master_addr: str = "localhost",
    master_port: str = "9999",
    conv: bool = False,
    gray: bool = True,
    lr: float = 0.01,
    batch_size: int = 256,
    num_workers: int = 0,
    epochs: int = 20,
):
    setup(rank, world_size, master_addr, master_port)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Cifar10Model(gray, conv)
    summary(model, input_size=(batch_size, 1 if gray else 3, 32, 32))

    model = DDP(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    dataloader = load_cifar10_sampler(gray, True, batch_size, num_workers)

    if rank == 0:
        plot_images(gray)

    history = []

    for epoch in range(epochs):
        dataloader.sampler.set_epoch(epoch)  # shuffle

        avg_loss, avg_acc, gnorm, elapsed = train_ddp_grad_avarage(
            model,
            dataloader,
            optimizer,
            criterion,
            rank,
            world_size,
        )

        throughput = (len(dataloader.dataset) / world_size) / elapsed

        if rank == 0:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Acc: {avg_acc * 100:.2f}% | "
                f"Time: {elapsed:.2f}s | "
                f"Throughput: {throughput:.0f} samples/s"
            )
            history.append((avg_loss, avg_acc, gnorm, elapsed, throughput))
        else:
            print(f"Epoch {epoch + 1:02d}/{epochs} | Time: {elapsed:.2f}s")

    if rank == 0:
        testloader = load_cifar10_sampler(gray, False, batch_size, num_workers)

        test_acc, conf_matrix = evaluate_classification(model, testloader, 10, device)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")

        plot_grid(
            history,
            ["Loss", "Accuracy", "Gradient Norm", "Time", "Throughput"],
            2,
        )
        plot_confusion_matrix(conf_matrix)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank", type=int, required=True, help="rank de esta PC (0, 1, 2...)"
    )
    parser.add_argument("--world-size", type=int, required=True, help="total de PCs")
    parser.add_argument(
        "--master-addr", type=str, default="127.0.0.1", help="IP de la PC con rank 0"
    )
    parser.add_argument("--master-port", type=str, default="9999")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="número de procesos de carga de datos",
    )
    parser.add_argument("--conv", action="store_true", help="usar modelo convolucional")
    parser.add_argument("--rgb", action="store_true", help="usar imágenes en RGB")

    args = parser.parse_args()

    train(
        args.rank,
        args.world_size,
        args.master_addr,
        args.master_port,
        args.conv,
        not args.rgb,
        args.lr,
        args.batch_size,
        args.num_workers,
        args.epochs,
    )

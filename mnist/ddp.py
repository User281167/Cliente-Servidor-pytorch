import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary

from evaluates import evaluate_classification
from trainers import train_ddp_grad_avarage
from utils import format_elapse, plot_confusion_matrix, plot_grid, time_wrapper

from .load_data import get_distributed_mnist_dataloader, get_mnist_dataloader
from .model import MnistModel


def setup(rank: int, world_size: int, master_addr: str, master_port: str):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(
        backend="gloo" if not torch.cuda.is_available() else "nccl",
        rank=rank,
        world_size=world_size,
    )


@time_wrapper
def train(
    rank: int = 0,
    world_size: int = 1,
    master_addr: str = "localhost",
    master_port: str = "9999",
    conv: bool = False,
    lr: float = 0.01,
    batch_size: int = 256,
    epochs: int = 20,
):
    setup(rank, world_size, master_addr, master_port)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MnistModel(conv)
    summary(model, input_size=(batch_size, 1, 28, 28))
    model = DDP(model)  # modelo distribuido

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    dataloader = get_distributed_mnist_dataloader(batch_size=batch_size)

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
            if epoch % (epochs // 10 or 1) == 0 or epoch == epochs - 1 or epoch == 0:
                print(
                    f"Epoch {epoch + 1:02d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc * 100:.2f}% | "
                    f"GNorm: {gnorm:.4f} | Throughput: {throughput:.0f} samples/s | Time: {format_elapse(elapsed)}"
                )

            history.append((avg_loss, avg_acc, gnorm, elapsed, throughput))
        else:
            print(f"Epoch {epoch + 1:02d}/{epochs} | Time: {format_elapse(elapsed)}")

    # indicar al master que este proceso ha terminado
    dist.destroy_process_group()

    if rank == 0:
        testloader = get_mnist_dataloader(train=False)

        test_acc, conf_matrix = evaluate_classification(model, testloader, 10, device)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")

        plot_grid(
            history,
            ["Loss", "Accuracy", "Gradient Norm", "Time", "Throughput"],
            2,
        )
        plot_confusion_matrix(conf_matrix)


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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--conv", action="store_true", help="usar modelo convolucional")

    args = parser.parse_args()

    train(
        args.rank,
        args.world_size,
        args.master_addr,
        args.master_port,
        args.conv,
        args.lr,
        args.batch_size,
        args.epochs,
    )

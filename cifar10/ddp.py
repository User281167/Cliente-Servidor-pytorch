import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary

from cifar10.model import Cifar10Model
from evaluates import evaluate_classification, evaluate_classification_metrics
from trainers import train_ddp_grad_avarage
from utils import format_elapse, plot_confusion_matrix, plot_grid, time_wrapper

from .load_data import (
    cifar10_classes,
    get_cifar10_dataloader,
    get_distributed_cifar10_dataloader,
    plot_images,
)
from .save import save_history


def setup(rank: int, world_size: int, master_addr: str, master_port: str) -> None:
    # Indicar a pytorch sobre el worker con rank 0
    # Indicar el id del rank local
    # Indicar Uso de cpu o gpu
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
    gray: bool = True,
    lr: float = 0.001,
    batch_size: int = 256,
    epochs: int = 20,
    test: bool = False,
    save_path: str | None = None,
) -> None:
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    setup(rank, world_size, master_addr, master_port)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Cifar10Model(gray, conv)

    # gray solo tiene un canal de color
    summary(model, input_size=(batch_size, 1 if gray else 3, 32, 32))
    model = DDP(model)  # modelo datos distribuido

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataloader = get_distributed_cifar10_dataloader(gray=gray, batch_size=batch_size)
    testloader = get_cifar10_dataloader(gray=gray, train=False, batch_size=batch_size)

    if rank == 0:
        plot_images(gray)

    history = []

    for epoch in range(epochs):
        dataloader.sampler.set_epoch(epoch)

        avg_loss, avg_acc, gnorm, elapsed = train_ddp_grad_avarage(
            model,
            dataloader,
            optimizer,
            criterion,
            rank,
            world_size,
        )
        throughput = (len(dataloader.dataset) / world_size) / elapsed

        loss_test = acc_test = None
        if test and rank == 0:
            loss_test, acc_test = evaluate_classification_metrics(
                model,
                testloader,
                device=device,
            )

        if rank == 0:
            if epoch % (epochs // 10 or 1) == 0 or epoch == epochs - 1 or epoch == 0:
                test_log = (
                    f"Test Loss: {loss_test:.4f} | Test Acc: {acc_test * 100:.2f}% | "
                    if test
                    else ""
                )
                print(
                    f"Epoch {epoch + 1:02d}/{epochs} | "
                    f"Loss: {avg_loss:.4f} | Acc: {avg_acc * 100:.2f}% | "
                    f"{test_log}"
                    f"GNorm: {gnorm:.4f} | "
                    f"Throughput: {throughput:.0f} samples/s | "
                    f"Time: {format_elapse(elapsed)}"
                )

            if test:
                history.append(
                    (
                        (avg_loss, loss_test),
                        (avg_acc, acc_test),
                        gnorm,
                        elapsed,
                        throughput,
                    )
                )
            else:
                history.append((avg_loss, avg_acc, gnorm, elapsed, throughput))
        else:
            print(f"Epoch {epoch + 1:02d}/{epochs} | Time: {format_elapse(elapsed)}")
            history.append(elapsed)

    # indicar al master que este proceso ha terminado
    dist.destroy_process_group()

    if save_path is not None:
        save_history(history, save_path, rank, test)

    if rank != 0:
        return

    test_acc, conf_matrix = evaluate_classification(model, testloader, 10, device)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    plot_grid(
        history,
        [
            "Loss Train" if not test else ("Loss", "Train", "Test"),
            "Accuracy Train" if not test else ("Accuracy", "Train", "Test"),
            "Gradient Norm",
            "Time",
            "Throughput",
        ],
        2,
        save_path=save_path,
    )
    plot_confusion_matrix(conf_matrix, save_path=save_path, class_names=cifar10_classes)

    if save_path:
        with open(os.path.join(save_path, "metrics.txt"), "w") as f:
            f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Learning Rate: {lr}\n")
            f.write(f"Conv: {conv}\n")
            f.write(f"Gray: {gray}\n")


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
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--conv", action="store_true", help="usar modelo convolucional")
    parser.add_argument("--rgb", action="store_true", help="usar imágenes en RGB")
    parser.add_argument(
        "--test", action="store_true", help="evaluar en el conjunto de prueba"
    )
    parser.add_argument(
        "--save-path", type=str, default=None, help="ruta para guardar resultados"
    )

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
        args.epochs,
        args.test,
        args.save_path,
    )

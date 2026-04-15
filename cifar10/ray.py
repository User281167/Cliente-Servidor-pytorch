import time

import torch

from trainers.grad_norm import grad_norm


def train_ray_epoch(model, dataloader, optimizer, criterion):
    """
    Versión Ray-compatible del trainer.
    Ya NO necesita rank/world_size ni dist.all_reduce —
    Ray Train + DDP sincronizan los gradientes automáticamente.
    """
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = len(dataloader)

    start_time = time.perf_counter()

    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Mismo truco que antes: acumular gradientes y hacer un solo step
        (loss / n_batches).backward()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    gnorm = grad_norm(model)
    optimizer.step()

    elapsed = time.perf_counter() - start_time

    avg_loss = total_loss / n_batches
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc, gnorm, elapsed


import os

import ray
import torch
import torch.nn as nn
from ray import train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer, prepare_data_loader, prepare_model
from torch.utils.data import DataLoader

from cifar10.model import Cifar10Model
from evaluates import evaluate_classification, evaluate_classification_metrics
from utils import plot_confusion_matrix, plot_grid

from .load_data import (
    cifar10_classes,
    get_cifar10_dataloader,
    plot_images,
    preload_cifar10_to_ram,  # usamos la función base, no la distributed
)
from .save import save_history


def train_func(config: dict):
    """
    Esta función se ejecuta en CADA worker de Ray.
    Ray la lanza en paralelo, le asigna el rank, y gestiona DDP.
    """
    # --- Hiperparámetros desde config ---
    lr = config.get("lr", 0.001)
    batch_size = config.get("batch_size", 256)
    epochs = config.get("epochs", 20)
    gray = config.get("gray", True)
    conv = config.get("conv", False)
    do_test = config.get("test", False)

    # --- Dispositivo: Ray lo asigna automáticamente si use_gpu=True ---
    device = train.get_context().get_local_rank()  # para saber en qué GPU estamos

    # --- Modelo ---
    model = Cifar10Model(gray, conv)
    # prepare_model hace: DDP(model.to(device)) — todo en una línea
    model = prepare_model(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- DataLoader distribuido ---
    # preload_cifar10_to_ram carga el dataset completo como TensorDataset
    # prepare_data_loader le agrega un DistributedSampler automáticamente
    train_dataset = preload_cifar10_to_ram(train=True, gray=gray)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    train_loader = prepare_data_loader(train_loader)  # <- shard automático por worker

    # El test loader solo lo usa el worker 0, pero no hay problema en crearlo en todos
    test_loader = get_cifar10_dataloader(gray=gray, train=False, batch_size=batch_size)

    # --- Loop de entrenamiento ---
    for epoch in range(epochs):
        avg_loss, avg_acc, gnorm, elapsed = train_ray_epoch(
            model, train_loader, optimizer, criterion
        )

        throughput = len(train_loader.dataset) / elapsed

        metrics = {
            "loss": avg_loss,
            "acc": avg_acc,
            "grad_norm": gnorm,
            "throughput": throughput,
            "epoch": epoch,
        }

        # Evaluación en test — solo tiene sentido en el worker 0
        # train.get_context().get_world_rank() == 0 es el equivalente a rank == 0
        if do_test and train.get_context().get_world_rank() == 0:
            loss_test, acc_test = evaluate_classification_metrics(
                model,
                test_loader,
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                criterion,
            )
            metrics["loss_test"] = loss_test
            metrics["acc_test"] = acc_test

        # Ray agrega estas métricas de todos los workers (promedio por defecto)
        # y las guarda en los logs — reemplaza al print del rank 0
        train.report(metrics)

    # No necesitas dist.destroy_process_group() — Ray lo hace al terminar


def run_training(
    conv: bool = False,
    gray: bool = True,
    lr: float = 0.001,
    batch_size: int = 256,
    epochs: int = 20,
    test: bool = False,
    num_workers: int = 2,
    use_gpu: bool = False,
    save_path: str | None = None,
    ray_address: str | None = None,
):
    """
    Punto de entrada principal. Configura y lanza el TorchTrainer.
    """
    if ray_address:
        ray.init(address=ray_address)

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "gray": gray,
            "conv": conv,
            "test": test,
        },
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
            # resources_per_worker={"CPU": 2}  # opcional: cuántos CPUs por worker
        ),
        run_config=RunConfig(
            name="cifar10_ray",
            storage_path=save_path or "~/ray_results",
        ),
    )

    result = trainer.fit()

    print(f"\nEntrenamiento terminado.")
    print(f"Métricas finales: {result.metrics}")

    return result


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--conv", action="store_true")
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Dirección del clúster Ray. Ej: '192.168.1.10:6379'. "
        "Si no se pasa, corre en local.",
    )

    args = parser.parse_args()

    # python -m cifar10.ray --epochs 20 --num-workers 2 --ray-address '192.168.20.246:6379'
    #
    # Nótese que ya NO pasas --rank ni --master-addr:
    # Ray gestiona todo eso internamente
    run_training(
        conv=args.conv,
        gray=not args.rgb,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        test=args.test,
        num_workers=args.num_workers,
        use_gpu=args.gpu,
        save_path=args.save_path,
        ray_address=args.ray_address,
    )

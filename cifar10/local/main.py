import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from evaluates import evaluate_classification
from trainers import train_grad_average
from utils.plots import plot_confusion_matrix, plot_grid

from .load_data import load_cifar10
from .model import Cifar10Conv


def run(GRAY_SCALE=False):
    model = Cifar10Conv(gray=GRAY_SCALE)
    summary(model, input_size=(1, 1 if GRAY_SCALE else 3, 32, 32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    batch_size = 256
    train_loader, test_loader = load_cifar10(batch_size, gray=GRAY_SCALE)

    print("Entrenando con Gradient Averaging...")

    history = []
    for epoch in range(num_epochs):
        loss, acc, gnorm, elapsed, throughput = train_grad_average(
            model, train_loader, optimizer, criterion, device
        )

        print(
            f"Epoch {epoch + 1:02d} | Loss: {loss:.4f} | Acc: {acc * 100:.2f}% | "
            f"GNorm: {gnorm:.4f} | Time: {elapsed:.2f}s | Throughput: {throughput:.0f} samples/s"
        )

        history.append((loss, acc, gnorm, elapsed, throughput))

    test_acc, conf_matrix = evaluate_classification(model, test_loader, device=device)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    plot_grid(history, ["Loss", "Accuracy", "Gradient Norm", "Time", "Throughput"], 3)
    plot_confusion_matrix(conf_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10 model with Gradient Averaging"
    )
    parser.add_argument(
        "--gray", action="store_true", help="Use grayscale images", default=True
    )
    args = parser.parse_args()

    run(GRAY_SCALE=args.gray)

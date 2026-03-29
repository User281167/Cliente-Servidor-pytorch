import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from cifar10.model import Cifar10Model
from evaluates import evaluate_classification
from trainers import train_grad_average
from utils.plots import plot_confusion_matrix, plot_grid

from .load_data import load_cifar10


def train(gray=True, conv=False, epochs=20, batch_size=256):
    model = Cifar10Model(gray=gray, conv=conv)
    summary(model, input_size=(1, 1 if gray else 3, 32, 32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader, test_loader = load_cifar10(batch_size, gray=gray)

    print("Entrenando con Gradient Averaging...")

    history = []
    for epoch in range(epochs):
        loss, acc, gnorm, elapsed, throughput = train_grad_average(
            model, train_loader, optimizer, criterion, device
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs:} | Loss: {loss:.4f} | Acc: {acc * 100:.2f}% | "
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
    parser.add_argument("--rgb", action="store_true", help="Use RGB images")
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=20)
    parser.add_argument("--batch-size", type=int, help="Batch size", default=256)
    parser.add_argument("--conv", action="store_true", help="Use convolutional model")
    args = parser.parse_args()

    train(
        gray=not args.rgb,
        conv=args.conv,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from evaluates import evaluate_classification
from mnist.model import MnistModel
from trainers import train_grad_average
from utils import plot_confusion_matrix, plot_grid

from .load_data import load_mnist


def train(conv=False, epochs=2, batch_size=256):
    model = MnistModel(conv)
    summary(model, input_size=(batch_size, 1, 28, 28))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader, test_loader = load_mnist(batch_size)

    print("Entrenando con Gradient Averaging...")

    history = []
    for epoch in range(epochs):
        loss, acc, gnorm, elapsed, throughput = train_grad_average(
            model, train_loader, optimizer, criterion, device
        )

        print(
            f"Epoch {epoch + 1:02d} | Loss: {loss:.4f} | Acc: {acc * 100:.2f}% | "
            f"GNorm: {gnorm:.4f} | Time: {elapsed:.2f}s | Throughput: {throughput:.0f} samples/s"
        )

        history.append((loss, acc, gnorm, elapsed, throughput))

    test_acc, conf_matrix = evaluate_classification(model, test_loader, 10, device)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    plot_grid(
        history,
        ["Loss", "Accuracy", "Gradient Norm", "Time", "Throughput"],
        2,
    )
    plot_confusion_matrix(conf_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv", action="store_true", help="Use convolutional model")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    args = parser.parse_args()

    train(conv=args.conv, epochs=args.epochs, batch_size=args.batch_size)

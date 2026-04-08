import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from evaluates import evaluate_classification, evaluate_classification_metrics
from trainers import train_grad_average
from utils import format_elapse, plot_confusion_matrix, plot_grid, time_wrapper

from .load_data import get_mnist_dataloader
from .model import MnistModel


@time_wrapper
def train(conv=False, epochs=20, batch_size=256):
    model = MnistModel(conv)
    summary(model, input_size=(batch_size, 1, 28, 28))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = get_mnist_dataloader(batch_size=batch_size)
    test_loader = get_mnist_dataloader(train=False, batch_size=batch_size)

    print("Entrenando con Gradient Averaging...")

    history = []
    for epoch in range(epochs):
        loss, acc, gnorm, elapsed, throughput = train_grad_average(
            model, train_loader, optimizer, criterion, device
        )

        loss_test, acc_test = evaluate_classification_metrics(
            model, test_loader, device, criterion
        )

        if epoch % (epochs // 10 or 1) == 0 or epoch == epochs - 1 or epoch == 0:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | Loss: {loss:.4f} | Acc: {acc * 100:.2f}% | "
                f"Test Loss: {loss_test:.4f} | Test Acc: {acc_test * 100:.2f}% | "
                f"GNorm: {gnorm:.4f} | Throughput: {throughput:.0f} samples/s | Time: {format_elapse(elapsed)} | "
            )

        history.append(((loss, loss_test), (acc, acc_test), gnorm, elapsed, throughput))

    test_acc, conf_matrix = evaluate_classification(model, test_loader, 10, device)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    plot_grid(
        history,
        [
            "Loss - Train/Test",
            "Accuracy - Train/Test",
            "Gradient Norm",
            "Time",
            "Throughput",
        ],
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

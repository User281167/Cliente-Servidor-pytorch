import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from cifar10.model import Cifar10Model
from evaluates import evaluate_classification, evaluate_classification_metrics
from trainers import train_grad_average
from utils.format_elapse import format_elapse, time_wrapper
from utils.plots import plot_confusion_matrix, plot_grid

from .load_data import cifar10_classes, get_cifar10_dataloader, plot_images


@time_wrapper
def train(gray=True, conv=False, epochs=20, batch_size=256, lr=0.001):
    model = Cifar10Model(gray=gray, conv=conv)
    summary(model, input_size=(batch_size, 1 if gray else 3, 32, 32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = get_cifar10_dataloader(train=True, gray=gray, batch_size=batch_size)
    test_loader = get_cifar10_dataloader(train=False, gray=gray, batch_size=batch_size)
    plot_images(gray=gray)

    history = []
    for epoch in range(epochs):
        loss, acc, gnorm, elapsed, throughput = train_grad_average(
            model, train_loader, optimizer, criterion, device
        )

        loss_test, acc_test = evaluate_classification_metrics(
            model, test_loader, device=device
        )

        if epoch % (epochs // 10 or 1) == 0 or epoch == epochs - 1 or epoch == 0:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | Loss: {loss:.4f} | Acc: {acc * 100:.2f}% | "
                f"Test Loss: {loss_test:.4f} | Test Acc: {acc_test * 100:.2f}% | "
                f"GNorm: {gnorm:.4f} | Throughput: {throughput:.0f} samples/s | Time: {format_elapse(elapsed)} | "
            )

        history.append(((loss, loss_test), (acc, acc_test), gnorm, elapsed, throughput))

    test_acc, conf_matrix = evaluate_classification(model, test_loader, device=device)
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
        3,
    )
    plot_confusion_matrix(conf_matrix, class_names=cifar10_classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10 model with Gradient Averaging"
    )
    parser.add_argument("--rgb", action="store_true", help="Use RGB images")
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=20)
    parser.add_argument("--batch-size", type=int, help="Batch size", default=256)
    parser.add_argument("--conv", action="store_true", help="Use convolutional model")
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.001)
    args = parser.parse_args()

    train(
        gray=not args.rgb,
        conv=args.conv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

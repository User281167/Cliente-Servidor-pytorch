import time

import torch

from .grad_norm import grad_norm


def train_grad_average(model, train_loader, optimizer, criterion, device):
    """
    Gradient Averaging:
    - acumula gradientes de todos los batches
    - hace un solo optimizer.step() al final de la epoch
    """
    model.train()

    optimizer.zero_grad()
    start_time = time.perf_counter()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = len(train_loader)

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        (loss / n_batches).backward()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    gnorm = grad_norm(model)
    optimizer.step()

    elapsed = time.perf_counter() - start_time
    throughput = total_samples / elapsed

    avg_loss = total_loss / n_batches
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc, gnorm, elapsed, throughput

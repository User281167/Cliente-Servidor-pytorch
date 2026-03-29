import time

import torch
import torch.distributed as dist

from .grad_norm import grad_norm


def train_ddp_grad_avarage(model, dataloader, optimizer, criterion, rank, world_size):
    model.train()
    optimizer.zero_grad()

    total_loss = torch.tensor(0.0)
    total_correct = torch.tensor(0.0)
    total_samples = torch.tensor(0.0)
    start_time = time.perf_counter()

    n_batches = len(dataloader)

    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        # igual que tu grad average local — un solo step por epoch
        (loss / n_batches).backward()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    gnorm = grad_norm(model)
    optimizer.step()
    elapsed = time.perf_counter() - start_time

    # AllReduce: suma loss/correct/samples de todos los ranks
    # rank 0 tiene las métricas globales reales
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

    if rank == 0:
        avg_loss = (total_loss / (n_batches * world_size)).item()
        avg_acc = (total_correct / total_samples).item()

        return avg_loss, avg_acc, gnorm, elapsed
    else:
        return None, None, gnorm, elapsed

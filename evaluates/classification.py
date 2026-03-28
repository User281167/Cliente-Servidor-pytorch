import torch
from torchmetrics.classification import MulticlassConfusionMatrix


def evaluate_classification(model, test_loader, num_classes=10, device=None):
    """Evalua accuracy y confusion matrix en el test set."""
    model.eval()
    correct = 0
    total = 0
    confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(
        device or torch.device("cpu")
    )

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            confusion_matrix.update(predicted, labels)

    return correct / total, confusion_matrix.compute().cpu()

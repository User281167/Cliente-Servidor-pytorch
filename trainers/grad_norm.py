def grad_norm(model):
    """Calcula la norma L2 de todos los gradientes del modelo."""
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += param.grad.norm(2).item() ** 2

    return total**0.5

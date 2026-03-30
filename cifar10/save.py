import os

import pandas as pd


def save_history(
    history: list,
    save_path: str,
    rank: int,
    test: bool,
) -> None:
    """Guardar el historial de entrenamiento en archivos Excel (por rango de métricas + describe)."""
    if rank == 0:
        if test:
            rows = [
                [loss_train, loss_test, acc_train, acc_test, gnorm, elapsed, throughput]
                for (loss_train, loss_test), (
                    acc_train,
                    acc_test,
                ), gnorm, elapsed, throughput in history
            ]

            columns = [
                "loss_train",
                "loss_test",
                "acc_train",
                "acc_test",
                "gnorm",
                "elapsed",
                "throughput",
            ]
        else:
            rows = history
            columns = ["loss", "acc", "gnorm", "elapsed", "throughput"]
    else:
        rows = history
        columns = ["elapsed"]

    df = pd.DataFrame(rows, columns=columns, dtype=float)
    df.to_excel(
        os.path.join(save_path, f"history_rank_{rank}.xlsx"),
        index=False,
    )

    description = df.describe(percentiles=[0.1, 0.5, 0.9])
    description.to_excel(
        os.path.join(save_path, f"description_rank_{rank}.xlsx"),
        index=True,
    )

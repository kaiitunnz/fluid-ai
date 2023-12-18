import os
from typing import Any, Dict, List, Iterator

import matplotlib.pyplot as plt  # type: ignore
import torch
import torch.nn as nn

from .wrapper import ModelWrapper


def save_model(
    path: str,
    epoch: int,
    model: ModelWrapper,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
):
    """
    Save the model for inference.
    """
    for params in model.parameters():
        params.requires_grad = False
    device = model.device()
    torch.save(
        {
            "epoch": epoch,
            "model": model.cpu(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": criterion,
        },
        path,
    )
    for params in model.parameters():
        params.requires_grad = True
    model.to(device)


def load_model(path: str) -> ModelWrapper:
    return torch.load(path)["model"]


def save_plots(
    save_dir: str,
    train_loss: List[float],
    valid_loss: List[float],
    train_metrics: Dict[str, List[Any]],
    val_metrics: Dict[str, List[Any]],
):
    # metric plots
    for metric, hist in train_metrics.items():
        plt.figure(figsize=(10, 7))
        plt.plot(hist, color="green", linestyle="-", label=metric)
        plt.plot(hist, color="blue", linestyle="-", label=metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"{metric}_hist.png"))

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", linestyle="-", label="train loss")
    plt.plot(valid_loss, color="red", linestyle="-", label="validataion loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_hist.png"))


class BatchLoader:
    batch_size: int
    data: List[torch.Tensor]
    i: int = 0
    length: int

    def __init__(self, batch_size: int, data: List[torch.Tensor]):
        self.batch_size = batch_size
        self.data = data
        self.length = len(self.data)

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.batch_size <= 0:
            return iter([torch.cat(self.data, dim=0)])
        return self

    def __next__(self) -> torch.Tensor:
        if self.i >= self.length:
            raise StopIteration()
        batch = self.data[self.i : self.i + self.batch_size]
        self.i += self.batch_size
        return torch.cat(batch, dim=0)

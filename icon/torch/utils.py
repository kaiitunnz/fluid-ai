import os
from typing import List

import matplotlib.pyplot as plt  # type: ignore
import torch
import torch.nn as nn

from .models import ModelWrapper


def save_model(
    path: str,
    epoch: int,
    model: ModelWrapper,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
):
    torch.save(
        {
            "epoch": epoch,
            "model": model,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": criterion,
        },
        path,
    )


def load_model(path: str) -> ModelWrapper:
    return torch.load(path)["model"]


def save_plots(
    save_dir: str,
    train_acc: List[float],
    valid_acc: List[float],
    train_loss: List[float],
    valid_loss: List[float],
):
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color="green", linestyle="-", label="train accuracy")
    plt.plot(valid_acc, color="blue", linestyle="-", label="validataion accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "accuracy_hist.png"))

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", linestyle="-", label="train loss")
    plt.plot(valid_loss, color="red", linestyle="-", label="validataion loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_hist.png"))

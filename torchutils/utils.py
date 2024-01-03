import os
from typing import Any, Dict, Iterator, List, Optional

import matplotlib.pyplot as plt  # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

from .wrapper import ModelWrapper


def save_model(
    path: str,
    epoch: int,
    model: ModelWrapper,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
):
    """Saves the model checkpoint with some additional states

    Parameters
    ----------
    path : str
        Path to save the checkpoint.
    epoch : int
        Epoch at which the checkpoint is saved.
    model : ModelWrapper
        Model to be saved.
    optimizer : Optimizer
        PyTorch optimizer used to train the model.
    criterion : Module
        PyTorch criterion used to train the model.
    """
    requires_grad = next(model.parameters()).requires_grad
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
        params.requires_grad = requires_grad
    model.to(device)


def load_model(path: str) -> ModelWrapper:
    """Loads the model saved by `save_model()`

    Parameters
    ----------
    path : str
        Path to the saved model.

    Returns
    -------
    ModelWrapper
        Loaded model.
    """
    return torch.load(path)["model"]


def save_plots(
    save_dir: str,
    train_loss: List[float],
    valid_loss: List[float],
    train_metrics: Dict[str, List[Any]],
    val_metrics: Dict[str, List[Any]],
):
    """Saves the plots of training results

    Parameters
    ----------
    save_dir : str
        Path to the directory to save the plots.
    train_loss : List[float]
        Training loss history.
    valid_loss : List[float]
        Validation loss history.
    train_metrics : Dict[str, List[Any]]
        History of training metrics' values.
    val_metrics : Dict[str, List[Any]]
        History of validation metrics' values.
    """
    # metric plots
    metric_set = set(train_metrics.keys())
    metric_set.update(val_metrics.keys())
    for metric in metric_set:
        plt.figure(figsize=(10, 7))
        tmp = train_metrics.get(metric)
        if tmp is not None:
            plt.plot(tmp, color="green", linestyle="-", label="training")
        tmp = val_metrics.get(metric)
        if tmp is not None:
            plt.plot(tmp, color="blue", linestyle="-", label="validation")
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


def get_data_loaders(
    data: Dataset,
    batch_size: int = 16,
    sampler: Optional[Sampler] = None,
    num_workers: int = 4,
) -> DataLoader:
    """Get a data loader for the given dataset.

    Parameters
    ----------
    data : Dataset
        The target dataset.
    batch_size : int
        The batch size.
    sampler : Optional[Sampler]
        Sampler, if used. Otherwise, None.
    num_workers : int
        Number of worker threads to load data.

    Returns
    -------
    DataLoader
        A data loader for the given dataset.
    """
    if sampler is None:
        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
    return DataLoader(
        data,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )


class BatchLoader:
    """
    A lightweight data loader.

    Attributes
    ----------
    batch_size : int
        Batch size.
    data : List[torch.Tensor]
        List of data samples.
    """

    batch_size: int
    data: List[torch.Tensor]

    _i: int = 0

    def __init__(self, batch_size: int, data: List[torch.Tensor]):
        """
        Parameters
        ----------
        batch_size : int
            Batch size.
        data : List[torch.Tensor]
            List of data samples.
        """
        self.batch_size = batch_size
        self.data = data

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.batch_size <= 0:
            return iter([torch.stack(self.data, dim=0)])
        return self

    def __next__(self) -> torch.Tensor:
        if self._i >= len(self.data):
            raise StopIteration()
        batch = self.data[self._i : self._i + self.batch_size]
        self._i += self.batch_size
        return torch.stack(batch, dim=0)

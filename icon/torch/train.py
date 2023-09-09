import json
import os
from tqdm import tqdm  # type: ignore
from typing import List, NamedTuple, Optional, Tuple

import pandas as pd  # type: ignore
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix  # type: ignore
from torch.utils.data import DataLoader

from .models import ModelWrapper
from .utils import load_model, save_model, save_plots


class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float("inf")

    def stop(self, loss: float) -> bool:
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss >= (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class TrainConfig(NamedTuple):
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    num_epochs: int = 50
    early_stopper: Optional[EarlyStopper] = None
    device: torch.device = torch.device("cpu")
    results_dir: Optional[str] = None
    save_period: int = 5
    pretrained: bool = True
    overwrite: bool = False


class EvalConfig(NamedTuple):
    results_dir: str
    classes: Optional[List[str]]
    device: torch.device = torch.device("cpu")
    overwrite: bool = False


def _init_training(train_config: TrainConfig):
    if train_config.results_dir is None:
        return None
    checkpoint_path = _get_checkpoint_path(train_config)
    if checkpoint_path is not None:
        os.makedirs(checkpoint_path, exist_ok=train_config.overwrite)
    plot_path = _get_plot_path(train_config)
    if plot_path is not None:
        os.makedirs(plot_path, exist_ok=train_config.overwrite)


def _init_eval(eval_config: EvalConfig):
    os.makedirs(eval_config.results_dir, exist_ok=eval_config.overwrite)


def _get_checkpoint_path(train_config: TrainConfig) -> Optional[str]:
    if train_config.results_dir is None:
        return None
    return os.path.join(train_config.results_dir, "checkpoints")


def _get_plot_path(train_config: TrainConfig) -> Optional[str]:
    if train_config.results_dir is None:
        return None
    return os.path.join(train_config.results_dir, "plots")


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for image, labels in tqdm(train_loader):
        counter += 1
        image = image.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, labels)
            train_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == labels).sum().item()
            loss.backward()
            optimizer.step()

    epoch_loss = train_running_loss / counter
    epoch_acc = 100.0 * (train_running_correct / len(train_loader.dataset))  # type: ignore
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for _, (image, labels) in enumerate(tqdm(val_loader)):
            counter += 1
            image = image.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(image)
                loss = criterion(outputs, labels)
                valid_running_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
                valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100.0 * (valid_running_correct / len(val_loader.dataset))  # type: ignore
    return epoch_loss, epoch_acc


def train(
    model: ModelWrapper,
    train_config: TrainConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    _init_training(train_config)
    optimizer = train_config.optimizer
    criterion = train_config.criterion
    scheduler = train_config.scheduler
    early_stopper = train_config.early_stopper
    num_epochs = train_config.num_epochs
    device = train_config.device
    save_period = train_config.save_period
    checkpoint_path = _get_checkpoint_path(train_config)
    plot_path = _get_plot_path(train_config)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{total_trainable_params:,} training parameters.")

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    best_val_acc: float = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} of {num_epochs}")
        train_epoch_loss, train_epoch_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_epoch_loss, val_epoch_acc = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            try:
                scheduler.step(metrics=val_epoch_loss)  # type: ignore
            except TypeError:
                scheduler.step()

        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        train_acc.append(train_epoch_acc)
        val_acc.append(val_epoch_acc)
        print(
            f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}"
        )
        print(
            f"Validation loss: {val_epoch_loss:.3f}, validation acc: {val_epoch_acc:.3f}"
        )
        print("-" * 10)

        if checkpoint_path is not None:
            if val_epoch_acc >= best_val_acc and checkpoint_path is not None:
                save_model(
                    os.path.join(checkpoint_path, "best.pt"),
                    epoch,
                    model,
                    optimizer,
                    criterion,
                )
                best_val_acc = val_epoch_acc
            if epoch % save_period == 0:
                save_model(
                    os.path.join(checkpoint_path, f"epoch{epoch + 1}.pt"),
                    epoch,
                    model,
                    optimizer,
                    criterion,
                )

        if early_stopper is not None and early_stopper.stop(-val_epoch_acc):
            print("-" * 10)
            print("No improvement on the validation accuracy. Training stopped.")
            break

    if plot_path is not None:
        save_plots(plot_path, train_acc, val_acc, train_loss, val_loss)
        with open(os.path.join(plot_path, "train_acc_hist.json"), "w") as f:
            json.dump(train_acc, f)
        with open(os.path.join(plot_path, "val_acc_hist.json"), "w") as f:
            json.dump(val_acc, f)
        with open(os.path.join(plot_path, "train_loss_hist.json"), "w") as f:
            json.dump(train_loss, f)
        with open(os.path.join(plot_path, "val_loss_hist.json"), "w") as f:
            json.dump(val_loss, f)
        print(
            f'Training and validation results have been saved to "{train_config.results_dir}"'
        )

    if checkpoint_path is not None:
        best_model = load_model(os.path.join(checkpoint_path, "best.pt"))
        model.load_state_dict(best_model.state_dict())


def eval(
    model: ModelWrapper,
    eval_config: EvalConfig,
    test_loader: DataLoader,
):
    model.eval()
    _init_eval(eval_config)

    results_dir = eval_config.results_dir
    classes = eval_config.classes
    device = eval_config.device
    model.to(device)

    all_preds, all_targets = [], []
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for _, (image, labels) in enumerate(tqdm(test_loader)):
            counter += 1
            image = image.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(image)
                _, preds = torch.max(outputs.data, 1)
                valid_running_correct += (preds == labels).sum().item()
                all_preds.extend(preds.tolist())
                all_targets.extend(labels.tolist())

    print("Overall accuracy:", 100 * (valid_running_correct / len(test_loader.dataset)))  # type: ignore
    conf_mat = confusion_matrix(all_targets, all_preds, normalize=None)
    pd.DataFrame(conf_mat, columns=classes, index=classes).to_csv(
        os.path.join(results_dir, "confusion_matrix.csv")
    )
    normalized_conf_mat = confusion_matrix(all_targets, all_preds, normalize="true")
    pd.DataFrame(normalized_conf_mat, columns=classes, index=classes).to_csv(
        os.path.join(results_dir, "normalized_confusion_matrix.csv")
    )
    print(f'Confusion matrixes have been saved to "{results_dir}"')

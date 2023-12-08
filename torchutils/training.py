import json
import os
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import pandas as pd  # type: ignore
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix  # type: ignore
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm  # type: ignore

from .metric import MetricList
from .utils import load_model, save_model, save_plots
from .wrapper import ModelWrapper


class RandomSampler(Sampler):
    """
    Adapted from https://github.com/ufoym/imbalanced-dataset-sampler/tree/master.
    """

    dataset: Dataset
    indices: List[int]
    num_samples: int
    weights: torch.Tensor

    def __init__(
        self,
        dataset: Dataset,
        labels: List[int],
        ratio: Optional[Union[List[float], Dict[int, float]]],
    ):
        self.indices = list(range(len(dataset)))  # type: ignore
        self.num_samples = len(self.indices)

        df = pd.DataFrame({"label": labels}, index=self.indices)
        df = df.sort_index()

        label_to_count = df["label"].value_counts()
        if ratio is None:
            label_set = set(labels)
            ratio_df = pd.Series([1.0] * len(label_set), index=list(label_set))
        elif isinstance(ratio, list):
            ratio_df = pd.Series(ratio)
        else:
            index, values = zip(*ratio.items())
            ratio_df = pd.Series(values, index=index)
        weights = ratio_df[df["label"]] / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


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
    train_metrics: MetricList
    val_metrics: MetricList
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    num_epochs: int = 50
    early_stopper: Optional[EarlyStopper] = None
    device: torch.device = torch.device("cpu")
    results_dir: Optional[str] = None
    save_period: int = 5
    pretrained: bool = True
    overwrite: bool = False
    verbose: bool = True


class EvalConfig(NamedTuple):
    results_dir: str
    classes: Optional[List[str]]
    metrics: MetricList
    device: torch.device = torch.device("cpu")
    overwrite: bool = False
    verbose: bool = True


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
    model: ModelWrapper,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    metrics: MetricList,
    device: torch.device,
    verbose: bool,
) -> float:
    model.train()
    train_running_loss = 0.0
    metrics.reset()
    counter = 0
    for image, labels in tqdm(train_loader, disable=not verbose):
        counter += 1
        image = image.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            outputs = model(image).squeeze(dim=1)
            loss = criterion(outputs, labels.to(torch.float))
            train_running_loss += loss.item()
            preds = model.get_pred_idx(outputs)
            metrics.update(preds, labels)
            loss.backward()
            optimizer.step()

    epoch_loss = train_running_loss / counter
    return epoch_loss


def validate(
    model: ModelWrapper,
    val_loader: DataLoader,
    criterion: nn.Module,
    metrics: MetricList,
    device: torch.device,
    verbose: bool,
) -> float:
    model.eval()
    valid_running_loss = 0.0
    metrics.reset()
    counter = 0
    with torch.no_grad():
        for _, (image, labels) in enumerate(tqdm(val_loader, disable=not verbose)):
            counter += 1
            image = image.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(image).squeeze(dim=1)
                loss = criterion(outputs, labels.to(torch.float))
                valid_running_loss += loss.item()
                preds = model.get_pred_idx(outputs)
                metrics.update(preds, labels)

    epoch_loss = valid_running_loss / counter
    return epoch_loss


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
    train_metrics = train_config.train_metrics
    val_metrics = train_config.val_metrics
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
    best_val_metric = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} of {num_epochs}")
        train_epoch_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            train_metrics,
            device,
            train_config.verbose,
        )
        val_epoch_loss = validate(
            model, val_loader, criterion, val_metrics, device, train_config.verbose
        )

        if scheduler is not None:
            try:
                scheduler.step(metrics=val_epoch_loss)  # type: ignore
            except TypeError:
                scheduler.step()

        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        train_metrics_values = train_metrics.commit()
        val_metrics_values = val_metrics.commit()
        val_epoch_metric = val_metrics.main_value()
        print(
            f"Training loss: {train_epoch_loss:.3f}, training metrics: {train_metrics_values}"
        )
        print(
            f"Validation loss: {val_epoch_loss:.3f}, validation metrics: {val_metrics_values}"
        )
        print("-" * 10)

        if checkpoint_path is not None:
            if (
                val_metrics.is_current_better(best_val_metric)
                and checkpoint_path is not None
            ):
                save_model(
                    os.path.join(checkpoint_path, "best.pt"),
                    epoch,
                    model,
                    optimizer,
                    criterion,
                )
                best_val_metric = val_epoch_metric
            if epoch % save_period == 0:
                save_model(
                    os.path.join(checkpoint_path, f"epoch{epoch + 1}.pt"),
                    epoch,
                    model,
                    optimizer,
                    criterion,
                )

        if early_stopper is not None and early_stopper.stop(-val_epoch_metric):
            print("-" * 10)
            print("No improvement on the validation accuracy. Training stopped.")
            break

    if plot_path is not None:
        train_metrics_hist = train_metrics.histories()
        val_metrics_hist = val_metrics.histories()
        save_plots(
            plot_path, train_loss, val_loss, train_metrics_hist, val_metrics_hist
        )
        with open(os.path.join(plot_path, "train_acc_hist.json"), "w") as f:
            json.dump(train_metrics_hist, f)
        with open(os.path.join(plot_path, "val_acc_hist.json"), "w") as f:
            json.dump(val_metrics_hist, f)
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
    metrics = eval_config.metrics
    classes = eval_config.classes
    device = eval_config.device
    model.to(device)

    all_preds, all_targets = [], []
    metrics.reset()
    counter = 0
    with torch.no_grad():
        for _, (image, labels) in enumerate(
            tqdm(test_loader, disable=not eval_config.verbose)
        ):
            counter += 1
            image = image.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(image).squeeze(dim=1)
                preds = model.get_pred_idx(outputs)
                metrics.update(preds, labels)
                all_preds.extend(preds.tolist())
                all_targets.extend(labels.tolist())

    labels = None if eval_config.classes is None else range(len(eval_config.classes))
    for metric, value in metrics.values().items():
        print(f"{metric}: {value}")
    conf_mat = confusion_matrix(all_targets, all_preds, labels=labels, normalize=None)
    pd.DataFrame(conf_mat, columns=classes, index=classes).to_csv(
        os.path.join(results_dir, "confusion_matrix.csv")
    )
    normalized_conf_mat = confusion_matrix(
        all_targets, all_preds, labels=labels, normalize="true"
    )
    pd.DataFrame(normalized_conf_mat, columns=classes, index=classes).to_csv(
        os.path.join(results_dir, "normalized_confusion_matrix.csv")
    )
    print(f'Confusion matrixes have been saved to "{results_dir}"')

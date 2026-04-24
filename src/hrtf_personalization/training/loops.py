from __future__ import annotations

from dataclasses import dataclass
import time
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader

from hrtf_personalization.data import PreparedBatch


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 1.0e-4
    device: str = "cpu"
    use_direction: bool = False
    optimizer: str = "adam"
    loss: str = "mse"
    log_interval_batches: int = 50


class Trainer:
    def __init__(self, model: nn.Module, config: TrainingConfig) -> None:
        resolved_device = _resolve_device(config.device)
        self.model = model.to(resolved_device)
        self.config = TrainingConfig(
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            device=resolved_device,
            use_direction=config.use_direction,
            optimizer=config.optimizer,
            loss=config.loss,
            log_interval_batches=config.log_interval_batches,
        )
        self.loss_fn = _build_loss(config.loss)
        self.optimizer = _build_optimizer(model=self.model, optimizer_name=config.optimizer, lr=config.learning_rate)

    def fit(self, dataloader: DataLoader) -> list[float]:
        history: list[float] = []
        self.model.train()
        num_batches = len(dataloader)
        print(
            f"Training on {self.config.device} for {self.config.epochs} epoch(s) "
            f"with {num_batches} batch(es) per epoch.",
            flush=True,
        )
        for epoch_index in range(self.config.epochs):
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            batches = 0
            print(f"Epoch {epoch_index + 1}/{self.config.epochs} started.", flush=True)
            for batch_index, batch in enumerate(dataloader, start=1):
                batch = self._move_batch_to_device(batch)
                prediction = self._predict(batch)
                loss = self.loss_fn(prediction, batch.hrtf)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                batches += 1
                if self.config.log_interval_batches > 0 and (
                    batch_index == 1
                    or batch_index == num_batches
                    or batch_index % self.config.log_interval_batches == 0
                ):
                    running_loss = epoch_loss / batches
                    print(
                        f"  batch {batch_index}/{num_batches} "
                        f"loss={loss.item():.6f} running_loss={running_loss:.6f}",
                        flush=True,
                    )
            average_loss = epoch_loss / max(batches, 1)
            epoch_seconds = time.perf_counter() - epoch_start
            history.append(average_loss)
            print(
                f"Epoch {epoch_index + 1}/{self.config.epochs} complete: "
                f"avg_loss={average_loss:.6f} time={epoch_seconds:.1f}s",
                flush=True,
            )
        return history

    def _predict(self, batch: PreparedBatch) -> torch.Tensor:
        if self.config.use_direction:
            return self.model(batch.anthropometrics, batch.ear_image, batch.direction, batch.ear_side)
        return self.model(batch.anthropometrics, batch.ear_image, batch.ear_side)

    def _move_batch_to_device(self, batch: PreparedBatch) -> PreparedBatch:
        return PreparedBatch(
            subject_ids=batch.subject_ids,
            anthropometrics=batch.anthropometrics.to(self.config.device),
            ear_image=batch.ear_image.to(self.config.device),
            ear_side=batch.ear_side.to(self.config.device),
            hrtf=batch.hrtf.to(self.config.device),
            direction=batch.direction.to(self.config.device),
        )


def _build_optimizer(model: nn.Module, optimizer_name: str, lr: float) -> torch.optim.Optimizer:
    name = optimizer_name.lower()
    if name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    msg = f"Unsupported optimizer: {optimizer_name}"
    raise ValueError(msg)


def _build_loss(loss_name: str) -> nn.Module:
    name = loss_name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "smooth_l1":
        return nn.SmoothL1Loss()
    msg = f"Unsupported loss: {loss_name}"
    raise ValueError(msg)


def _resolve_device(requested_device: str) -> str:
    if requested_device.lower() != "mps":
        return requested_device
    if torch.backends.mps.is_available():
        return "mps"
    warnings.warn(
        "Requested device 'mps' is unavailable in this runtime. Falling back to CPU.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "cpu"

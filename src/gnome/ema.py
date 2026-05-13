"""Exponential Moving Average of model weights.

After each optimizer step, call ema.update(model).
For evaluation, use `with ema.apply(model):` — weights are swapped in,
then automatically restored on exit.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager

import torch.nn as nn


class EMA:
    """Shadow-weight EMA for a PyTorch model.

    Args:
        model:  The model whose parameters to track.
        decay:  EMA decay factor. 0.999 keeps 99.9% of the old shadow
                weights each step — slow, stable smoothing.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, any] = {}
        self._backup: dict[str, any] = {}

        # Initialise shadow weights as a deep copy of the current weights.
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    # ------------------------------------------------------------------
    # Core update — call once per optimizer step.
    # ------------------------------------------------------------------

    def update(self, model: nn.Module) -> None:
        """Blend current model weights into shadow weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    # ------------------------------------------------------------------
    # Context manager — swap in EMA weights for eval, restore after.
    # ------------------------------------------------------------------

    @contextmanager
    def apply(self, model: nn.Module):
        """Context manager: temporarily replace model weights with EMA shadow.

        Usage::

            with ema.apply(model):
                val_mae = evaluate(model, val_loader)
            # model weights are back to raw training weights here
        """
        # Back up raw training weights.
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        try:
            yield
        finally:
            # Restore raw training weights unconditionally.
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self._backup[name])
            self._backup.clear()

###
# Modified by Francesco Laiti - date 23 February 2024
# Fetched from https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/optim/lr_scheduler.py
###

import torch
from torch.optim.lr_scheduler import _LRScheduler

AVAI_SCHEDS = ["single_step", "multi_step", "cosine"]


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self, optimizer, successor, warmup_epoch, last_epoch=-1, verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self, optimizer, successor, warmup_epoch, cons_lr, last_epoch=-1, verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(optimizer, successor, warmup_epoch, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


class LinearWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self, optimizer, successor, warmup_epoch, min_lr, last_epoch=-1, verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(optimizer, successor, warmup_epoch, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs]


def build_lr_scheduler(
    optimizer,
    lr_scheduler,
    max_epoch,
    warmup_epoch=0,
    warmup_recount=False,
    warmup_type=None,
    warmup_cons_lr=0.01,
    warmup_min_lr=0.001,
    stepsize=None,
    gamma=None,
):
    """
    A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        lr_scheduler (str): Type of learning rate scheduler.
        stepsize (int or list/tuple): Step size for learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
        max_epoch (int): Maximum number of epochs.
        warmup_epoch (int, optional): Number of warmup epochs.
        warmup_recount (bool, optional): Recount option for warmup.
        warmup_type (str, optional): Type of warmup ('constant' or 'linear').
        warmup_cons_lr (float, optional): Learning rate for constant warmup.
        warmup_min_lr (float, optional): Minimum learning rate for linear warmup.
    """

    if lr_scheduler not in AVAI_SCHEDS:
        raise ValueError(
            f"scheduler must be one of {AVAI_SCHEDS}, but got {lr_scheduler}"
        )

    if lr_scheduler == "single_step":
        if isinstance(stepsize, (list, tuple)):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                "For single_step lr_scheduler, stepsize must "
                f"be an integer, but got {type(stepsize)}"
            )

        if stepsize <= 0:
            stepsize = max_epoch

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == "multi_step":
        if not isinstance(stepsize, (list, tuple)):
            raise TypeError(
                "For multi_step lr_scheduler, stepsize must "
                f"be a list, but got {type(stepsize)}"
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=float(max_epoch)
        )

    if warmup_epoch > 0:
        if not warmup_recount:
            scheduler.last_epoch = warmup_epoch

        if warmup_type == "constant":
            scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, warmup_epoch, warmup_cons_lr
            )

        elif warmup_type == "linear":
            scheduler = LinearWarmupScheduler(
                optimizer, scheduler, warmup_epoch, warmup_min_lr
            )

        else:
            raise ValueError

    return scheduler

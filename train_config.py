"""Training Configuration for SimpleBaseline (Xiao et al. 2018).

Replicates the exact training parameters and schedules from mmpose:
configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py
"""

from typing import Dict, Tuple
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, MultiStepLR

# ==========================================
# 1. Dataset & Architecture Specifications
# ==========================================
DATA_CFG: Dict = {
    "num_joints": 17,
    "input_size": (192, 256),       # (W, H)
    "heatmap_size": (48, 64),       # (W, H)
    "sigma": 2.0,
    "padding_factor": 1.25,
    "scale_factor_range": (0.75, 1.25),
    "rotate_factor_deg": 40.0,
    "flip_prob": 0.5,
    "mean": [123.675, 116.28, 103.53],
    "std": [58.395, 57.12, 57.375],
}

# ==========================================
# 2. Optimization & Schedule Configuration
# ==========================================
TRAIN_CFG: Dict = {
    # Optimizer
    "optimizer": "Adam",
    "lr": 5e-4,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,

    # Warmup & Decay
    "warmup_iters": 500,
    "warmup_start_factor": 0.001,
    "milestones": [170, 200],       # Epochs for lr decay by 0.1
    "gamma": 0.1,

    # Training runtime
    "max_epochs": 210,
    "batch_size_per_gpu": 64,
    "default_num_gpus": 8,
    "effective_batch_size": 512,
    "val_interval": 10,
    "checkpoint_interval": 10,
    "best_metric": "coco/AP",
}


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    lr: float = TRAIN_CFG["lr"],
    max_epochs: int = TRAIN_CFG["max_epochs"],
    milestones: Tuple[int, ...] = tuple(TRAIN_CFG["milestones"]),
    gamma: float = TRAIN_CFG["gamma"],
    warmup_epochs: int = 5,
) -> Tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    """Build Adam optimizer with LinearLR warmup and MultiStepLR decay.

    Args:
        model (torch.nn.Module): The SimpleBaseline model instance.
        lr (float): Peak learning rate. Default: 5e-4.
        max_epochs (int): Total training epochs. Default: 210.
        milestones (Tuple[int, ...]): Epoch milestones for 0.1x decay. Default: (170, 200).
        gamma (float): Decay factor. Default: 0.1.
        warmup_epochs (int): Number of warmup epochs. Default: 5.

    Returns:
        Tuple[Optimizer, LRScheduler]: Optimizer and combined learning rate scheduler.
    """
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=TRAIN_CFG["betas"],
        eps=TRAIN_CFG["eps"],
        weight_decay=TRAIN_CFG["weight_decay"],
    )

    # Linear warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=TRAIN_CFG["warmup_start_factor"],
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    # Multi-step decay scheduler
    multistep_milestones = [m - warmup_epochs for m in milestones]
    multistep_scheduler = MultiStepLR(
        optimizer,
        milestones=multistep_milestones,
        gamma=gamma,
    )

    # Chained sequential scheduler
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, multistep_scheduler],
        milestones=[warmup_epochs],
    )

    return optimizer, scheduler


if __name__ == "__main__":
    from model import SimpleBaseline

    model = SimpleBaseline(num_joints=17, pretrained_backbone=False)
    opt, sched = build_optimizer_and_scheduler(model)

    print("Training Configuration Specification:")
    for k, v in TRAIN_CFG.items():
        print(f"  {k:22s}: {v}")

    # Verify learning rate schedule progression
    lrs = []
    for epoch in range(TRAIN_CFG["max_epochs"]):
        lrs.append(sched.get_last_lr()[0])
        sched.step()

    print(f"\nInitial LR (epoch 0):       {lrs[0]:.7f}")
    print(f"Post-warmup LR (epoch 5):   {lrs[5]:.7f}")
    print(f"Milestone 1 LR (epoch 170): {lrs[170]:.7f}")
    print(f"Milestone 2 LR (epoch 200): {lrs[200]:.7f}")
    print(f"Final LR (epoch 209):       {lrs[209]:.7f}")

    assert abs(lrs[5] - 5e-4) < 1e-6, "Post warmup LR should be 5e-4"
    assert abs(lrs[170] - 5e-5) < 1e-7, "Epoch 170 LR should be 5e-5"
    assert abs(lrs[200] - 5e-6) < 1e-8, "Epoch 200 LR should be 5e-6"
    print("\nTraining configuration and LR schedule verified successfully!")

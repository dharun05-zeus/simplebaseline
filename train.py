"""Training Loop for SimpleBaseline (Xiao et al. 2018).

Wires together model.py, coco_dataset.py, pipeline.py, and train_config.py.
Config: configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py
"""

import argparse
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR, _LRScheduler
from torch.utils.data import DataLoader, Dataset

from coco_dataset import CocoKeypointDataset, collate_fn
from model import SimpleBaseline
from msra_heatmap_codec import MSRAHeatmap
from pipeline import build_topdown_pipeline
from train_config import DATA_CFG, TRAIN_CFG


class SyntheticPoseDataset(Dataset):
    """Synthetic dataset for in-memory smoke testing without downloading COCO."""

    def __init__(self, num_samples: int = 8) -> None:
        super().__init__()
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img = torch.randn(3, 256, 192, dtype=torch.float32)
        heatmaps = torch.rand(17, 64, 48, dtype=torch.float32)
        weights = torch.ones(17, dtype=torch.float32)
        return {
            "inputs": img,
            "target_heatmaps": heatmaps,
            "target_weights": weights,
            "img_id": idx,
            "bbox": np.array([50.0, 50.0, 100.0, 150.0], dtype=np.float32),
        }


def build_dataloader(
    ann_file: str,
    img_dir: str,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Build PyTorch DataLoader with COCO dataset and top-down preprocessing pipeline.

    Args:
        ann_file (str): Path to COCO keypoint annotations JSON.
        img_dir (str): Path to image directory.
        batch_size (int): Batch size per step. Default: 4.
        num_workers (int): DataLoader worker subprocesses. Default: 0.
        shuffle (bool): Whether to shuffle dataset. Default: True.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    codec = MSRAHeatmap(
        input_size=DATA_CFG["input_size"],
        heatmap_size=DATA_CFG["heatmap_size"],
        sigma=DATA_CFG["sigma"],
    )
    pipeline = build_topdown_pipeline(
        codec=codec,
        input_size=DATA_CFG["input_size"],
        is_train=True,
    )
    dataset = CocoKeypointDataset(
        ann_file=ann_file,
        img_dir=img_dir,
        pipeline_transforms=pipeline,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )


def build_optimizer_and_scheduler(
    model: nn.Module,
    total_iters_per_epoch: int,
    lr: float = TRAIN_CFG["lr"],
    warmup_iters: int = TRAIN_CFG["warmup_iters"],
    warmup_start_factor: float = TRAIN_CFG["warmup_start_factor"],
    milestones: Sequence[int] = TRAIN_CFG["milestones"],
    gamma: float = TRAIN_CFG["gamma"],
) -> Tuple[Optimizer, _LRScheduler]:
    """Build Adam optimizer with linear iteration warmup and multistep epoch decay.

    Replicates mmpose's learning rate schedule:
    - Linear warmup from lr * warmup_start_factor over warmup_iters
    - MultiStepLR decay by gamma at milestone epochs (converted to iteration units)

    Args:
        model (nn.Module): SimpleBaseline model.
        total_iters_per_epoch (int): Number of iterations per epoch.
        lr (float): Peak learning rate. Default: 5e-4.
        warmup_iters (int): Warmup step count. Default: 500.
        warmup_start_factor (float): Initial warmup multiplier. Default: 0.001.
        milestones (Sequence[int]): Decay milestone epochs. Default: [170, 200].
        gamma (float): Learning rate decay factor. Default: 0.1.

    Returns:
        Tuple[Optimizer, _LRScheduler]: Configured optimizer and chained scheduler.
    """
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=TRAIN_CFG["betas"],
        eps=TRAIN_CFG["eps"],
        weight_decay=TRAIN_CFG["weight_decay"],
    )

    # In case warmup_iters is larger than total training iterations in small runs
    warmup_iters = max(1, min(warmup_iters, total_iters_per_epoch * 10))

    # 1. Warmup scheduler (iteration-based)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_iters,
    )

    # 2. Main multistep scheduler (iteration-based)
    milestones_iters = [
        max(1, m * total_iters_per_epoch - warmup_iters) for m in milestones
    ]
    main_scheduler = MultiStepLR(
        optimizer,
        milestones=milestones_iters,
        gamma=gamma,
    )

    # 3. Sequential chained scheduler
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_iters],
    )

    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    device: torch.device,
    epoch: int,
    log_interval: int = 10,
    max_iters: Optional[int] = None,
) -> float:
    """Execute one training epoch.

    Args:
        model (nn.Module): SimpleBaseline model.
        dataloader (DataLoader): Data loader.
        optimizer (Optimizer): Optimizer.
        scheduler (_LRScheduler): Iteration-based scheduler.
        device (torch.device): Compute device ('cpu' or 'cuda').
        epoch (int): Current epoch number.
        log_interval (int): Print log frequency. Default: 10.
        max_iters (int, optional): Max iterations to run (for quick smoke tests).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_steps = 0

    for step, batch in enumerate(dataloader):
        if max_iters is not None and step >= max_iters:
            break

        inputs = batch["inputs"].to(device)
        target_heatmaps = batch["target_heatmaps"].to(device)
        target_weights = batch["target_weights"].to(device)

        optimizer.zero_grad()
        loss = model.loss(inputs, target_heatmaps, target_weights)
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_loss = loss.item()
        running_loss += current_loss
        total_steps += 1

        if (step + 1) % log_interval == 0 or (step + 1) == len(dataloader):
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch [{epoch:03d}] Step [{step + 1:04d}/{len(dataloader):04d}] "
                f"Loss: {current_loss:.6f} | Avg: {running_loss / total_steps:.6f} | "
                f"LR: {current_lr:.8f}"
            )

    return running_loss / max(1, total_steps)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    epoch: int,
    path: str,
) -> None:
    """Save model and training state checkpoint."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, path)
    print(f"Checkpoint successfully saved to: {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    path: str,
    device: torch.device,
) -> int:
    """Load model and training state checkpoint.

    Returns:
        int: Next epoch to resume from (saved_epoch + 1).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    resumed_epoch = checkpoint["epoch"] + 1
    print(f"Resumed from checkpoint: {path} (Epoch {checkpoint['epoch']})")
    return resumed_epoch


def run_synthetic_smoke_test() -> None:
    """Run an end-to-end synthetic training and checkpointing smoke test without external files."""
    print("Running in-memory synthetic smoke test...")
    device = torch.device("cpu")

    # 1. Instantiate synthetic dataset & dataloader
    dataset = SyntheticPoseDataset(num_samples=8)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # 2. Build model without downloading weights for instant test
    model = SimpleBaseline(num_joints=17, pretrained_backbone=False).to(device)

    # 3. Build optimizer and scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        total_iters_per_epoch=len(dataloader),
        warmup_iters=2,
    )

    # 4. Train for 2 iterations and verify loss properties
    losses = []
    model.train()
    for step, batch in enumerate(dataloader):
        if step >= 2:
            break
        inputs = batch["inputs"].to(device)
        targets = batch["target_heatmaps"].to(device)
        weights = batch["target_weights"].to(device)

        optimizer.zero_grad()
        loss = model.loss(inputs, targets, weights)
        assert torch.isfinite(loss), f"Loss is non-finite: {loss.item()}"
        assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"

        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    print(f"  Step 1 Loss: {losses[0]:.6f}")
    print(f"  Step 2 Loss: {losses[1]:.6f}")

    # 5. Verify checkpoint save & load restoration
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test_ckpt.pth")
        save_checkpoint(model, optimizer, scheduler, epoch=0, path=ckpt_path)

        # Create fresh model & optimizer to test restoration
        new_model = SimpleBaseline(num_joints=17, pretrained_backbone=False).to(device)
        new_opt, new_sched = build_optimizer_and_scheduler(
            model=new_model,
            total_iters_per_epoch=len(dataloader),
            warmup_iters=2,
        )

        resumed_epoch = load_checkpoint(
            new_model, new_opt, new_sched, ckpt_path, device
        )
        assert resumed_epoch == 1, f"Expected resumed epoch 1, got {resumed_epoch}"

        # Verify model parameters match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2), "Restored model weights do not match saved weights!"

        # Verify optimizer state matches
        opt_state_orig = optimizer.state_dict()["state"]
        opt_state_new = new_opt.state_dict()["state"]
        assert len(opt_state_orig) == len(opt_state_new), "Optimizer state lengths differ!"
        for param_id in opt_state_orig:
            for state_key in opt_state_orig[param_id]:
                v1 = opt_state_orig[param_id][state_key]
                v2 = opt_state_new[param_id][state_key]
                if isinstance(v1, torch.Tensor):
                    assert torch.equal(v1, v2), f"Optimizer state tensor {state_key} mismatch!"
                else:
                    assert v1 == v2, f"Optimizer state value {state_key} mismatch!"

    print("Smoke test passed")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SimpleBaseline pose estimation model on COCO."
    )
    parser.add_argument(
        "--ann-file",
        type=str,
        default=None,
        help="Path to COCO person keypoints annotation JSON",
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default=None,
        help="Path to COCO images directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAIN_CFG["max_epochs"],
        help=f"Total training epochs (default: {TRAIN_CFG['max_epochs']})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per step (default: 4)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader subprocess workers (default: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to train on ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to checkpoint file to resume from",
    )
    parser.add_argument(
        "--max-iters-per-epoch",
        type=int,
        default=None,
        help="Optional cap on iterations per epoch for fast smoke testing",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Iteration interval for printing training logs",
    )

    args = parser.parse_args()

    # If no annotations or images provided, execute synthetic smoke test
    if args.ann_file is None or args.img_dir is None:
        print("No --ann-file or --img-dir provided.")
        run_synthetic_smoke_test()
        return

    # Real data training flow
    device = torch.device(args.device)
    print(f"Starting SimpleBaseline training on device: {device}")
    print(f"  Annotation file: {args.ann_file}")
    print(f"  Images dir:      {args.img_dir}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Epochs:          {args.epochs}")

    dataloader = build_dataloader(
        ann_file=args.ann_file,
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    total_iters_per_epoch = len(dataloader)
    print(f"  Dataset samples: {len(dataloader.dataset):,}")
    print(f"  Steps per epoch: {total_iters_per_epoch:,}")

    model = SimpleBaseline(
        num_joints=DATA_CFG["num_joints"],
        pretrained_backbone=True,
        input_size=DATA_CFG["input_size"],
        heatmap_size=DATA_CFG["heatmap_size"],
        sigma=DATA_CFG["sigma"],
    ).to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        total_iters_per_epoch=total_iters_per_epoch,
    )

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            path=args.resume,
            device=device,
        )

    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
            max_iters=args.max_iters_per_epoch,
        )
        print(f"--> Epoch {epoch:03d} Completed | Avg Loss: {avg_loss:.6f}")

        # Save epoch checkpoint
        ckpt_name = f"simplebaseline_res50_epoch_{epoch:03d}.pth"
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        save_checkpoint(model, optimizer, scheduler, epoch, ckpt_path)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

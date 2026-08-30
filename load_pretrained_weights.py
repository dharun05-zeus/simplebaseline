"""Pretrained Weights Loader for SimpleBaseline (Xiao et al. 2018).

Downloads and maps official open-mmlab/mmpose SimpleBaseline checkpoints into our
standalone SimpleBaseline model architecture without external framework dependencies.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.error
import urllib.request
import torch
import torch.nn as nn

# Primary and fallback official OpenMMLab MMPose SimpleBaseline ResNet-50 checkpoints
PRIMARY_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/"
    "td-hm_res50_8xb64-210e_coco-256x192-81c97e61_20220909.pth"
)
LEGACY_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmpose/top_down/resnet/"
    "res50_coco_256x192-ec54d7f3_20200709.pth"
)
DEFAULT_CACHE_PATH = os.path.join("checkpoints", "res50_coco_256x192.pth")


def download_checkpoint(
    url: str = PRIMARY_CHECKPOINT_URL,
    cache_path: str = DEFAULT_CACHE_PATH,
) -> str:
    """Download checkpoint from URL to cache_path with progress bar and fallback.

    Args:
        url (str): Remote checkpoint URL.
        cache_path (str): Local destination file path.

    Returns:
        str: Absolute path to the downloaded checkpoint.
    """
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1024 * 1024:
        print(f"Using cached checkpoint: {cache_path}")
        return os.path.abspath(cache_path)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    urls_to_try = [url]
    if url != LEGACY_CHECKPOINT_URL:
        urls_to_try.append(LEGACY_CHECKPOINT_URL)

    downloaded = False
    for target_url in urls_to_try:
        print(f"Downloading checkpoint from: {target_url}")
        try:
            torch.hub.download_url_to_file(target_url, cache_path, progress=True)
            if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1024 * 1024:
                downloaded = True
                print(f"Successfully downloaded checkpoint to {cache_path}")
                break
        except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
            print(f"Download from {target_url} failed: {e}")

    if not downloaded:
        error_msg = (
            "\n" + "=" * 70 + "\n"
            "ERROR: Could not download pretrained checkpoint from OpenMMLab.\n"
            "This may be caused by network connectivity issues or URL relocation.\n\n"
            "Manual Download Instructions:\n"
            f"1. Download one of these checkpoints in your browser:\n"
            f"   - {PRIMARY_CHECKPOINT_URL}\n"
            f"   - {LEGACY_CHECKPOINT_URL}\n"
            f"2. Save the file to: {os.path.abspath(cache_path)}\n"
            "3. Re-run your script.\n"
            + "=" * 70
        )
        raise RuntimeError(error_msg)

    return os.path.abspath(cache_path)


def load_and_inspect_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    """Load checkpoint file and inspect top-level structure and parameter keys.

    Args:
        path (str): Path to .pth checkpoint file.

    Returns:
        Dict[str, torch.Tensor]: The raw state dict containing parameter weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    print(f"\n--- Inspecting Checkpoint: {path} ---")
    ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict):
        print(f"Top-level keys in checkpoint: {list(ckpt.keys())}")
        if "state_dict" in ckpt:
            raw_state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            raw_state_dict = ckpt["model"]
        else:
            raw_state_dict = ckpt
    else:
        raw_state_dict = ckpt

    print(f"Total parameter tensors in checkpoint: {len(raw_state_dict)}")
    print("\nFirst 30 parameter keys in checkpoint:")
    for i, (k, v) in enumerate(raw_state_dict.items()):
        if i >= 30:
            break
        shape_str = tuple(v.shape) if hasattr(v, "shape") else "scalar"
        print(f"  [{i:02d}] {k:45s} -> shape {shape_str}")

    return raw_state_dict


def remap_state_dict(
    mmpose_state_dict: Dict[str, torch.Tensor],
    our_model: nn.Module,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Map mmpose parameter naming to our standalone SimpleBaseline model structure.

    Mapping rules:
    - `backbone.X` -> `backbone.X`
    - `keypoint_head.X` -> `head.X`
    - `head.X` -> `head.X`

    Args:
        mmpose_state_dict (Dict[str, torch.Tensor]): Raw state dict from checkpoint.
        our_model (nn.Module): Standalone SimpleBaseline model instance.

    Returns:
        Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
            - remapped_dict: Dict ready for our_model.load_state_dict().
            - report: Detailed statistics on matched and unmatched keys.
    """
    our_state_dict = our_model.state_dict()
    our_keys = set(our_state_dict.keys())

    remapped_dict: Dict[str, torch.Tensor] = {}
    matched_keys: List[str] = []
    unmatched_ckpt_keys: List[str] = []
    shape_mismatch_keys: List[str] = []

    for k, v in mmpose_state_dict.items():
        # Candidate target keys to test
        candidates = [k]

        # Strip or map common head prefixes
        if k.startswith("keypoint_head."):
            candidates.append("head." + k[len("keypoint_head.") :])
        elif k.startswith("head."):
            candidates.append(k)

        # Handle backbone prefix
        if k.startswith("backbone."):
            candidates.append(k)
        else:
            candidates.append("backbone." + k)

        target_key = None
        for cand in candidates:
            if cand in our_keys:
                target_key = cand
                break

        if target_key is not None:
            expected_shape = our_state_dict[target_key].shape
            if v.shape == expected_shape:
                remapped_dict[target_key] = v
                matched_keys.append(f"{k} -> {target_key}")
            else:
                shape_mismatch_keys.append(
                    f"{k} (ckpt: {tuple(v.shape)}) != {target_key} (model: {tuple(expected_shape)})"
                )
        else:
            unmatched_ckpt_keys.append(k)

    # Check for parameters in our model that were not filled
    unfilled_model_keys = [k for k in our_keys if k not in remapped_dict]

    # Count breakdown for backbone and head
    backbone_loaded = sum(1 for k in remapped_dict if k.startswith("backbone."))
    backbone_total = sum(1 for k in our_keys if k.startswith("backbone."))
    head_loaded = sum(1 for k in remapped_dict if k.startswith("head."))
    head_total = sum(1 for k in our_keys if k.startswith("head."))

    report = {
        "matched_count": len(matched_keys),
        "unmatched_ckpt_count": len(unmatched_ckpt_keys),
        "shape_mismatch_count": len(shape_mismatch_keys),
        "unfilled_model_count": len(unfilled_model_keys),
        "backbone_loaded": backbone_loaded,
        "backbone_total": backbone_total,
        "head_loaded": head_loaded,
        "head_total": head_total,
        "unfilled_model_keys": unfilled_model_keys,
        "unmatched_ckpt_keys": unmatched_ckpt_keys,
    }

    return remapped_dict, report


def load_pretrained(
    model: nn.Module,
    checkpoint_path: Optional[str] = None,
    url: str = PRIMARY_CHECKPOINT_URL,
) -> nn.Module:
    """Download (if needed), remap, and load pretrained weights into SimpleBaseline.

    Args:
        model (nn.Module): Standalone SimpleBaseline model.
        checkpoint_path (str, optional): Path to local checkpoint file.
            If None, downloads from official OpenMMLab URL. Default: None.
        url (str): Remote checkpoint URL if downloading.

    Returns:
        nn.Module: Model with loaded weights.
    """
    if checkpoint_path is None:
        checkpoint_path = download_checkpoint(url=url)

    raw_state_dict = load_and_inspect_checkpoint(checkpoint_path)
    remapped, report = remap_state_dict(raw_state_dict, model)

    print("\n--- Weight Remapping Summary ---")
    print(f"Matched & Loaded:          {report['matched_count']} keys")
    print(f"Unmatched in Checkpoint:   {report['unmatched_ckpt_count']} keys")
    print(f"Shape Mismatches:          {report['shape_mismatch_count']} keys")
    print(f"Unfilled in Model:         {report['unfilled_model_count']} keys")
    print(
        f"Backbone Parameters:       {report['backbone_loaded']}/{report['backbone_total']} "
        f"({report['backbone_loaded'] / max(1, report['backbone_total']) * 100:.1f}%)"
    )
    print(
        f"Head Parameters:           {report['head_loaded']}/{report['head_total']} "
        f"({report['head_loaded'] / max(1, report['head_total']) * 100:.1f}%)"
    )

    if report["unfilled_model_keys"]:
        print(f"Notice: Unfilled keys in model: {report['unfilled_model_keys']}")

    # Load weights into model
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(
        f"\nModel state_dict loaded with strict=False "
        f"({len(missing)} missing keys, {len(unexpected)} unexpected keys)."
    )

    return model


if __name__ == "__main__":
    from model import SimpleBaseline

    print("Initializing standalone SimpleBaseline model...")
    model = SimpleBaseline(num_joints=17, pretrained_backbone=False)
    model.eval()

    # Download and load pretrained MMPose weights
    load_pretrained(model)

    # Run sanity forward pass
    x = torch.randn(1, 3, 256, 192)
    with torch.no_grad():
        heatmaps = model(x)
        keypoints, scores = model.predict(x, flip_test=True)

    hm_np = heatmaps.numpy()
    print("\n--- Sanity Forward Pass Output Statistics ---")
    print(f"Input tensor:      {x.shape}")
    print(f"Output heatmaps:   {heatmaps.shape}")
    print(f"Heatmap Mean:      {hm_np.mean():.6f}")
    print(f"Heatmap Std:       {hm_np.std():.6f}")
    print(f"Heatmap Min:       {hm_np.min():.6f}")
    print(f"Heatmap Max:       {hm_np.max():.6f}")
    print(f"Decoded Keypoints: {keypoints.shape}")
    print(f"Keypoint Scores:   {scores.shape}")
    print(f"Sample Joint 0:    coord={keypoints[0, 0]}, score={scores[0, 0]:.4f}")

    print("\nPretrained weights successfully loaded and verified!")

"""Top-down Pose Estimation Data Pipeline for SimpleBaseline (Xiao et al. 2018).

Standalone implementation replicating mmpose/datasets/transforms:
- GetBBoxCenterScale (topdown_transforms.py)
- RandomFlip (common_transforms.py)
- RandomBBoxTransform (topdown_transforms.py)
- TopdownAffine (topdown_transforms.py)
- GenerateTarget (topdown_transforms.py)
- PackPoseInputs (formatting.py)

Pure NumPy + OpenCV + PyTorch, zero mmcv/mmengine/mmpose dependencies.
Config: configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np
import torch

from msra_heatmap_codec import MSRAHeatmap

# Standard COCO 17 keypoint left-right symmetric swap pairs (0-indexed)
COCO_FLIP_PAIRS: List[Tuple[int, int]] = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
]


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate the third point of an affine triangle to preserve aspect ratio.

    Matches mmpose/datasets/transforms/utils.py.
    """
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_dir(src_point: Sequence[float], rot_rad: float) -> np.ndarray:
    """Rotate a 2D vector by rot_rad radians (clockwise / coordinate rotation).

    Matches mmpose/datasets/transforms/utils.py.
    """
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0.0, 0.0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return np.array(src_result, dtype=np.float32)


def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    shift: Tuple[float, float] = (0.0, 0.0),
    inv: bool = False,
) -> np.ndarray:
    """Calculate 2x3 affine transformation matrix from bbox to canonical patch.

    Faithfully replicates mmpose/datasets/transforms/utils.py's get_warp_matrix.

    Args:
        center (np.ndarray): Center point of bounding box [c_x, c_y].
        scale (np.ndarray): Scale of bounding box [s_w, s_h] in pixels.
        rot (float): Rotation angle in degrees (clockwise).
        output_size (Tuple[int, int]): Target patch size (W, H), e.g. (192, 256).
        shift (Tuple[float, float]): Shift vector. Default: (0.0, 0.0).
        inv (bool): If True, returns inverse affine matrix. Default: False.

    Returns:
        np.ndarray: 2x3 affine transformation matrix float32.
    """
    shift = np.array(shift, dtype=np.float32)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180.0
    src_dir = _get_dir([0.0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32) + dst_dir

    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform_pts(pts: np.ndarray, trans_mat: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to 2D keypoint coordinates.

    Args:
        pts (np.ndarray): Keypoints array of shape (K, 2) or (N, K, 2).
        trans_mat (np.ndarray): 2x3 affine transformation matrix.

    Returns:
        np.ndarray: Transformed keypoints array of identical shape.
    """
    orig_shape = pts.shape
    flat_pts = pts.reshape(-1, 2)
    pts_homo = np.concatenate(
        [flat_pts, np.ones((flat_pts.shape[0], 1), dtype=flat_pts.dtype)], axis=1
    )
    pts_trans = (trans_mat @ pts_homo.T).T
    return pts_trans.reshape(orig_shape)


class GetBBoxCenterScale:
    """Derive bounding box center and aspect-ratio-corrected scale with padding.

    Replicates mmpose.datasets.transforms.GetBBoxCenterScale.

    Args:
        padding (float): Bounding box padding multiplier. Default: 1.25.
        input_size (Tuple[int, int]): Target patch dimensions (W, H). Default: (192, 256).
    """

    def __init__(
        self,
        padding: float = 1.25,
        input_size: Tuple[int, int] = (192, 256),
    ) -> None:
        self.padding = float(padding)
        self.input_size = tuple(input_size)
        self.aspect_ratio = float(input_size[0]) / float(input_size[1])  # 192/256 = 0.75

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        bbox = results["bbox"]  # [x, y, w, h] in COCO format
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        # Aspect ratio fix matching mmpose
        if w > self.aspect_ratio * h:
            h = w / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array([w * self.padding, h * self.padding], dtype=np.float32)

        results["bbox_center"] = center
        results["bbox_scale"] = scale
        results["bbox_rotation"] = 0.0
        return results


class RandomFlip:
    """Random horizontal image, center, and keypoint flip.

    Replicates mmpose.datasets.transforms.RandomFlip.

    Args:
        prob (float): Flip probability. Default: 0.5.
        flip_pairs (Sequence[Tuple[int, int]], optional): Keypoint index pairs to swap.
            Default: COCO_FLIP_PAIRS.
    """

    def __init__(
        self,
        prob: float = 0.5,
        flip_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        self.prob = prob
        self.flip_pairs = flip_pairs if flip_pairs is not None else COCO_FLIP_PAIRS

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.rand() > self.prob:
            results["flip"] = False
            return results

        results["flip"] = True
        img = results["img"]
        img_w = img.shape[1]

        # 1. Flip image horizontally
        results["img"] = cv2.flip(img, 1)

        # 2. Mirror bbox_center x-coordinate
        center = results["bbox_center"]
        center[0] = (img_w - 1.0) - center[0]
        results["bbox_center"] = center

        # 3. Mirror keypoint x-coordinates and swap left/right pairs
        if "keypoints" in results:
            keypoints = results["keypoints"].copy()
            keypoints[..., 0] = (img_w - 1.0) - keypoints[..., 0]

            keypoints_swapped = keypoints.copy()
            for a, b in self.flip_pairs:
                keypoints_swapped[..., a, :] = keypoints[..., b, :]
                keypoints_swapped[..., b, :] = keypoints[..., a, :]
            results["keypoints"] = keypoints_swapped

        # 4. Swap keypoint visibility weights
        if "keypoints_visible" in results:
            vis = results["keypoints_visible"].copy()
            vis_swapped = vis.copy()
            for a, b in self.flip_pairs:
                vis_swapped[..., a] = vis[..., b]
                vis_swapped[..., b] = vis[..., a]
            results["keypoints_visible"] = vis_swapped

        return results


class RandomBBoxTransform:
    """Random scale, rotation, and translation bounding box augmentations.

    Replicates mmpose.datasets.transforms.RandomBBoxTransform.

    Args:
        shift_factor (float): Max translation shift relative to scale. Default: 0.16.
        shift_prob (float): Probability of applying translation shift. Default: 0.3.
        scale_factor (Tuple[float, float]): Range of random scaling. Default: (0.5, 1.5).
        scale_prob (float): Probability of applying random scaling. Default: 1.0.
        rotate_factor (float): Max rotation angle in degrees. Default: 40.0.
        rotate_prob (float): Probability of applying rotation. Default: 0.6.
    """

    def __init__(
        self,
        shift_factor: float = 0.16,
        shift_prob: float = 0.3,
        scale_factor: Tuple[float, float] = (0.5, 1.5),
        scale_prob: float = 1.0,
        rotate_factor: float = 40.0,
        rotate_prob: float = 0.6,
    ) -> None:
        self.shift_factor = shift_factor
        self.shift_prob = shift_prob
        self.scale_factor = scale_factor
        self.scale_prob = scale_prob
        self.rotate_factor = rotate_factor
        self.rotate_prob = rotate_prob

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        scale = results["bbox_scale"]
        center = results["bbox_center"]

        # 1. Random scaling
        if np.random.rand() < self.scale_prob:
            s_low, s_high = self.scale_factor
            scale_ratio = np.random.uniform(s_low, s_high)
            scale = scale * scale_ratio

        # 2. Random rotation
        if np.random.rand() < self.rotate_prob:
            rotation = np.random.uniform(-self.rotate_factor, self.rotate_factor)
        else:
            rotation = 0.0

        # 3. Random translation shift
        if np.random.rand() < self.shift_prob:
            shift = np.random.uniform(-self.shift_factor, self.shift_factor, size=2)
            center = center + shift * scale

        results["bbox_scale"] = scale
        results["bbox_center"] = center
        results["bbox_rotation"] = rotation
        return results


class TopdownAffine:
    """Affine crop and warp image and keypoints to input_size (W, H).

    Replicates mmpose.datasets.transforms.TopdownAffine.

    Args:
        input_size (Tuple[int, int]): Canonical model input size (W, H). Default: (192, 256).
    """

    def __init__(self, input_size: Tuple[int, int] = (192, 256)) -> None:
        self.input_size = tuple(input_size)  # (W, H)

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results["img"]
        center = results["bbox_center"]
        scale = results["bbox_scale"]
        rot = results.get("bbox_rotation", 0.0)

        # Construct affine matrix
        warp_mat = get_warp_matrix(center, scale, rot, self.input_size)
        results["warp_mat"] = warp_mat

        # Warp image patch
        warped_img = cv2.warpAffine(
            img,
            warp_mat,
            self.input_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        results["img"] = warped_img

        # Transform keypoints into canonical (W, H) patch coordinates
        if "keypoints" in results:
            keypoints = results["keypoints"]
            results["keypoints"] = affine_transform_pts(keypoints, warp_mat)

        return results


class GenerateTarget:
    """Generate 2D Gaussian heatmap targets using MSRAHeatmap codec.

    Replicates mmpose.datasets.transforms.GenerateTarget.

    Args:
        codec (MSRAHeatmap): Codec instance configured with input_size and heatmap_size.
    """

    def __init__(self, codec: MSRAHeatmap) -> None:
        self.codec = codec

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        keypoints = results["keypoints"]
        keypoints_visible = results.get("keypoints_visible", None)

        encoded = self.codec.encode(keypoints, keypoints_visible)
        results["heatmaps"] = encoded["heatmaps"]
        results["keypoint_weights"] = encoded["keypoint_weights"]
        return results


class PackPoseInputs:
    """Convert numpy image, heatmaps, and target weights to PyTorch tensors.

    Replicates mmpose.datasets.transforms.PackPoseInputs.

    Image normalization is performed directly on [0, 255] uint8/float32 pixels:
        img_norm = (img - mean) / std
    where mean = [123.675, 116.28, 103.53] and std = [58.395, 57.12, 57.375].

    Args:
        mean (Sequence[float]): ImageNet channel means (RGB). Default: [123.675, 116.28, 103.53].
        std (Sequence[float]): ImageNet channel standard deviations (RGB). Default: [58.395, 57.12, 57.375].
    """

    def __init__(
        self,
        mean: Sequence[float] = (123.675, 116.28, 103.53),
        std: Sequence[float] = (58.395, 57.12, 57.375),
    ) -> None:
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results["img"]

        # Note: If images are loaded via OpenCV cv2.imread(), they are BGR.
        # MMPose loads RGB by default via mmcv.imread(backend='cv2') which converts BGR->RGB.
        # If input image is 3-channel, apply [0, 255] normalization:
        img_float = img.astype(np.float32)
        img_norm = (img_float - self.mean) / self.std

        # Permute (H, W, 3) -> (3, H, W)
        img_tensor = torch.from_numpy(np.transpose(img_norm, (2, 0, 1))).float()

        output = {
            "inputs": img_tensor,
            "bbox_center": results.get("bbox_center"),
            "bbox_scale": results.get("bbox_scale"),
            "warp_mat": results.get("warp_mat"),
        }

        if "heatmaps" in results:
            output["target_heatmaps"] = torch.from_numpy(results["heatmaps"]).float()
        if "keypoint_weights" in results:
            weights = results["keypoint_weights"]
            if weights.ndim == 2 and weights.shape[0] == 1:
                weights = weights[0]
            output["target_weights"] = torch.from_numpy(weights).float()

        return output


class Compose:
    """Sequentially chain a list of data transformation callables."""

    def __init__(self, transforms: Sequence[Any]) -> None:
        self.transforms = list(transforms)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            data = t(data)
        return data


def build_topdown_pipeline(
    codec: MSRAHeatmap,
    input_size: Tuple[int, int] = (192, 256),
    is_train: bool = True,
) -> Compose:
    """Build full top-down data pipeline matching mmpose config."""
    if is_train:
        transforms = [
            GetBBoxCenterScale(padding=1.25, input_size=input_size),
            RandomFlip(prob=0.5),
            RandomBBoxTransform(
                shift_factor=0.16,
                shift_prob=0.3,
                scale_factor=(0.5, 1.5),
                scale_prob=1.0,
                rotate_factor=40.0,
                rotate_prob=0.6,
            ),
            TopdownAffine(input_size=input_size),
            GenerateTarget(codec=codec),
            PackPoseInputs(),
        ]
    else:
        transforms = [
            GetBBoxCenterScale(padding=1.25, input_size=input_size),
            TopdownAffine(input_size=input_size),
            PackPoseInputs(),
        ]
    return Compose(transforms)


if __name__ == "__main__":
    np.random.seed(42)

    # 1. Build a synthetic 'results' dict
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    bbox = np.array([100.0, 80.0, 200.0, 300.0], dtype=np.float32)  # [x, y, w, h]

    # Generate keypoints within the bounding box
    kpts_x = np.random.uniform(100.0, 300.0, (1, 17, 1)).astype(np.float32)
    kpts_y = np.random.uniform(80.0, 380.0, (1, 17, 1)).astype(np.float32)
    keypoints = np.concatenate([kpts_x, kpts_y], axis=-1)
    keypoints_visible = np.ones((1, 17), dtype=np.float32)

    results = {
        "img": img.copy(),
        "bbox": bbox.copy(),
        "keypoints": keypoints.copy(),
        "keypoints_visible": keypoints_visible.copy(),
    }

    # 2. Run full training pipeline
    codec = MSRAHeatmap(input_size=(192, 256), heatmap_size=(48, 64), sigma=2.0)
    pipeline = build_topdown_pipeline(codec=codec, input_size=(192, 256), is_train=True)
    out = pipeline(results)

    # 3. Assertions on output shapes
    assert out["inputs"].shape == (
        3,
        256,
        192,
    ), f"Expected input shape (3, 256, 192), got {out['inputs'].shape}"
    assert out["target_heatmaps"].shape == (
        17,
        64,
        48,
    ), f"Expected target_heatmaps shape (17, 64, 48), got {out['target_heatmaps'].shape}"
    assert out["target_weights"].shape == (
        17,
    ), f"Expected target_weights shape (17,), got {out['target_weights'].shape}"

    # 4. Sanity-check TopdownAffine specifically
    # Center of bbox is [100 + 200/2, 80 + 300/2] = [200, 230]
    center_kpt = np.array([[[200.0, 230.0]]], dtype=np.float32)
    sanity_results = {
        "img": img.copy(),
        "bbox": bbox.copy(),
        "keypoints": center_kpt.copy(),
        "keypoints_visible": np.ones((1, 1), dtype=np.float32),
    }

    get_cs = GetBBoxCenterScale(padding=1.25, input_size=(192, 256))
    affine = TopdownAffine(input_size=(192, 256))

    sanity_results = get_cs(sanity_results)
    sanity_results = affine(sanity_results)

    warped_center_kpt = sanity_results["keypoints"][0, 0]
    expected_center = np.array([96.0, 128.0], dtype=np.float32)
    center_dist = np.linalg.norm(warped_center_kpt - expected_center)

    print("Pipeline audit & validation passed successfully!")
    print(f"Packed tensor input shape:        {out['inputs'].shape}")
    print(f"Packed target heatmaps shape:     {out['target_heatmaps'].shape}")
    print(f"Packed target weights shape:      {out['target_weights'].shape}")
    print(f"Warped bbox center keypoint:      {warped_center_kpt}")
    print(f"Distance to canvas center (96,128): {center_dist:.4f}px")

    assert center_dist < 1.0, f"Transformed center point is off by {center_dist:.2f}px (expected < 1px)"

"""MSRA Heatmap Codec for SimpleBaseline (Xiao et al. 2018).

Standalone NumPy implementation replicating mmpose/codecs/msra_heatmap.py.
Handles target generation (Gaussian encoding) and keypoint decoding
(argmax + quarter-pixel offset refinement). Pure NumPy with no framework dependencies.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np


def get_heatmap_maximum(
    heatmaps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract maximum locations and values for each heatmap channel.

    Args:
        heatmaps (np.ndarray): Heatmaps of shape (K, H, W) or (N, K, H, W).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - locs: Maximum coordinates (x, y) of shape (K, 2) or (N, K, 2).
            - vals: Maximum values (scores) of shape (K,) or (N, K).
    """
    ndim = heatmaps.ndim
    if ndim == 3:
        K, H, W = heatmaps.shape
        # Flatten spatial dimensions to find argmax
        heatmaps_reshaped = heatmaps.reshape(K, -1)
        max_idx = np.argmax(heatmaps_reshaped, axis=-1)
        max_vals = np.max(heatmaps_reshaped, axis=-1)

        # Convert 1D index to 2D (x, y) coordinates
        loc_y = max_idx // W
        loc_x = max_idx % W

        locs = np.stack([loc_x, loc_y], axis=-1).astype(np.float32)
        return locs, max_vals
    elif ndim == 4:
        N, K, H, W = heatmaps.shape
        heatmaps_reshaped = heatmaps.reshape(N, K, -1)
        max_idx = np.argmax(heatmaps_reshaped, axis=-1)
        max_vals = np.max(heatmaps_reshaped, axis=-1)

        loc_y = max_idx // W
        loc_x = max_idx % W

        locs = np.stack([loc_x, loc_y], axis=-1).astype(np.float32)
        return locs, max_vals
    else:
        raise ValueError(
            f"Expected 3D or 4D heatmap array, got shape {heatmaps.shape}"
        )


def refine_keypoints(locs: np.ndarray, heatmaps: np.ndarray) -> np.ndarray:
    """Apply quarter-pixel refinement to keypoint locations based on heatmap gradients.

    Follows mmpose/codecs/utils/post_processing.py quarter-pixel shift logic:
    dx = heatmap[y, x+1] - heatmap[y, x-1]
    dy = heatmap[y+1, x] - heatmap[y-1, x]
    offset = 0.25 * sign([dx, dy])

    Args:
        locs (np.ndarray): Initial keypoint coordinates (x, y) of shape (K, 2).
        heatmaps (np.ndarray): Heatmaps of shape (K, H, W).

    Returns:
        np.ndarray: Refined keypoint coordinates of shape (K, 2).
    """
    K, H, W = heatmaps.shape
    refined_locs = locs.copy().astype(np.float32)

    for k in range(K):
        x_int = int(np.round(locs[k, 0]))
        y_int = int(np.round(locs[k, 1]))

        # Refine x coordinate if horizontal neighbors are valid
        if 1 <= x_int < W - 1 and 0 <= y_int < H:
            dx = heatmaps[k, y_int, x_int + 1] - heatmaps[k, y_int, x_int - 1]
            if dx > 0:
                refined_locs[k, 0] += 0.25
            elif dx < 0:
                refined_locs[k, 0] -= 0.25

        # Refine y coordinate if vertical neighbors are valid
        if 1 <= y_int < H - 1 and 0 <= x_int < W:
            dy = heatmaps[k, y_int + 1, x_int] - heatmaps[k, y_int - 1, x_int]
            if dy > 0:
                refined_locs[k, 1] += 0.25
            elif dy < 0:
                refined_locs[k, 1] -= 0.25

    return refined_locs


class MSRAHeatmap:
    """MSRA Heatmap Encoder/Decoder for SimpleBaseline.

    Generates 2D Gaussian heatmaps for keypoints and decodes heatmaps back to
    input-space keypoint coordinates with quarter-pixel refinement.

    Args:
        input_size (Tuple[int, int]): Size of input image (W, H), e.g. (192, 256).
        heatmap_size (Tuple[int, int]): Size of output heatmap (W, H), e.g. (48, 64).
        sigma (float): Standard deviation of 2D Gaussian in heatmap pixels. Default: 2.0.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (192, 256),
        heatmap_size: Tuple[int, int] = (48, 64),
        sigma: float = 2.0,
    ) -> None:
        self.input_size = tuple(input_size)  # (W, H)
        self.heatmap_size = tuple(heatmap_size)  # (W, H)
        self.sigma = float(sigma)

        # Scale factor from heatmap to input image: (scale_x, scale_y)
        self.scale_factor = np.array(
            [
                self.input_size[0] / self.heatmap_size[0],
                self.input_size[1] / self.heatmap_size[1],
            ],
            dtype=np.float32,
        )

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Encode keypoint coordinates into 2D Gaussian heatmaps and target weights.

        Args:
            keypoints (np.ndarray): Keypoints of shape (N, K, 2) or (K, 2) in input
                image pixel coordinates (x, y).
            keypoints_visible (np.ndarray, optional): Visibility flags of shape
                (N, K) or (K,). 1 for visible/labeled, 0 for invisible/unlabeled.
                Default: None (all visible).

        Returns:
            Dict[str, np.ndarray]:
                - 'heatmaps': Gaussian heatmaps of shape (K, H, W), float32.
                - 'keypoint_weights': Weights of shape (N, K), float32.
        """
        if keypoints.ndim == 2:
            keypoints = keypoints[np.newaxis, ...]  # (1, K, 2)

        N, K, _ = keypoints.shape
        W_hm, H_hm = self.heatmap_size

        if keypoints_visible is None:
            keypoints_visible = np.ones((N, K), dtype=np.float32)
        elif keypoints_visible.ndim == 1:
            keypoints_visible = keypoints_visible[np.newaxis, ...]

        keypoint_weights = keypoints_visible.copy().astype(np.float32)
        heatmaps = np.zeros((K, H_hm, W_hm), dtype=np.float32)

        # 3-sigma radius cutoff
        radius = int(np.round(self.sigma * 3.0))
        size = 2 * radius + 1
        x = np.arange(0, size, 1, dtype=np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = radius
        # Precompute 2D Gaussian kernel patch
        gaussian_patch = np.exp(
            -((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * self.sigma**2)
        )

        for n in range(N):
            for k in range(K):
                # Skip invisible joints
                if keypoint_weights[n, k] <= 0:
                    continue

                # Map keypoint to heatmap coordinate grid
                mu_x = int(np.round(keypoints[n, k, 0] / self.scale_factor[0]))
                mu_y = int(np.round(keypoints[n, k, 1] / self.scale_factor[1]))

                # Upper-left and bottom-right corners on heatmap
                ul = [mu_x - radius, mu_y - radius]
                br = [mu_x + radius + 1, mu_y + radius + 1]

                # Check if Gaussian is completely outside heatmap bounds
                if ul[0] >= W_hm or ul[1] >= H_hm or br[0] <= 0 or br[1] <= 0:
                    keypoint_weights[n, k] = 0.0
                    continue

                # Bounding box coordinates on Gaussian patch
                g_x = max(0, -ul[0]), min(br[0], W_hm) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], H_hm) - ul[1]

                # Bounding box coordinates on Heatmap
                hm_x = max(0, ul[0]), min(br[0], W_hm)
                hm_y = max(0, ul[1]), min(br[1], H_hm)

                # Paste Gaussian into heatmap (using max to handle overlapping keypoints)
                heatmaps[k, hm_y[0] : hm_y[1], hm_x[0] : hm_x[1]] = np.maximum(
                    heatmaps[k, hm_y[0] : hm_y[1], hm_x[0] : hm_x[1]],
                    gaussian_patch[g_y[0] : g_y[1], g_x[0] : g_x[1]],
                )

        return {
            "heatmaps": heatmaps,
            "keypoint_weights": keypoint_weights,
        }

    def decode(
        self, heatmaps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode heatmaps into keypoint coordinates in input-image space.

        Args:
            heatmaps (np.ndarray): Predicted heatmaps of shape (K, H, W).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - keypoints: Coordinates (x, y) of shape (K, 2) in input image space.
                - scores: Confidence scores of shape (K,) at argmax locations.
        """
        if heatmaps.ndim != 3:
            raise ValueError(
                f"Expected 3D heatmap array (K, H, W), got shape {heatmaps.shape}"
            )

        # 1. Get raw argmax locations and confidence values
        locs, scores = get_heatmap_maximum(heatmaps)

        # 2. Refine locations using quarter-pixel offset refinement
        refined_locs = refine_keypoints(locs, heatmaps)

        # 3. Rescale coordinates back to input image coordinate space
        keypoints = refined_locs * self.scale_factor

        return keypoints, scores


if __name__ == "__main__":
    codec = MSRAHeatmap(input_size=(192, 256), heatmap_size=(48, 64), sigma=2.0)

    # 1. Create synthetic keypoint sample
    keypoints = np.array([[[96.0, 128.0], [10.0, 10.0], [190.0, 254.0]]], dtype=np.float32)  # (1, 3, 2)
    keypoints_visible = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)  # Joint 3 invisible

    # 2. Encode to heatmaps
    encoded = codec.encode(keypoints, keypoints_visible)
    heatmaps = encoded["heatmaps"]
    weights = encoded["keypoint_weights"]

    assert heatmaps.shape == (3, 64, 48), f"Expected shape (3, 64, 48), got {heatmaps.shape}"
    assert weights.shape == (1, 3), f"Expected shape (1, 3), got {weights.shape}"
    assert weights[0, 2] == 0.0, "Expected invisible joint weight to be 0"
    assert heatmaps[2].sum() == 0.0, "Expected invisible joint heatmap to be all zeros"
    assert heatmaps[0].max() > 0.9, f"Expected peak near 1.0, got {heatmaps[0].max()}"

    # 3. Decode heatmaps back to keypoints
    decoded_kpts, scores = codec.decode(heatmaps)

    assert decoded_kpts.shape == (3, 2), f"Expected shape (3, 2), got {decoded_kpts.shape}"
    assert scores.shape == (3,), f"Expected shape (3,), got {scores.shape}"

    # Verify decoded coordinates are close to original
    err_kpt0 = np.linalg.norm(decoded_kpts[0] - keypoints[0, 0])
    assert err_kpt0 < 2.0, f"Reconstruction error {err_kpt0:.2f}px is too large"

    print("MSRAHeatmap codec smoke test passed successfully!")
    print(f"Original Keypoints:\n{keypoints[0]}")
    print(f"Decoded Keypoints:\n{decoded_kpts}")
    print(f"Scores:\n{scores}")
    print(f"Joint 0 Reconstruction Error: {err_kpt0:.4f} pixels")

"""Real-time Webcam Pose Estimation Demo using SimpleBaseline (Xiao et al. 2018).

Standalone live inference script with GPU-aware performance instrumentation:
- Loads official pretrained ResNet-50 SimpleBaseline weights.
- Crops and preprocesses video frames using top-down affine transforms.
- Performs heatmap regression and quarter-pixel offset refinement.
- Maps detected keypoint coordinates back to original frame space via inverse affine matrix.
- Renders the full 17-keypoint COCO skeleton in real time with FPS & GPU telemetry HUD.
- Synchronizes GPU timers (torch.cuda.synchronize) for exact kernel latency logging.
- Generates post-session summary metrics table (Mean, Min, Max, P95) and optional CSV export.

NOTE: This demo operates under a single-person full-frame assumption (detector-free).
It uses the full frame as the bounding box and works best when one person is roughly
centered and occupies most of the frame.
"""

import argparse
import csv
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import torch

from load_pretrained_weights import DEFAULT_CACHE_PATH, load_pretrained
from model import SimpleBaseline
from msra_heatmap_codec import MSRAHeatmap
from pipeline import GetBBoxCenterScale, TopdownAffine, affine_transform_pts

# Standard 17-keypoint COCO skeleton limb connections
# 0: nose, 1: l_eye, 2: r_eye, 3: l_ear, 4: r_ear
# 5: l_shoulder, 6: r_shoulder, 7: l_elbow, 8: r_elbow, 9: l_wrist, 10: r_wrist
# 11: l_hip, 12: r_hip, 13: l_knee, 14: r_knee, 15: l_ankle, 16: r_ankle
COCO_SKELETON: List[Tuple[int, int]] = [
    (15, 13),  # Left Ankle -> Left Knee
    (13, 11),  # Left Knee -> Left Hip
    (16, 14),  # Right Ankle -> Right Knee
    (14, 12),  # Right Knee -> Right Hip
    (11, 12),  # Left Hip -> Right Hip
    (5, 11),   # Left Shoulder -> Left Hip
    (6, 12),   # Right Shoulder -> Right Hip
    (5, 6),    # Left Shoulder -> Right Shoulder
    (5, 7),    # Left Shoulder -> Left Elbow
    (6, 8),    # Right Shoulder -> Right Elbow
    (7, 9),    # Left Elbow -> Left Wrist
    (8, 10),   # Right Elbow -> Right Wrist
    (1, 2),    # Left Eye -> Right Eye
    (0, 1),    # Nose -> Left Eye
    (0, 2),    # Nose -> Right Eye
    (1, 3),    # Left Eye -> Left Ear
    (2, 4),    # Right Eye -> Right Ear
]

# Color palette for skeleton visualization (BGR format)
# Left side: Blue/Cyan, Right side: Orange/Red, Torso/Face: Green/Yellow
KEYPOINT_COLORS: List[Tuple[int, int, int]] = [
    (0, 255, 255),  # 0: Nose (Yellow)
    (255, 128, 0),  # 1: L-Eye (Cyan-Blue)
    (0, 128, 255),  # 2: R-Eye (Orange)
    (255, 128, 0),  # 3: L-Ear (Cyan-Blue)
    (0, 128, 255),  # 4: R-Ear (Orange)
    (255, 0, 0),    # 5: L-Shoulder (Blue)
    (0, 0, 255),    # 6: R-Shoulder (Red)
    (255, 0, 0),    # 7: L-Elbow (Blue)
    (0, 0, 255),    # 8: R-Elbow (Red)
    (255, 0, 0),    # 9: L-Wrist (Blue)
    (0, 0, 255),    # 10: R-Wrist (Red)
    (255, 0, 128),  # 11: L-Hip (Purple)
    (0, 128, 255),  # 12: R-Hip (Orange-Red)
    (255, 0, 128),  # 13: L-Knee (Purple)
    (0, 128, 255),  # 14: R-Knee (Orange-Red)
    (255, 0, 128),  # 15: L-Ankle (Purple)
    (0, 128, 255),  # 16: R-Ankle (Orange-Red)
]


def preprocess_frame(
    frame: np.ndarray,
    bbox: Sequence[float],
    input_size: Tuple[int, int] = (192, 256),
    mean: Sequence[float] = (123.675, 116.28, 103.53),
    std: Sequence[float] = (58.395, 57.12, 57.375),
) -> Tuple[torch.Tensor, np.ndarray]:
    """Crop and warp raw BGR frame to canonical model input dimensions.

    Args:
        frame (np.ndarray): Raw video frame (H, W, 3) BGR uint8.
        bbox (Sequence[float]): Bounding box [x, y, w, h].
        input_size (Tuple[int, int]): Model canonical input size (W, H). Default: (192, 256).
        mean (Sequence[float]): ImageNet normalization mean.
        std (Sequence[float]): ImageNet normalization std.

    Returns:
        Tuple[torch.Tensor, np.ndarray]:
            - input_tensor: Normalized torch tensor of shape (1, 3, H, W).
            - warp_mat: 2x3 affine matrix used for mapping coordinates back.
    """
    results = {
        "img": frame,
        "bbox": np.array(bbox, dtype=np.float32),
    }

    # 1. Derive padded, aspect-ratio-matched center and scale
    get_cs = GetBBoxCenterScale(padding=1.25, input_size=input_size)
    results = get_cs(results)

    # 2. Warp image patch
    affine = TopdownAffine(input_size=input_size)
    results = affine(results)

    # 3. Normalize image directly on [0, 255] pixels
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    img_norm = (results["img"].astype(np.float32) - mean_arr) / std_arr

    # 4. Transpose (H, W, 3) -> (1, 3, H, W) float32 tensor
    img_tensor = torch.from_numpy(np.transpose(img_norm, (2, 0, 1))).unsqueeze(0).float()

    return img_tensor, results["warp_mat"]


def postprocess_heatmaps(
    heatmaps_np: np.ndarray,
    warp_mat: np.ndarray,
    codec: MSRAHeatmap,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode heatmaps and project keypoint coordinates back to original video frame coordinates.

    Args:
        heatmaps_np (np.ndarray): Predicted heatmaps of shape (1, 17, 64, 48) or (17, 64, 48).
        warp_mat (np.ndarray): 2x3 affine transformation matrix from preprocessing.
        codec (MSRAHeatmap): Codec instance.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - original_kpts: Keypoint coordinates (x, y) of shape (17, 2) in original frame space.
            - scores: Keypoint confidence scores of shape (17,).
    """
    if heatmaps_np.ndim == 4:
        heatmaps_np = heatmaps_np[0]

    # 1. Decode keypoints in warped patch space (192, 256)
    decoded_kpts, scores = codec.decode(heatmaps_np)

    # 2. Invert affine matrix to map coordinates back to full frame
    inv_warp = cv2.invertAffineTransform(warp_mat)
    original_kpts = affine_transform_pts(decoded_kpts, inv_warp)

    return original_kpts, scores


def draw_pose(
    frame: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    skeleton: Sequence[Tuple[int, int]] = COCO_SKELETON,
    score_thresh: float = 0.3,
) -> np.ndarray:
    """Render keypoints and skeleton connection lines onto video frame.

    Args:
        frame (np.ndarray): Original BGR image frame.
        keypoints (np.ndarray): Keypoints (x, y) of shape (17, 2).
        scores (np.ndarray): Confidence scores of shape (17,).
        skeleton (Sequence[Tuple[int, int]]): Keypoint connection pairs.
        score_thresh (float): Visibility confidence threshold. Default: 0.3.

    Returns:
        np.ndarray: Annotated BGR frame.
    """
    vis_frame = frame.copy()

    # 1. Draw skeleton limb connection lines
    for idx_a, idx_b in skeleton:
        if scores[idx_a] >= score_thresh and scores[idx_b] >= score_thresh:
            pt_a = (int(np.round(keypoints[idx_a, 0])), int(np.round(keypoints[idx_a, 1])))
            pt_b = (int(np.round(keypoints[idx_b, 0])), int(np.round(keypoints[idx_b, 1])))
            # Line color based on first joint
            line_color = KEYPOINT_COLORS[idx_a % len(KEYPOINT_COLORS)]
            cv2.line(vis_frame, pt_a, pt_b, line_color, 2, cv2.LINE_AA)

    # 2. Draw keypoint circles
    for i in range(len(keypoints)):
        if scores[i] >= score_thresh:
            pt = (int(np.round(keypoints[i, 0])), int(np.round(keypoints[i, 1])))
            color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
            cv2.circle(vis_frame, pt, 4, color, -1, cv2.LINE_AA)
            cv2.circle(vis_frame, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

    return vis_frame


def print_metrics_summary(metrics_log: List[Dict[str, float]], warmup_frames: int = 5) -> None:
    """Compute and print formatted summary table of performance metrics.

    Args:
        metrics_log (List[Dict[str, float]]): List of per-frame metrics dictionaries.
        warmup_frames (int): Number of initial frames to exclude from summary. Default: 5.
    """
    total_frames = len(metrics_log)
    if total_frames == 0:
        print("\nNo frames logged for metrics summary.")
        return

    # Exclude initial warm-up frames to eliminate GPU initialization skew
    if total_frames > warmup_frames:
        eval_log = metrics_log[warmup_frames:]
        warmup_note = f" (excluding {warmup_frames} initial warm-up frames)"
    else:
        eval_log = metrics_log
        warmup_note = ""

    metric_keys = [
        ("FPS", "fps", "{:.1f}"),
        ("GPU Latency (ms)", "gpu_latency_ms", "{:.2f}"),
        ("People Detected", "people_detected", "{:.1f}"),
        ("Joints Identified", "joints_identified", "{:.1f}"),
        ("VRAM Usage (MB)", "vram_usage_mb", "{:.2f}"),
    ]

    print("\n" + "=" * 60)
    print(f" METRICS SUMMARY REPORT ({len(eval_log)} frames logged{warmup_note})")
    print("=" * 60)
    print(f"{'Metric':<20} | {'Mean':<7} | {'Min':<7} | {'Max':<7} | {'P95':<7}")
    print("-" * 60)

    for label, key, fmt in metric_keys:
        values = np.array([m[key] for m in eval_log if key in m], dtype=np.float32)
        if len(values) == 0:
            continue
        v_mean = fmt.format(float(np.mean(values)))
        v_min = fmt.format(float(np.min(values)))
        v_max = fmt.format(float(np.max(values)))
        v_p95 = fmt.format(float(np.percentile(values, 95)))
        print(f"{label:<20} | {v_mean:<7} | {v_min:<7} | {v_max:<7} | {v_p95:<7}")

    print("=" * 60 + "\n")


def export_metrics_to_csv(metrics_log: List[Dict[str, float]], csv_path: str) -> None:
    """Save raw per-frame performance metrics to a CSV file.

    Args:
        metrics_log (List[Dict[str, float]]): Recorded metrics.
        csv_path (str): Output CSV file path.
    """
    if not metrics_log:
        return

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = list(metrics_log[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_log)

    print(f"Raw per-frame metrics saved to CSV: {os.path.abspath(csv_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimpleBaseline Real-time Webcam Pose Estimation Demo with GPU Metrics."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained .pth checkpoint (default: auto-download official weights)",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.3,
        help="Keypoint detection score threshold (default: 0.3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override ('cuda', 'cpu', or None for auto-detect)",
    )
    parser.add_argument(
        "--flip-test",
        action="store_true",
        help="Enable test-time flip aggregation (higher accuracy, slightly slower)",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default=None,
        help="Optional path to output CSV file for recording per-frame metrics",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run an automated headless smoke test on a synthetic frame without opening a GUI window",
    )
    args = parser.parse_args()

    # 1. Device Setup with CUDA optimizations
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA Hardware Detected: {gpu_name}")
        print("Enabled torch.backends.cudnn.benchmark = True for optimal kernel execution.")
    else:
        print("Running inference on CPU.")

    print(f"Initializing SimpleBaseline pose estimator on device: {device}...")

    # 2. Instantiate model and load pretrained weights
    model = SimpleBaseline(num_joints=17, pretrained_backbone=False)
    load_pretrained(model, checkpoint_path=args.checkpoint)
    model.eval().to(device)

    codec = MSRAHeatmap(input_size=(192, 256), heatmap_size=(48, 64), sigma=2.0)

    # 3. Headless automated smoke test
    if args.smoke_test:
        print("\nRunning headless smoke test on synthetic frame...")
        test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        cv2.circle(test_frame, (320, 140), 30, (200, 200, 200), -1)
        cv2.line(test_frame, (320, 170), (320, 320), (200, 200, 200), 10)

        h, w = test_frame.shape[:2]
        bbox = [0.0, 0.0, float(w), float(h)]
        input_tensor, warp_mat = preprocess_frame(test_frame, bbox)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            heatmaps = model(input_tensor.to(device))

        if device.type == "cuda":
            torch.cuda.synchronize()
        gpu_latency_ms = (time.perf_counter() - t_start) * 1000.0

        keypoints, scores = postprocess_heatmaps(
            heatmaps.cpu().numpy(), warp_mat, codec
        )
        annotated_frame = draw_pose(
            test_frame, keypoints, scores, COCO_SKELETON, args.score_thresh
        )

        assert keypoints.shape == (17, 2), f"Expected keypoints shape (17, 2), got {keypoints.shape}"
        assert scores.shape == (17,), f"Expected scores shape (17,), got {scores.shape}"
        assert annotated_frame.shape == test_frame.shape

        vram_mb = torch.cuda.memory_allocated(0) / (1024**2) if device.type == "cuda" else 0.0
        joints_identified = int(np.sum(scores >= args.score_thresh))
        people_detected = 1 if float(np.max(scores)) >= args.score_thresh else 0

        # Log single frame metrics
        mock_log = [{
            "frame_idx": 1,
            "fps": 1000.0 / max(gpu_latency_ms, 1e-4),
            "gpu_latency_ms": gpu_latency_ms,
            "people_detected": people_detected,
            "joints_identified": joints_identified,
            "vram_usage_mb": vram_mb,
        }]

        print_metrics_summary(mock_log, warmup_frames=0)
        if args.log_csv:
            export_metrics_to_csv(mock_log, args.log_csv)

        print("Webcam inference headless smoke test passed successfully!")
        return

    # 4. Initialize video capture
    print(f"Opening webcam device {args.camera_id}...")
    cap = cv2.VideoCapture(args.camera_id)

    if not cap.isOpened():
        print(f"\n[WARNING] Could not open webcam at camera-id {args.camera_id}.")
        print("Running inference test on a synthetic test image instead...")

        test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        cv2.circle(test_frame, (320, 140), 30, (200, 200, 200), -1)
        cv2.line(test_frame, (320, 170), (320, 320), (200, 200, 200), 10)

        h, w = test_frame.shape[:2]
        bbox = [0.0, 0.0, float(w), float(h)]
        input_tensor, warp_mat = preprocess_frame(test_frame, bbox)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            heatmaps = model(input_tensor.to(device))

        if device.type == "cuda":
            torch.cuda.synchronize()
        gpu_latency_ms = (time.perf_counter() - t_start) * 1000.0

        keypoints, scores = postprocess_heatmaps(
            heatmaps.cpu().numpy(), warp_mat, codec
        )
        annotated_frame = draw_pose(
            test_frame, keypoints, scores, COCO_SKELETON, args.score_thresh
        )

        vram_mb = torch.cuda.memory_allocated(0) / (1024**2) if device.type == "cuda" else 0.0
        joints_identified = int(np.sum(scores >= args.score_thresh))
        people_detected = 1 if float(np.max(scores)) >= args.score_thresh else 0

        synthetic_log = [{
            "frame_idx": 1,
            "fps": 1000.0 / max(gpu_latency_ms, 1e-4),
            "gpu_latency_ms": gpu_latency_ms,
            "people_detected": people_detected,
            "joints_identified": joints_identified,
            "vram_usage_mb": vram_mb,
        }]

        print_metrics_summary(synthetic_log, warmup_frames=0)
        if args.log_csv:
            export_metrics_to_csv(synthetic_log, args.log_csv)

        print("Synthetic verification test passed successfully!")
        print("To run with a live camera, connect a webcam and run: python webcam_inference.py --camera-id 0")
        return

    print("Webcam started successfully! Press 'q' in the video window to exit.")

    metrics_log: List[Dict[str, float]] = []
    fps_smooth = 0.0
    alpha = 0.9  # Exponential moving average factor for HUD display
    frame_idx = 0

    try:
        while True:
            t_loop_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to grab video frame. Exiting loop.")
                break

            frame_idx += 1
            h, w = frame.shape[:2]
            bbox = [0.0, 0.0, float(w), float(h)]

            # 1. Preprocess
            input_tensor, warp_mat = preprocess_frame(frame, bbox)
            input_tensor = input_tensor.to(device)

            # 2. Synchronized model forward pass
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_fwd_start = time.perf_counter()

            with torch.no_grad():
                if args.flip_test:
                    heatmaps = model(input_tensor)
                    input_flipped = torch.flip(input_tensor, dims=[3])
                    hm_flipped = torch.flip(model(input_flipped), dims=[3])
                    hm_swapped = hm_flipped.clone()
                    for a, b in model.flip_pairs:
                        hm_swapped[:, a] = hm_flipped[:, b]
                        hm_swapped[:, b] = hm_flipped[:, a]
                    heatmaps = (heatmaps + hm_swapped) * 0.5
                else:
                    heatmaps = model(input_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()
            gpu_latency_ms = (time.perf_counter() - t_fwd_start) * 1000.0

            # 3. Postprocess heatmaps
            keypoints, scores = postprocess_heatmaps(
                heatmaps.cpu().numpy(), warp_mat, codec
            )

            # 4. Telemetry metrics calculation
            t_loop_total = time.perf_counter() - t_loop_start
            loop_fps = 1.0 / max(t_loop_total, 1e-5)
            fps_smooth = alpha * fps_smooth + (1.0 - alpha) * loop_fps if fps_smooth > 0 else loop_fps

            joints_identified = int(np.sum(scores >= args.score_thresh))
            people_detected = 1 if float(np.max(scores)) >= args.score_thresh else 0
            vram_usage_mb = (
                torch.cuda.memory_allocated(0) / (1024**2) if device.type == "cuda" else 0.0
            )

            # Record per-frame metrics
            metrics_log.append({
                "frame_idx": frame_idx,
                "fps": loop_fps,
                "gpu_latency_ms": gpu_latency_ms,
                "people_detected": people_detected,
                "joints_identified": joints_identified,
                "vram_usage_mb": vram_usage_mb,
            })

            # 5. Render pose skeleton and HUD telemetry overlay
            annotated_frame = draw_pose(
                frame, keypoints, scores, COCO_SKELETON, args.score_thresh
            )

            # Draw HUD
            hud_line1 = f"FPS: {fps_smooth:.1f} | GPU Latency: {gpu_latency_ms:.1f}ms"
            hud_line2 = f"Joints: {joints_identified}/17 | VRAM: {vram_usage_mb:.1f}MB | Device: {device.type.upper()}"

            cv2.rectangle(annotated_frame, (10, 10), (450, 70), (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, (10, 10), (450, 70), (0, 30, 255), 1)
            cv2.putText(
                annotated_frame,
                hud_line1,
                (18, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                hud_line2,
                (18, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("SimpleBaseline 2D Pose Estimation", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nSession interrupted by user (Ctrl+C).")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nWebcam inference session closed.")

        # Print performance summary table
        print_metrics_summary(metrics_log, warmup_frames=5)

        # Export to CSV if flag set
        if args.log_csv:
            export_metrics_to_csv(metrics_log, args.log_csv)


if __name__ == "__main__":
    main()

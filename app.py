"""Flask Web Application for SimpleBaseline 2D Pose Estimation.

Features:
- Live Webcam Pose Estimation (WebRTC / MediaDevices + API inference)
- Image / Video upload mode
- Red & Black cyber-styled UI matching reference layout
- Real-time telemetry (FPS, Latency, Keypoint count, Device info)
"""

import base64
import io
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

from load_pretrained_weights import DEFAULT_CACHE_PATH, load_pretrained
from model import SimpleBaseline
from msra_heatmap_codec import MSRAHeatmap
from pipeline import GetBBoxCenterScale, TopdownAffine, affine_transform_pts
from webcam_inference import (
    COCO_SKELETON,
    KEYPOINT_COLORS,
    draw_pose,
    postprocess_heatmaps,
    preprocess_frame,
)

app = Flask(__name__)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize SimpleBaseline model
print(f"Loading SimpleBaseline pose estimation model on device: {DEVICE}...")
MODEL = SimpleBaseline(num_joints=17, pretrained_backbone=False)
load_pretrained(MODEL, checkpoint_path=DEFAULT_CACHE_PATH if os.path.exists(DEFAULT_CACHE_PATH) else None)
MODEL.eval().to(DEVICE)

# Codec for 192x256 input and 48x64 heatmap
CODEC = MSRAHeatmap(input_size=(192, 256), heatmap_size=(48, 64), sigma=2.0)

# Red & Black theme custom skeleton colors (Red, Crimson, Ruby, White accents)
RED_THEME_KEYPOINT_COLORS = [
    (0, 0, 255),      # 0: Nose (Vibrant Red)
    (80, 80, 255),    # 1: L-Eye (Light Coral Red)
    (50, 50, 255),    # 2: R-Eye (Bright Red)
    (80, 80, 255),    # 3: L-Ear (Light Coral Red)
    (50, 50, 255),    # 4: R-Ear (Bright Red)
    (30, 30, 255),    # 5: L-Shoulder (Crimson)
    (0, 0, 200),      # 6: R-Shoulder (Dark Ruby)
    (50, 50, 255),    # 7: L-Elbow (Neon Red)
    (0, 0, 220),      # 8: R-Elbow (Deep Red)
    (100, 100, 255),  # 9: L-Wrist (Pinkish Red)
    (0, 0, 255),      # 10: R-Wrist (Vibrant Red)
    (30, 30, 255),    # 11: L-Hip (Crimson)
    (0, 0, 200),      # 12: R-Hip (Dark Ruby)
    (50, 50, 255),    # 13: L-Knee (Neon Red)
    (0, 0, 220),      # 14: R-Knee (Deep Red)
    (100, 100, 255),  # 15: L-Ankle (Pinkish Red)
    (0, 0, 255),      # 16: R-Ankle (Vibrant Red)
]


def run_inference_on_frame(
    frame_bgr: np.ndarray,
    score_thresh: float = 0.3,
    flip_test: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Preprocess, predict pose keypoints, and compute metrics.

    Args:
        frame_bgr (np.ndarray): Input image in BGR format.
        score_thresh (float): Visibility confidence threshold.
        flip_test (bool): Whether to run test-time flip aggregation.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, float]:
            - keypoints: Keypoint coordinates in original frame space.
            - scores: Confidence score for each keypoint.
            - inference_ms: Model inference latency in milliseconds.
            - total_ms: Total processing latency in milliseconds.
    """
    t_start = time.perf_counter()
    h, w = frame_bgr.shape[:2]
    bbox = [0.0, 0.0, float(w), float(h)]

    # Preprocessing
    input_tensor, warp_mat = preprocess_frame(frame_bgr, bbox)
    input_tensor = input_tensor.to(DEVICE)

    # Model inference
    t_inf_start = time.perf_counter()
    with torch.no_grad():
        if flip_test:
            heatmaps = MODEL(input_tensor)
            input_flipped = torch.flip(input_tensor, dims=[3])
            hm_flipped = torch.flip(MODEL(input_flipped), dims=[3])
            hm_swapped = hm_flipped.clone()
            for a, b in MODEL.flip_pairs:
                hm_swapped[:, a] = hm_flipped[:, b]
                hm_swapped[:, b] = hm_flipped[:, a]
            heatmaps = (heatmaps + hm_swapped) * 0.5
        else:
            heatmaps = MODEL(input_tensor)

    inference_ms = (time.perf_counter() - t_inf_start) * 1000.0

    # Postprocessing
    keypoints, scores = postprocess_heatmaps(
        heatmaps.cpu().numpy(), warp_mat, CODEC
    )
    total_ms = (time.perf_counter() - t_start) * 1000.0

    return keypoints, scores, inference_ms, total_ms


@app.route("/")
def index():
    """Render main application interface."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    """Return backend status, device info, and model footprint."""
    device_name = "NVIDIA CUDA GPU" if torch.cuda.is_available() else "Intel/AMD CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        # Approximate parameter size: ~34M params * 4 bytes ≈ 136 MB
        vram_mb = sum(p.numel() * p.element_size() for p in MODEL.parameters()) / (1024 * 1024)

    return jsonify({
        "status": "ready",
        "model_name": "SimpleBaseline ResNet-50 (MMPose)",
        "num_keypoints": 17,
        "input_size": [192, 256],
        "device": str(DEVICE),
        "device_name": device_name,
        "memory_mb": round(vram_mb, 2),
    })


@app.route("/api/predict_frame", methods=["POST"])
def predict_frame():
    """Predict pose on a single frame from webcam client stream."""
    try:
        data = request.get_json(force=True)
        image_data = data.get("image", "")
        score_thresh = float(data.get("score_thresh", 0.3))
        flip_test = bool(data.get("flip_test", False))
        draw_on_server = bool(data.get("render_overlay", True))

        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 JPEG/PNG
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        binary_data = base64.b64decode(image_data)
        nparr = np.frombuffer(binary_data, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            return jsonify({"error": "Failed to decode image frame"}), 400

        keypoints, scores, inference_ms, total_ms = run_inference_on_frame(
            frame_bgr, score_thresh=score_thresh, flip_test=flip_test
        )

        detected_count = int(np.sum(scores >= score_thresh))

        response = {
            "keypoints": keypoints.tolist(),
            "scores": scores.tolist(),
            "detected_count": detected_count,
            "inference_ms": round(inference_ms, 1),
            "total_ms": round(total_ms, 1),
            "people_count": 1 if detected_count > 3 else 0,
        }

        if draw_on_server:
            annotated = draw_pose(
                frame_bgr,
                keypoints,
                scores,
                COCO_SKELETON,
                score_thresh=score_thresh,
            )
            # Encode back to JPEG
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_base64 = base64.b64encode(buffer).decode("utf-8")
            response["annotated_image"] = f"data:image/jpeg;base64,{annotated_base64}"

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict_upload", methods=["POST"])
def predict_upload():
    """Process uploaded image file and return annotated result with telemetry."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        score_thresh = float(request.form.get("score_thresh", 0.3))
        flip_test = request.form.get("flip_test", "false").lower() == "true"

        in_memory_file = io.BytesIO(file.read())
        image_pil = Image.open(in_memory_file).convert("RGB")
        frame_rgb = np.array(image_pil)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        keypoints, scores, inference_ms, total_ms = run_inference_on_frame(
            frame_bgr, score_thresh=score_thresh, flip_test=flip_test
        )

        detected_count = int(np.sum(scores >= score_thresh))

        annotated = draw_pose(
            frame_bgr,
            keypoints,
            scores,
            COCO_SKELETON,
            score_thresh=score_thresh,
        )

        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        annotated_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "annotated_image": f"data:image/jpeg;base64,{annotated_base64}",
            "keypoints": keypoints.tolist(),
            "scores": scores.tolist(),
            "detected_count": detected_count,
            "inference_ms": round(inference_ms, 1),
            "total_ms": round(total_ms, 1),
            "people_count": 1 if detected_count > 3 else 0,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n========================================================")
    print(f"  HUMAN POSE ESTIMATION WEB APP (RED & BLACK THEME)")
    print(f"  SimpleBaseline (ResNet-50 + HeatmapHead)")
    print(f"  Server running on http://127.0.0.1:{port}")
    print(f"========================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)

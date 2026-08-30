# SimpleBaseline (Xiao et al. 2018) - Standalone Replication

A standalone, lightweight PyTorch implementation of **SimpleBaseline** for 2D Human Pose Estimation (Xiao et al., ECCV 2018), faithfully replicating the `open-mmlab/mmpose` `main` branch architecture (`td-hm_res50_8xb64-210e_coco-256x192.py`) with **zero dependencies on OpenMMLab libraries** (`mmcv`, `mmengine`, `mmpose`, `pycocotools`).

---

## 🌟 Key Features

- **Pure PyTorch & NumPy**: Built from scratch using standard `torch.nn` and `numpy`.
- **Exact MMPose Compatibility**: 100% parameter naming compatibility with official MMPose ResNet-50 checkpoints (338/338 keys matched).
- **Interactive Web Interface**: Cyber Red & Black themed real-time pose estimation web application with camera streaming, image upload mode, and live 6-card HUD telemetry.
- **GPU-Aware Telemetry**: GPU-synchronized timing (`torch.cuda.synchronize`), per-frame latency measurement, statistical summary reporting (Mean/Min/Max/P95), and CSV logging.
- **Top-Down Data Pipeline**: Complete standalone data augmentation and transformation pipeline (`GetBBoxCenterScale`, `RandomFlip`, `RandomBBoxTransform`, `TopdownAffine`, `GenerateTarget`, `PackPoseInputs`).
- **Complete Training Loop**: Full training script with Adam optimizer, LinearLR warmup + MultiStepLR schedule, checkpoint save/restore, and in-memory mock validation.
- **Automated Test Suite**: 11 unit and integration tests covering all model modules, loss, codec, dataset, and server endpoints.

---

## 🏛️ Architecture Overview

```
Input Image (B, 3, 256, 192)
      │
      ▼
ResNet-50 Backbone (4 stages, Bottleneck blocks)
      │ (Stage 4 output: B, 2048, 8, 6)
      ▼
HeatmapHead (3x Deconv: 2048 -> 256 -> 256 -> 256)
      │ (Upsampled: 8x6 -> 16x12 -> 32x24 -> 64x48)
      ▼
1x1 Conv (Final Projection)
      │
      ▼
Predicted Heatmap Logits (B, 17, 64, 48)
      ├──► KeypointMSELoss (Target-weighted MSE loss)
      └──► MSRAHeatmap Codec (Argmax + 1/4-pixel gradient refinement) ──► (B, 17, 2) Keypoints
```

---

## 📁 Repository Structure

| File | Description |
| :--- | :--- |
| `resnet_backbone.py` | 4-stage ResNet-50 backbone with `Bottleneck` blocks. Torchvision `state_dict` compatible. |
| `heatmap_head.py` | Progressive 3-stage deconvolution upsampling head with 1x1 conv outputting raw logits. |
| `msra_heatmap_codec.py` | 2D Gaussian heatmap encoding ($3\sigma$ radius) and quarter-pixel offset decoding. |
| `keypoint_mse_loss.py` | Target-weighted MSE loss masking invisible joints during heatmap regression. |
| `model.py` | Top-level `SimpleBaseline(nn.Module)` integrating backbone, head, codec, and loss. |
| `pipeline.py` | Standalone top-down augmentation transforms (`TopdownAffine`, `RandomFlip`, etc.). |
| `coco_dataset.py` | Pure Python COCO dataset loader parsing JSON annotations without `pycocotools`. |
| `train_config.py` | Exact training hyperparameters (Adam lr=5e-4, MultiStepLR [170, 200], 210 epochs). |
| `train.py` | Training loop with checkpoint save/resume and synthetic fallback test. |
| `load_pretrained_weights.py` | Official OpenMMLab checkpoint downloader, key inspector, and remapper. |
| `webcam_inference.py` | Live webcam pose estimation with GPU synchronization and statistical telemetry. |
| `app.py` & `templates/` | Flask web application with Red & Black cyber-styled UI. |
| `test_suite.py` | Comprehensive 11-module automated unit and integration test suite. |

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.9+
- PyTorch & Torchvision
- OpenCV (`opencv-python`)
- NumPy
- Flask

```bash
pip install torch torchvision opencv-python numpy flask pillow
```

---

### 2. Download Pretrained Weights & Sanity Check
Automatically download and map official MMPose weights into SimpleBaseline:
```bash
python load_pretrained_weights.py
```

---

### 3. Launch Web Interface (Red & Black Cyber Theme)
```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser to access:
- **Live Camera Mode**: Real-time pose estimation directly in the browser.
- **Upload Mode**: Drag-and-drop local images for pose estimation.
- **Live HUD Telemetry**: Real-time FPS, GPU Latency, HTTP Latency, Keypoint count, and VRAM memory footprint.

---

### 4. Real-time Webcam Pose Estimation with GPU Telemetry
```bash
# Run on default webcam
python webcam_inference.py --camera-id 0

# Run with test-time flip aggregation and CSV logging
python webcam_inference.py --camera-id 0 --flip-test --log-csv session_metrics.csv
```

When exiting (`'q'` or `Ctrl+C`), a statistical summary report is printed:
```text
============================================================
 METRICS SUMMARY REPORT (300 frames logged (excluding 5 initial warm-up frames))
============================================================
Metric               | Mean    | Min     | Max     | P95    
------------------------------------------------------------
FPS                  | 32.4    | 26.8    | 36.1    | 34.5   
GPU Latency (ms)     | 13.82   | 11.20   | 18.40   | 15.60  
People Detected      | 1.0     | 1.0     | 1.0     | 1.0    
Joints Identified    | 16.8    | 14.0    | 17.0    | 17.0   
VRAM Usage (MB)      | 136.50  | 136.50  | 136.50  | 136.50 
============================================================
```

---

### 5. Training on COCO Keypoints
```bash
# Full training on COCO dataset
python train.py --ann-file data/coco/annotations/person_keypoints_train2017.json --img-dir data/coco/train2017 --epochs 210 --batch-size 64 --device cuda

# Quick CPU test (runs on synthetic dataset if no paths provided)
python train.py
```

---

### 6. Run Automated Test Suite
Run the 11-module unit and integration test suite:
```bash
python test_suite.py
```
```text
test_01_resnet_backbone ... ok
test_02_heatmap_head ... ok
test_03_msra_heatmap_codec ... ok
test_04_keypoint_mse_loss ... ok
test_05_pipeline_transform ... ok
test_06_simplebaseline_full_model ... ok
test_07_coco_dataset ... ok
test_08_train_smoke_test ... ok
test_09_load_pretrained_weights ... ok
test_10_webcam_inference_pipeline ... ok
test_11_web_app_endpoints ... ok

----------------------------------------------------------------------
Ran 11 tests in 8.168s

OK
```

---

## 📜 References

- Xiao, Bin, Haiping Wu, and Yichen Wei. "Simple baselines for human pose estimation and tracking." *Proceedings of the European Conference on Computer Vision (ECCV)*. 2018.
- OpenMMLab MMPose: https://github.com/open-mmlab/mmpose

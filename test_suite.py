"""Comprehensive Test Suite for SimpleBaseline Replication.

Tests all components independently and integrated:
1. ResNet backbone
2. HeatmapHead
3. MSRAHeatmap Codec
4. KeypointMSELoss
5. Data Pipeline Transforms
6. SimpleBaseline Full Model & Flip Test
7. Optimizer & Scheduler Configuration
"""

import unittest
import numpy as np
import torch

from resnet_backbone import ResNet, Bottleneck
from heatmap_head import HeatmapHead
from msra_heatmap_codec import MSRAHeatmap, get_heatmap_maximum, refine_keypoints
from keypoint_mse_loss import KeypointMSELoss
from pipeline import build_topdown_pipeline, get_warp_matrix
from model import SimpleBaseline, COCO_FLIP_PAIRS
from train_config import build_optimizer_and_scheduler, TRAIN_CFG


class TestSimpleBaselineReplication(unittest.TestCase):

    def test_01_resnet_backbone(self):
        """Test ResNet backbone architecture and output shapes."""
        model = ResNet(layers=[3, 4, 6, 3], in_channels=3, out_indices=(3,))
        model.eval()
        x = torch.randn(2, 3, 256, 192)
        out = model(x)

        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].shape, (2, 2048, 8, 6))

    def test_02_heatmap_head(self):
        """Test HeatmapHead 3-stage deconvolution and output shape."""
        head = HeatmapHead(in_channels=2048, out_channels=17)
        head.eval()
        feats = (torch.randn(2, 2048, 8, 6),)
        out = head(feats)

        self.assertEqual(out.shape, (2, 17, 64, 48))
        self.assertEqual(len(head.deconv_layers), 9)  # 3 x (ConvTranspose2d, BatchNorm2d, ReLU)

    def test_03_msra_heatmap_codec(self):
        """Test MSRA Heatmap encoding and quarter-pixel decoding."""
        codec = MSRAHeatmap(input_size=(192, 256), heatmap_size=(48, 64), sigma=2.0)

        # Ground truth keypoints
        kpts = np.array([[[96.0, 128.0], [48.0, 64.0], [10.0, 10.0]]], dtype=np.float32)
        vis = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)

        encoded = codec.encode(kpts, vis)
        heatmaps = encoded["heatmaps"]
        weights = encoded["keypoint_weights"]

        self.assertEqual(heatmaps.shape, (3, 64, 48))
        self.assertEqual(weights.shape, (1, 3))
        self.assertEqual(weights[0, 2], 0.0)
        self.assertEqual(heatmaps[2].sum(), 0.0)
        self.assertGreater(heatmaps[0].max(), 0.9)

        # Decode
        decoded_kpts, scores = codec.decode(heatmaps)
        self.assertEqual(decoded_kpts.shape, (3, 2))
        self.assertEqual(scores.shape, (3,))
        self.assertAlmostEqual(scores[0], 1.0, places=4)
        self.assertAlmostEqual(scores[2], 0.0, places=4)

        # Coordinate accuracy
        np.testing.assert_allclose(decoded_kpts[0], kpts[0, 0], atol=1.0)
        np.testing.assert_allclose(decoded_kpts[1], kpts[0, 1], atol=1.0)

    def test_04_keypoint_mse_loss(self):
        """Test KeypointMSELoss target weighting and mask invariance."""
        loss_fn = KeypointMSELoss(use_target_weight=True, loss_weight=1.0)

        pred = torch.randn(2, 17, 64, 48)
        target = torch.randn(2, 17, 64, 48)
        weights = torch.ones(2, 17)
        weights[:, 3] = 0.0  # Joint 3 invisible

        loss_orig = loss_fn(pred, target, weights)
        self.assertGreater(loss_orig.item(), 0.0)
        self.assertEqual(loss_orig.ndim, 0)

        # Corrupt invisible joint in prediction
        pred_corrupted = pred.clone()
        pred_corrupted[:, 3] = torch.randn(2, 64, 48) * 1000.0
        loss_corrupted = loss_fn(pred_corrupted, target, weights)

        self.assertAlmostEqual(loss_orig.item(), loss_corrupted.item(), places=5)

    def test_05_pipeline_transform(self):
        """Test top-down data augmentation pipeline."""
        codec = MSRAHeatmap(input_size=(192, 256), heatmap_size=(48, 64), sigma=2.0)
        pipeline = build_topdown_pipeline(codec=codec, input_size=(192, 256), is_train=True)

        sample = {
            "img": np.zeros((300, 300, 3), dtype=np.uint8),
            "bbox": np.array([50, 50, 100, 150], dtype=np.float32),
            "keypoints": np.random.uniform(50, 150, (1, 17, 2)).astype(np.float32),
            "keypoints_visible": np.ones((1, 17), dtype=np.float32),
        }

        out = pipeline(sample)
        self.assertEqual(out["inputs"].shape, (3, 256, 192))
        self.assertEqual(out["target_heatmaps"].shape, (17, 64, 48))
        self.assertEqual(out["target_weights"].shape, (17,))

    def test_06_simplebaseline_full_model(self):
        """Test end-to-end SimpleBaseline forward, loss, backward, and flip prediction."""
        model = SimpleBaseline(num_joints=17, pretrained_backbone=False)

        # Forward
        x = torch.randn(2, 3, 256, 192)
        heatmaps = model(x)
        self.assertEqual(heatmaps.shape, (2, 17, 64, 48))

        # Loss & Backprop
        target_hm = torch.rand(2, 17, 64, 48)
        target_w = torch.ones(2, 17)
        loss = model.loss(x, target_hm, target_w)
        loss.backward()

        self.assertIsNotNone(model.head.final_layer.weight.grad)
        self.assertIsNotNone(next(model.backbone.parameters()).grad)

        # Predict with flip test
        kpts, scores = model.predict(x, flip_test=True)
        self.assertEqual(kpts.shape, (2, 17, 2))
        self.assertEqual(scores.shape, (2, 17))

    def test_07_coco_dataset(self):
        """Test CocoKeypointDataset parsing, filtering, and DataLoader collate_fn."""
        import tempfile
        import json
        import cv2
        import os
        from torch.utils.data import DataLoader
        from coco_dataset import CocoKeypointDataset, collate_fn

        codec = MSRAHeatmap(input_size=(192, 256), heatmap_size=(48, 64), sigma=2.0)
        pipeline = build_topdown_pipeline(codec=codec, input_size=(192, 256), is_train=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_ann_path = os.path.join(tmp_dir, "mock_person_kpts.json")
            mock_img_path = os.path.join(tmp_dir, "000000000001.jpg")

            mock_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(mock_img_path, mock_img)

            mock_coco = {
                "images": [{"id": 1, "file_name": "000000000001.jpg", "height": 480, "width": 640}],
                "annotations": [
                    {
                        "id": 101,
                        "image_id": 1,
                        "category_id": 1,
                        "iscrowd": 0,
                        "num_keypoints": 17,
                        "bbox": [100.0, 80.0, 200.0, 300.0],
                        "keypoints": [
                            140, 100, 2,  130, 95, 2,   150, 95, 2,   120, 110, 2,  160, 110, 2,
                            100, 150, 2,  180, 150, 2,  90, 200, 2,   190, 200, 2,  80, 250, 2,
                            200, 250, 2,  120, 230, 2,  160, 230, 2,  120, 300, 2,  160, 300, 2,
                            120, 370, 2,  160, 370, 2,
                        ],
                    }
                ],
            }

            with open(mock_ann_path, "w", encoding="utf-8") as f:
                json.dump(mock_coco, f)

            dataset = CocoKeypointDataset(ann_file=mock_ann_path, img_dir=tmp_dir, pipeline_transforms=pipeline)
            loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
            batch = next(iter(loader))

            self.assertEqual(len(dataset), 1)
            self.assertEqual(batch["inputs"].shape, (1, 3, 256, 192))
            self.assertEqual(batch["target_heatmaps"].shape, (1, 17, 64, 48))
            self.assertEqual(batch["target_weights"].shape, (1, 17))
            self.assertEqual(batch["img_id"][0], 1)

    def test_08_train_smoke_test(self):
        """Test train.py in-memory training and checkpointing."""
        from train import run_synthetic_smoke_test
        run_synthetic_smoke_test()

    def test_09_load_pretrained_weights(self):
        """Test weight remapping and pretrained loading on local cached checkpoint."""
        import os
        from load_pretrained_weights import load_pretrained, DEFAULT_CACHE_PATH
        if os.path.exists(DEFAULT_CACHE_PATH):
            model = SimpleBaseline(num_joints=17, pretrained_backbone=False)
            model.eval()
            load_pretrained(model, checkpoint_path=DEFAULT_CACHE_PATH)
            x = torch.randn(1, 3, 256, 192)
            out = model(x)
            self.assertEqual(out.shape, (1, 17, 64, 48))

    def test_10_webcam_inference_pipeline(self):
        """Test frame preprocessing, heatmap decoding, inverse affine mapping, and pose drawing."""
        import cv2
        from webcam_inference import preprocess_frame, postprocess_heatmaps, draw_pose, COCO_SKELETON
        from msra_heatmap_codec import MSRAHeatmap

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = [50, 50, 200, 300]
        input_tensor, warp_mat = preprocess_frame(frame, bbox)

        self.assertEqual(input_tensor.shape, (1, 3, 256, 192))
        self.assertEqual(warp_mat.shape, (2, 3))

        # Synthetic heatmaps
        heatmaps = np.zeros((1, 17, 64, 48), dtype=np.float32)
        heatmaps[0, :, 32, 24] = 1.0  # Center activation

        codec = MSRAHeatmap(input_size=(192, 256), heatmap_size=(48, 64), sigma=2.0)
        orig_kpts, scores = postprocess_heatmaps(heatmaps, warp_mat, codec)

        self.assertEqual(orig_kpts.shape, (17, 2))
        self.assertEqual(scores.shape, (17,))

        annotated = draw_pose(frame, orig_kpts, scores, COCO_SKELETON, score_thresh=0.3)
        self.assertEqual(annotated.shape, frame.shape)

    def test_11_web_app_endpoints(self):
        """Test Flask web app HTML route and JSON API endpoints."""
        import app
        import base64
        import cv2

        client = app.app.test_client()

        # 1. Test index page
        res = client.get('/')
        self.assertEqual(res.status_code, 200)

        # 2. Test status endpoint
        res_status = client.get('/api/status')
        self.assertEqual(res_status.status_code, 200)
        data = res_status.get_json()
        self.assertEqual(data['status'], 'ready')
        self.assertEqual(data['num_keypoints'], 17)

        # 3. Test predict_frame endpoint
        test_img = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', test_img)
        b64_str = base64.b64encode(buf).decode('utf-8')
        res_pred = client.post('/api/predict_frame', json={'image': f'data:image/jpeg;base64,{b64_str}', 'score_thresh': 0.3})
        self.assertEqual(res_pred.status_code, 200)
        pred_data = res_pred.get_json()
        self.assertIn('annotated_image', pred_data)
        self.assertIn('inference_ms', pred_data)


if __name__ == "__main__":
    unittest.main(verbosity=2)






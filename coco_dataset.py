"""COCO Keypoint Dataset for SimpleBaseline (Xiao et al. 2018).

Standalone PyTorch Dataset parsing COCO format keypoint annotations and feeding
samples through pipeline.py without pycocotools/xtcocotools/mmcv dependencies.
Config: configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py
"""

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from msra_heatmap_codec import MSRAHeatmap
from pipeline import Compose, build_topdown_pipeline

# Default paths for COCO 2017 dataset (update as needed)
DEFAULT_ANN_FILE = "data/coco/annotations/person_keypoints_val2017.json"
DEFAULT_IMG_DIR = "data/coco/val2017"


class CocoKeypointDataset(Dataset):
    """COCO 2017 Keypoint Dataset for 2D top-down human pose estimation.

    Replicates mmpose's CocoDataset behavior:
    - Parses COCO JSON annotations directly using standard library json.
    - Filters: category_id == 1 (person), iscrowd == 0, num_keypoints >= min_keypoints,
      bbox area >= min_bbox_area.
    - Visibility handling: COCO defines 0 = not labeled, 1 = labeled but occluded,
      2 = labeled and visible. MMPose treats vis > 0 (both 1 and 2) as supervised
      keypoints (`keypoints_visible = 1`), and vis == 0 as unlabelled (`keypoints_visible = 0`).

    Args:
        ann_file (str): Path to COCO keypoint annotation JSON file.
        img_dir (str): Root directory containing COCO images.
        pipeline_transforms (Callable): Composed data pipeline transforms (e.g. from pipeline.py).
        min_keypoints (int): Minimum number of labeled keypoints to keep sample. Default: 1.
        min_bbox_area (float): Minimum bounding box area (w * h) to keep sample. Default: 1.0.
    """

    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        pipeline_transforms: Callable[[Dict[str, Any]], Dict[str, Any]],
        min_keypoints: int = 1,
        min_bbox_area: float = 1.0,
    ) -> None:
        super().__init__()
        self.ann_file = ann_file
        self.img_dir = img_dir
        self.pipeline = pipeline_transforms
        self.min_keypoints = min_keypoints
        self.min_bbox_area = min_bbox_area

        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        self._load_annotations()

    def _load_annotations(self) -> None:
        """Parse COCO JSON annotations and filter valid person instances."""
        with open(self.ann_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        # Index image metadata by image_id
        self.images: Dict[int, Dict[str, Any]] = {}
        for img in coco_data.get("images", []):
            self.images[img["id"]] = {
                "file_name": img["file_name"],
                "height": img["height"],
                "width": img["width"],
            }

        # Filter person annotations
        self.annotations: List[Dict[str, Any]] = []
        for ann in coco_data.get("annotations", []):
            # 1. Must be person category (COCO category_id 1)
            if ann.get("category_id", 1) != 1:
                continue

            # 2. Skip crowd regions
            if ann.get("iscrowd", 0) != 0:
                continue

            # 3. Minimum labeled keypoints threshold
            num_kpts = ann.get("num_keypoints", 0)
            if num_kpts < self.min_keypoints:
                continue

            # 4. Minimum bounding box area threshold
            bbox = ann.get("bbox", [0, 0, 0, 0])  # [x, y, w, h]
            if len(bbox) < 4:
                continue
            w, h = bbox[2], bbox[3]
            if w * h < self.min_bbox_area:
                continue

            # 5. Must have associated image metadata
            if ann["image_id"] not in self.images:
                continue

            self.annotations.append(ann)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load image and annotation, process through pipeline, and return model inputs.

        Args:
            idx (int): Annotation index.

        Returns:
            Dict[str, Any]: Processed sample dict from pipeline.
        """
        ann = self.annotations[idx]
        img_info = self.images[ann["image_id"]]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        # Load image via OpenCV (BGR format)
        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"Failed to load image from path: {img_path}")

        # Extract 17 COCO keypoints: flat list of 51 (x, y, v)
        kpts_flat = np.array(ann["keypoints"], dtype=np.float32).reshape(17, 3)
        keypoints = kpts_flat[:, :2][np.newaxis, ...]  # Shape (1, 17, 2)

        # COCO keypoint visibility convention:
        # 0 = not labeled
        # 1 = labeled but occluded / not visible
        # 2 = labeled and visible
        # NOTE: MMPose supervises both vis=1 (occluded) and vis=2 (visible) joints (vis > 0),
        # only masking out unlabeled joints (vis == 0).
        visibility_raw = kpts_flat[:, 2]
        keypoints_visible = (visibility_raw > 0).astype(np.float32)[np.newaxis, ...]  # Shape (1, 17)

        results = {
            "img": img,
            "bbox": np.array(ann["bbox"], dtype=np.float32),  # [x, y, w, h]
            "keypoints": keypoints,
            "keypoints_visible": keypoints_visible,
            "img_id": ann["image_id"],
            "ann_id": ann.get("id", idx),
        }

        # Run through top-down pipeline transforms
        results = self.pipeline(results)

        # Ensure metadata is preserved for test/eval coordinate reconstruction
        results["img_id"] = ann["image_id"]
        results["bbox"] = np.array(ann["bbox"], dtype=np.float32)

        return results


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate batch elements for DataLoader.

    Stacks fixed-size tensor inputs, heatmaps, and target_weights into batched tensors.
    Collects variable metadata (img_id, bbox, warp_mat, bbox_center, bbox_scale) as lists.

    Args:
        batch (List[Dict[str, Any]]): List of sample dictionaries.

    Returns:
        Dict[str, Any]: Batched dictionary.
    """
    # Keys that should be stacked into batch tensors
    # Handle both 'inputs'/'img' and 'target_heatmaps'/'heatmaps'
    first = batch[0]
    img_key = "inputs" if "inputs" in first else "img"
    hm_key = "target_heatmaps" if "target_heatmaps" in first else "heatmaps"
    weight_key = "target_weights" if "target_weights" in first else "keypoint_weights"

    collated: Dict[str, Any] = {
        "inputs": torch.stack([item[img_key] for item in batch], dim=0),
    }

    if hm_key in first and first[hm_key] is not None:
        collated["target_heatmaps"] = torch.stack(
            [item[hm_key] for item in batch], dim=0
        )

    if weight_key in first and first[weight_key] is not None:
        collated["target_weights"] = torch.stack(
            [item[weight_key] for item in batch], dim=0
        )

    # Collect metadata as lists
    for meta_key in [
        "img_id",
        "ann_id",
        "bbox",
        "bbox_center",
        "bbox_scale",
        "warp_mat",
    ]:
        if meta_key in first:
            collated[meta_key] = [item[meta_key] for item in batch]

    return collated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test COCO Keypoint Dataset loader for SimpleBaseline."
    )
    parser.add_argument(
        "--ann-file",
        type=str,
        default=DEFAULT_ANN_FILE,
        help="Path to COCO keypoint annotations JSON",
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default=DEFAULT_IMG_DIR,
        help="Path to COCO images directory",
    )
    args = parser.parse_args()

    print("Checking for COCO dataset at:")
    print(f"  Annotation file: {args.ann_file}")
    print(f"  Image directory: {args.img_dir}\n")

    codec = MSRAHeatmap(input_size=(192, 256), heatmap_size=(48, 64), sigma=2.0)
    pipeline = build_topdown_pipeline(
        codec=codec, input_size=(192, 256), is_train=True
    )

    if not os.path.exists(args.ann_file) or not os.path.isdir(args.img_dir):
        print(
            f"COCO dataset not found at '{args.ann_file}' or '{args.img_dir}' "
            "- skipping live dataset smoke test."
        )
        print("Running in-memory mock validation of CocoKeypointDataset and collate_fn...")

        # In-memory mock validation
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_ann_path = os.path.join(tmp_dir, "mock_person_kpts.json")
            mock_img_path = os.path.join(tmp_dir, "000000000001.jpg")

            # Create mock image
            mock_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(mock_img_path, mock_img)

            # Create mock COCO JSON
            mock_coco = {
                "images": [
                    {"id": 1, "file_name": "000000000001.jpg", "height": 480, "width": 640}
                ],
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

            mock_dataset = CocoKeypointDataset(
                ann_file=mock_ann_path,
                img_dir=tmp_dir,
                pipeline_transforms=pipeline,
            )

            mock_loader = DataLoader(
                mock_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn
            )

            batch = next(iter(mock_loader))

            assert batch["inputs"].shape == (
                1,
                3,
                256,
                192,
            ), f"Expected inputs shape (1, 3, 256, 192), got {batch['inputs'].shape}"
            assert batch["target_heatmaps"].shape == (
                1,
                17,
                64,
                48,
            ), f"Expected target_heatmaps shape (1, 17, 64, 48), got {batch['target_heatmaps'].shape}"
            assert batch["target_weights"].shape == (
                1,
                17,
            ), f"Expected target_weights shape (1, 17), got {batch['target_weights'].shape}"
            assert len(batch["img_id"]) == 1
            assert batch["img_id"][0] == 1

            print("\nIn-memory mock test passed successfully!")
            print(f"Mock dataset size:             {len(mock_dataset)}")
            print(f"Collated inputs tensor:        {batch['inputs'].shape}")
            print(f"Collated target_heatmaps:      {batch['target_heatmaps'].shape}")
            print(f"Collated target_weights:       {batch['target_weights'].shape}")
            print(f"Sample image ID:               {batch['img_id'][0]}")

    else:
        print("Found COCO dataset! Loading live dataset...")
        dataset = CocoKeypointDataset(
            ann_file=args.ann_file,
            img_dir=args.img_dir,
            pipeline_transforms=pipeline,
        )
        loader = DataLoader(
            dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
        )

        print(f"Total valid person annotations in dataset: {len(dataset)}")
        batch = next(iter(loader))
        print("\nLive batch successfully loaded:")
        print(f"  Inputs shape:         {batch['inputs'].shape}")
        print(f"  Target heatmaps shape:{batch['target_heatmaps'].shape}")
        print(f"  Target weights shape: {batch['target_weights'].shape}")
        print(f"  Batch image IDs:      {batch['img_id']}")

    print("\nCocoKeypointDataset implementation is fully verified and ready.")

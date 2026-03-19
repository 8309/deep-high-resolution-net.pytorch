#!/usr/bin/env python3
"""Headless demo snapshot for cluster jobs (no GUI/imshow)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "lib"))

import models  # noqa: E402
from config import cfg, update_config  # noqa: E402
from core.inference import get_final_preds  # noqa: E402
from utils.transforms import get_affine_transform  # noqa: E402


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

SKELETON = [
    [1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9],
    [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
]

COCO_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
    [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
]

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless HRNet demo image inference")
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--model-file", required=False, default="", type=str)
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    args.modelDir = ""
    args.logDir = ""
    args.dataDir = ""
    args.prevModelDir = ""

    if args.model_file:
        args.opts = list(args.opts) + ["TEST.MODEL_FILE", args.model_file]

    return args


def box_to_center_scale(box, model_image_width, model_image_height):
    center = np.zeros((2), dtype=np.float32)
    (x1, y1), (x2, y2) = box
    box_width = x2 - x1
    box_height = y2 - y1

    center[0] = x1 + box_width * 0.5
    center[1] = y1 + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio

    scale = np.array([box_width / pixel_std, box_height / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def get_person_detection_boxes(model, image_bgr, threshold=0.7):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb / 255.0).permute(2, 0, 1).float().to(CTX)
    pred = model([img_tensor])

    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in pred[0]["labels"].detach().cpu().numpy().tolist()]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in pred[0]["boxes"].detach().cpu().numpy().tolist()]
    pred_scores = pred[0]["scores"].detach().cpu().numpy().tolist()

    person_boxes = []
    for cls_name, box, score in zip(pred_classes, pred_boxes, pred_scores):
        if cls_name == "person" and score >= threshold:
            person_boxes.append(box)

    return person_boxes


def get_pose_preds(pose_model, image, center, scale):
    trans = get_affine_transform(center, scale, 0, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR,
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model_input = transform(model_input).unsqueeze(0).to(CTX)
    pose_model.eval()

    with torch.no_grad():
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]),
        )

    return preds


def draw_pose(keypoints, image_bgr):
    for idx, (a, b) in enumerate(SKELETON):
        x_a, y_a = int(keypoints[a][0]), int(keypoints[a][1])
        x_b, y_b = int(keypoints[b][0]), int(keypoints[b][1])
        color = COCO_COLORS[idx]
        cv2.circle(image_bgr, (x_a, y_a), 4, color, -1)
        cv2.circle(image_bgr, (x_b, y_b), 4, color, -1)
        cv2.line(image_bgr, (x_a, y_a), (x_b, y_b), color, 2)


def main() -> int:
    args = parse_args()
    update_config(cfg, args)

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    image_path = Path(args.image).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not image_path.exists():
        print(f"[ERROR] image not found: {image_path}")
        return 2

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"[ERROR] failed to read image: {image_path}")
        return 2

    pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
    if not cfg.TEST.MODEL_FILE:
        print("[ERROR] TEST.MODEL_FILE is empty")
        return 2
    pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location=CTX), strict=False)
    if torch.cuda.is_available():
        pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
    pose_model.to(CTX)
    pose_model.eval()

    person_boxes = []
    try:
        detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        detector.to(CTX)
        detector.eval()
        person_boxes = get_person_detection_boxes(detector, image_bgr, threshold=0.8)
    except Exception as exc:
        print(f"[WARN] detector unavailable; fallback to full-image box: {exc}")

    if not person_boxes:
        h, w = image_bgr.shape[:2]
        person_boxes = [[(0, 0), (w - 1, h - 1)]]

    for box in person_boxes:
        cv2.rectangle(image_bgr, (int(box[0][0]), int(box[0][1])), (int(box[1][0]), int(box[1][1])), (0, 255, 0), 2)
        center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
        image_pose = image_bgr.copy() if not cfg.DATASET.COLOR_RGB else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        preds = get_pose_preds(pose_model, image_pose, center, scale)
        for kpt in preds:
            draw_pose(kpt, image_bgr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image_bgr)
    print(f"[INFO] wrote demo snapshot: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

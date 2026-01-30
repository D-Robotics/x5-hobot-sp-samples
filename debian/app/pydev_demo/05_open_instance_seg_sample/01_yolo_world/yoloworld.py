#!/usr/bin/env python3
# Copyright (c) 2025 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# flake8: noqa: E501
# flake8: noqa: E402

import os
import sys
import json
import argparse
from typing import List, Tuple

import cv2
import numpy as np

# Append parent directory to sys.path to enable local 'utils' module imports
sys.path.append(os.path.abspath("../.."))
import hbm_runtime
import utils.common_utils as common
import utils.draw_utils as draw


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    @brief Compute IoU between one box and a set of boxes.
    @param box Single box [x1, y1, x2, y2].
    @param boxes Array of boxes [N, 4].
    @return IoU array of shape [N].
    """
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    iou = intersection_area / np.maximum(union_area, 1e-6)
    return iou


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """
    @brief Non-Maximum Suppression on detection boxes.
    @param boxes Array of boxes [N, 4].
    @param scores Confidence scores [N].
    @param iou_threshold IoU threshold for suppression.
    @return Indices of kept boxes.
    """
    if boxes.size == 0:
        return []

    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes: List[int] = []

    while sorted_indices.size > 0:
        box_id = int(sorted_indices[0])
        keep_boxes.append(box_id)

        if sorted_indices.size == 1:
            break

        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


class YoloWorldDetector:
    """
    @brief YOLO-World text-driven detection using HB_HBMRuntime.

    This class handles vocabulary embedding loading, preprocessing,
    inference, and postprocessing to obtain final detection boxes.
    """

    def __init__(self, opt: "argparse.Namespace") -> None:
        """
        @brief Initialize detector with model, vocabulary, and prompts.

        @param opt Command line options:
            - model_path (str): Path to BPU model (*.bin).
            - label_file (str): Path to vocabulary embeddings JSON.
            - prompts (str): Comma-separated list of query words.
            - score_thres (float): Score threshold.
            - nms_thres (float): NMS IoU threshold.
        """
        self.label_file = opt.label_file
        self.prompts = [p.strip() for p in opt.prompts.split(",") if p.strip()]
        if not self.prompts:
            raise ValueError("At least one prompt must be provided.")

        self.score_thres = opt.score_thres
        self.nms_thres = opt.nms_thres

        # Load vocabulary embeddings
        with open(self.label_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        json_keys = list(data.keys())
        self.classes = json_keys

        # Build mapping from prompts to indices and embeddings
        key_indices: List[int] = []
        text_embeddings: List[np.ndarray] = []

        for voc in self.prompts:
            if voc not in json_keys:
                raise ValueError(f"Prompt '{voc}' not found in vocabulary.")
            key_index = json_keys.index(voc)
            key_indices.append(key_index)
            text_embeddings.append(np.array(data[voc], dtype=np.float32))

        # Pad to 32 prompts as in the original notebook
        if len(key_indices) == 0:
            raise ValueError("No valid prompts found.")

        while len(key_indices) < 32:
            key_indices.append(key_indices[-1])
            text_embeddings.append(text_embeddings[-1])

        self.key_indices = np.array(key_indices, dtype=np.int32)  # shape (32,)

        # Initialize HB_HBMRuntime model
        self.model = hbm_runtime.HB_HBMRuntime(opt.model_path)

        # Extract model IO info
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]

        # Adapt text embeddings shape to match model's 4D input requirement
        vocab_input_shape = self.input_shapes[self.input_names[1]]
        flat_embeddings = np.array(text_embeddings, dtype=np.float32).reshape(-1)
        if flat_embeddings.size != int(np.prod(vocab_input_shape)):
            raise ValueError(
                f"Text embedding size {flat_embeddings.size} does not match "
                f"model vocab input size {np.prod(vocab_input_shape)}."
            )
        self.text_embeddings = flat_embeddings.reshape(vocab_input_shape)

    def set_scheduling_params(self,
                              priority: int | None = None,
                              bpu_cores: list[int] | None = None) -> None:
        """
        @brief Set runtime scheduling parameters.

        @param priority (int, optional): Inference priority (0~255)
        @param bpu_cores (list[int], optional): List of BPU core indexes to use
        @return None
        """
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}

        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        @brief Preprocess input image to model input tensor.

        @param img BGR input image.
        @return (input_tensor, scale):
            - input_tensor: float32 tensor of shape (1, 3, 640, 640) in RGB.
            - scale: scale factor to map model coordinates back to original image.
        """
        img_h, img_w = img.shape[:2]
        resize_scale = 640.0 / max(img_h, img_w)
        scale = max(img_h, img_w) / 640.0

        image_resized = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)
        input_image = np.zeros((640, 640, 3), dtype=np.float32)
        input_image[: image_resized.shape[0], : image_resized.shape[1], :] = image_resized

        # BGR -> RGB and to NCHW
        input_image = input_image[:, :, [2, 1, 0]]
        input_image = input_image[None].transpose(0, 3, 1, 2)

        return input_image, scale

    def forward(self, input_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        @brief Run model inference.

        @param input_tensor Preprocessed image tensor.
        @return (classes_scores, bboxes):
            - classes_scores: shape (1, N, 32)
            - bboxes: shape (1, N, 4)
        """
        # Prepare input dict for HB_HBMRuntime
        inputs = {
            self.model_name: {
                self.input_names[0]: input_tensor,
                self.input_names[1]: self.text_embeddings
            }
        }
        outputs = self.model.run(inputs)[self.model_name]

        classes_scores = outputs[self.output_names[0]].squeeze(-1)
        bboxes = outputs[self.output_names[1]].squeeze(-1)
        return classes_scores, bboxes

    def post_process(self,
                     classes_scores: np.ndarray,
                     bboxes: np.ndarray,
                     scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        @brief Convert raw outputs to final boxes and scores.

        @param classes_scores Array of shape (1, N, 32).
        @param bboxes Array of shape (1, N, 4).
        @param scale Scale factor to map to original image.
        @return (boxes, cls_ids, scores):
            - boxes: [M, 4] in xyxy format on original image.
            - cls_ids: [M] indices into global vocabulary `self.classes`.
            - scores: [M] confidence scores.
        """
        rows = classes_scores.shape[1]

        boxes: List[List[float]] = []
        scores: List[float] = []
        class_ids: List[int] = []

        for i in range(rows):
            classes_score = classes_scores[0][i]
            _, max_score, _, (x, max_class_index) = cv2.minMaxLoc(classes_score)
            if max_score >= self.score_thres:
                box = [
                    bboxes[0][i][0],
                    bboxes[0][i][1],
                    bboxes[0][i][2],
                    bboxes[0][i][3],
                ]
                boxes.append(box)
                scores.append(float(max_score))
                class_ids.append(int(max_class_index))

        if not boxes:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32),
                    np.zeros((0,), dtype=np.float32))

        boxes_np = np.array(boxes, dtype=np.float32)
        scores_np = np.array(scores, dtype=np.float32)

        keep_indices = nms(boxes_np, scores_np, self.nms_thres)

        final_boxes = []
        final_scores = []
        final_cls_ids = []

        for idx in keep_indices:
            box = boxes_np[idx]
            # Map class index back to global vocabulary index
            vocab_idx = int(self.key_indices[class_ids[idx]])
            final_cls_ids.append(vocab_idx)
            final_scores.append(scores_np[idx])
            # Rescale coordinates to original resolution
            x1 = round(box[0] * scale)
            y1 = round(box[1] * scale)
            x2 = round(box[2] * scale)
            y2 = round(box[3] * scale)
            final_boxes.append([x1, y1, x2, y2])

        return (np.array(final_boxes, dtype=np.float32),
                np.array(final_cls_ids, dtype=np.int32),
                np.array(final_scores, dtype=np.float32))


def main() -> None:
    """
    @brief Entry point for running YOLO-World detection demo.

    This script loads a YOLO-World model, performs text-driven detection
    on a single image, and visualizes the result using bounding boxes.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str,
                        default="yolo_world.bin",
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""")
    parser.add_argument("--priority", type=int, default=0,
                        help="Model priority (0~255). 0 is lowest, 255 is highest. Default: 0.")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0],
                        help="BPU core indexes to run (e.g., --bpu-cores 0 1).")
    parser.add_argument("--test-img", type=str, default="dog.jpeg",
                        help="Path to test image.")
    parser.add_argument("--label-file", type=str,
                        default="offline_vocabulary_embeddings.json",
                        help="Path to vocabulary embeddings JSON file.")
    parser.add_argument("--prompts", type=str, default="dog",
                        help="Comma-separated list of query words, e.g. 'dog,cat'.")
    parser.add_argument("--img-save-path", type=str,
                        default="result.jpg",
                        help="Path to save result image.")
    parser.add_argument("--nms-thres", type=float, default=0.45,
                        help="IoU threshold for NMS.")
    parser.add_argument("--score-thres", type=float, default=0.05,
                        help="Score threshold for object filtering.")

    opt = parser.parse_args()

    # Auto-download model if missing (same URL as YOLO-EV11 sample)
    if not os.path.exists(opt.model_path):
        print(f"File {opt.model_path} does not exist. Downloading YOLO-World model...")
        os.system(
            "wget -c https://archive.d-robotics.cc/downloads/rdk_model_zoo/"
            "rdk_x5/yolo_world.bin"
        )

    # Instantiate detector
    detector = YoloWorldDetector(opt)

    # Set scheduling params and print model info
    detector.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)
    common.print_model_info(detector.model)

    # Load image
    img = common.load_image(opt.test_img)

    # Preprocess
    input_tensor, scale = detector.pre_process(img)

    # Inference
    classes_scores, bboxes = detector.forward(input_tensor)

    # Post-process
    boxes, cls_ids, scores = detector.post_process(classes_scores, bboxes, scale)

    # Draw detections
    if boxes.shape[0] > 0:
        draw.draw_boxes(img, boxes, cls_ids, scores, detector.classes, common.rdk_colors)
    else:
        print("No detections above threshold.")

    # Save result
    cv2.imwrite(opt.img_save_path, img)
    print(f"[Saved] Result saved to: {opt.img_save_path}")


if __name__ == "__main__":
    main()


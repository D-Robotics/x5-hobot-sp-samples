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
import cv2
import sys
import numpy as np
import hbm_runtime
import argparse
from typing import Optional, Dict, Tuple

# Append parent directory to sys.path to enable local 'utils' module imports
sys.path.append(os.path.abspath("../.."))
import utils.preprocess_utils as pre_utils
import utils.postprocess_utils as post_utils
import utils.common_utils as common
import utils.draw_utils as draw


class YoloV11_Seg:
    """
    @brief YOLOv11 instance segmentation wrapper using HB_HBMRuntime.

    This class supports preprocessing, inference, and postprocessing steps,
    including bounding box decoding, classification filtering, mask decoding, and resizing.
    """

    def __init__(self, opt):
        """
        @brief Initialize YOLOv11_Seg with model and parameters.

        @param opt (argparse.Namespace) Configuration options with fields:
            model_path, score_thres, is_open
        """
        # Load model and extract metadata
        self.model = hbm_runtime.HB_HBMRuntime(opt.model_path)
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]

        # Extract input resolution
        self.input_H = self.input_shapes[self.input_names[0]][2]
        self.input_W = self.input_shapes[self.input_names[0]][3]

        # Runtime and detection parameters
        self.score_thres = opt.score_thres
        self.conf_thres_raw = -np.log(1 / self.score_thres - 1)  # sigmoid inverse
        self.nms_thresh = opt.score_thres
        self.resize_type = 1
        self.classes_num = 80
        self.reg = 16
        self.mces_num = 32
        self.is_open = opt.is_open  # Whether to apply mask post-morphological ops

        # Feature map strides and anchor sizes
        self.strides = [8, 16, 32]
        self.anchor_sizes = [80, 40, 20]

        # Precompute regression bin weights
        self.weights_static = np.arange(self.reg, dtype=np.float32)[np.newaxis, np.newaxis, :]

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """
        @brief Set runtime scheduling parameters.

        @param priority (int, optional) Inference priority (0~255).
        @param bpu_cores (list[int], optional) Assigned BPU core indices.
        @return None
        """
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}

        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        @brief Preprocess image to model-required NV12 format.

        @param img (np.ndarray) Input BGR image.
        @return dict: {model_name: {input_name: y/uv tensors}}
        """
        resize_img = pre_utils.resized_image(img, self.input_W, self.input_H, self.resize_type)
        y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        nv12 = np.concatenate((y.reshape(-1), uv.reshape(-1)), axis=0)
        nv12 = nv12.reshape((1, self.input_H * 3 // 2, self.input_W, 1))

        return {
            self.model_name: {
                self.input_names[0]: nv12
            }
        }

    def forward(self,
                input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        @brief Run model inference.

        @param input_tensor (dict) Preprocessed input tensor.
        @return dict: Output tensors indexed by output name.
        """
        outputs = self.model.run(input_tensor)
        return outputs[self.model_name]

    def post_process(self,
                     outputs: Dict[str, np.ndarray],
                     img_w: int,
                     img_h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        @brief Decode model output into final boxes, scores, classes and masks.

        @param outputs (dict) Raw output tensors from inference.
        @param img_w (int) Original image width.
        @param img_h (int) Original image height.
        @return Tuple:
            - xyxy (np.ndarray): Bounding boxes in original image space (N, 4)
            - class_ids (np.ndarray): Class indices (N,)
            - scores (np.ndarray): Confidence scores (N,)
            - masks (np.ndarray): Segmentation masks (N, H, W)
        """
        all_dbboxes = []  # Decoded bounding boxes
        all_scores = []   # Classification scores
        all_ids = []      # Class IDs
        all_mces = []     # Mask coefficients

        # Dequantize outputs to FP32
        fp32_outputs = post_utils.dequantize_outputs(outputs, self.output_quants)

        # Retrieve shared mask prototype tensor
        protos_float32 = fp32_outputs[self.output_names[9]][0]
        Mask_H, Mask_W = protos_float32.shape[:2]

        # Process each scale level
        for i, (stride, anchor_size) in enumerate(zip(self.strides, self.anchor_sizes)):
            cls_key = self.output_names[3 * i]      # class confidence
            box_key = self.output_names[3 * i + 1]  # box regression
            mces_key = self.output_names[3 * i + 2]  # mask coeffs

            # Filter top-scoring detections
            scores, ids, valid_indices = post_utils.filter_classification(fp32_outputs[cls_key], self.conf_thres_raw)

            # Decode boxes and mask coefficients
            dbboxes = post_utils.decode_boxes(fp32_outputs[box_key], valid_indices,
                                              anchor_size, stride, self.weights_static)
            mces = post_utils.filter_mces(fp32_outputs[mces_key], valid_indices)

            # Accumulate results
            all_dbboxes.append(dbboxes)
            all_scores.append(scores)
            all_ids.append(ids)
            all_mces.append(mces)

        # Merge all scale-level results
        dbboxes = np.concatenate(all_dbboxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        ids = np.concatenate(all_ids, axis=0)
        mces = np.concatenate(all_mces, axis=0)

        # Apply NMS to reduce overlapping boxes
        keep = post_utils.NMS(dbboxes, scores, ids, self.nms_thresh)

        # Decode masks using prototypes and coefficients
        masks = post_utils.decode_masks(
            mces[keep], dbboxes[keep], protos_float32,
            self.input_W, self.input_H, Mask_W, Mask_H,
            mask_thresh=0.5
        )

        # Scale boxes back to original image resolution
        xyxy = post_utils.scale_coords_back(
            dbboxes[keep], img_w, img_h, self.input_W, self.input_H, self.resize_type
        )

        # Resize masks to box-aligned masks in original resolution
        resized_masks = post_utils.resize_masks_to_boxes(
            masks, xyxy, img_w, img_h, do_morph=self.is_open
        )

        return xyxy, ids[keep], scores[keep], resized_masks


def main():
    """
    @brief Entry point for running YOLOv11 instance segmentation demo.

    This script loads a YOLOv11 segmentation model, preprocesses the input image,
    performs inference and postprocessing, and visualizes the segmentation result
    with bounding boxes, colored masks, and optional edge contours.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        default='yolo11n_seg_bayese_640x640_nv12.bin',
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""")
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255).')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='BPU core indexes to run.')
    parser.add_argument('--test-img', type=str, default='bus.jpg',
                        help='Path to test image.')
    parser.add_argument('--label-file', type=str, default='coco_classes.names',
                        help='Path to label file.')
    parser.add_argument('--img-save-path', type=str, default='result.jpg',
                        help='Path to save result image.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='IoU threshold for NMS.')
    parser.add_argument('--score-thres', type=float, default=0.25, help='Score threshold.')
    parser.add_argument('--is-open', type=bool, default=True, help='Enable morphologyEx on masks.')
    parser.add_argument('--is-point', type=bool, default=True, help='Draw edge points.')

    opt = parser.parse_args()

    # Auto-download model if missing
    if not os.path.exists(opt.model_path):
        print(f"File {opt.model_path} does not exist. Downloading YOLOv11 segmentation model...")
        os.system("wget -c https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/"
                  "ultralytics_YOLO/yolo11n_seg_nashe_640x640_nv12.bin")
        opt.model_path = 'yolo11n_seg_nashe_640x640_nv12.bin'

    # Instantiate model wrapper
    yolov11_seg = YoloV11_Seg(opt)

    # Optional: set BPU runtime scheduling
    yolov11_seg.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print model metadata
    common.print_model_info(yolov11_seg.model)

    # Load test image
    img = common.load_image(opt.test_img)
    img_h, img_w = img.shape[:2]

    # Preprocessing
    input_array = yolov11_seg.pre_process(img)

    # Inference
    outputs = yolov11_seg.forward(input_array)

    # Postprocessing
    boxes, cls_ids, scores, masks = yolov11_seg.post_process(outputs, img_w, img_h)

    # Load label names
    coco_names = common.load_class_names(opt.label_file)

    # Draw bounding boxes
    draw.draw_boxes(img, boxes, cls_ids, scores, coco_names, common.rdk_colors)

    # Draw segmentation masks
    draw.draw_masks(img, boxes, masks, cls_ids, common.rdk_colors, alpha=0.4)

    # Draw mask contours (optional)
    if opt.is_point:
        draw.draw_contours(img, boxes, masks, cls_ids, common.rdk_colors, thickness=1)

    # Save output image
    cv2.imwrite(opt.img_save_path, img)
    print(f"[Saved] Result saved to: {opt.img_save_path}")


if __name__ == "__main__":
    main()

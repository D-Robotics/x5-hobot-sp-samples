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
import argparse
import hbm_runtime
import numpy as np
from typing import Dict, Optional, Tuple

# Append parent directory to sys.path to enable local 'utils' module imports
sys.path.append(os.path.abspath("../.."))
import utils.preprocess_utils as pre_utils
import utils.postprocess_utils as post_utils
import utils.common_utils as common
import utils.draw_utils as draw


coco_names = ["person"]


class YoloV11_Pose:
    """
    @brief YOLOv11 pose estimation wrapper using HB_HBMRuntime.

    This class handles preprocessing, inference, and postprocessing for
    keypoint detection models. It supports decoding bounding boxes and
    keypoints from multi-scale feature maps.
    """

    def __init__(self, opt: 'argparse.Namespace') -> None:
        """
        @brief Initialize the model and internal parameters.

        @param opt (argparse.Namespace): Options with attributes:
            - model_path (str): Path to BPU model file (*.bin)
            - score_thres (float): Confidence score threshold
        """
        # Load model from path
        self.model = hbm_runtime.HB_HBMRuntime(opt.model_path)

        # Extract model I/O info
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]

        # Get model input resolution
        self.input_H = self.input_shapes[self.input_names[0]][2]
        self.input_W = self.input_shapes[self.input_names[0]][3]

        # Inference thresholds and parameters
        self.score_thres = opt.score_thres
        self.conf_thres_raw = -np.log(1 / self.score_thres - 1)  # Sigmoid inverse
        self.nms_thresh = opt.score_thres
        self.resize_type = 1
        self.reg = 16

        # Anchor and stride settings
        self.strides = [8, 16, 32]
        self.anchor_sizes = [80, 40, 20]

        # Precompute bin center weights for DFL decoding
        self.weights_static = np.arange(self.reg, dtype=np.float32)[np.newaxis, np.newaxis, :]

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list[int]] = None) -> None:
        """
        @brief Set scheduling parameters such as priority and BPU cores.

        @param priority (int, optional): Runtime priority (0–255)
        @param bpu_cores (list[int], optional): List of BPU core indices to use
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
        @brief Preprocess the input image into NV12 format expected by the model.

        @param img (np.ndarray): Input image in BGR format (H, W, 3)
        @return dict: Nested dictionary containing Y and UV planes
        """
        # Resize to match model input size
        resize_img = pre_utils.resized_image(img, self.input_W, self.input_H, self.resize_type)

        # Convert to NV12 (Y + UV) format
        y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        nv12 = np.concatenate((y.reshape(-1), uv.reshape(-1)), axis=0)
        nv12 = nv12.reshape((1, self.input_H * 3 // 2, self.input_W, 1))

        return {
            self.model_name: {
                self.input_names[0]: nv12,
                # self.input_names[1]: uv
            }
        }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        @brief Run inference using the input tensor.

        @param input_tensor (dict): Input tensor formatted for HB_HBMRuntime.
        @return dict: Dictionary of model output tensors by name.
        """
        # Forward through HBM model
        outputs = self.model.run(input_tensor)
        return outputs[self.model_name]

    def post_process(self,
                     outputs: Dict[str, np.ndarray],
                     img_h: int,
                     img_w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        @brief Post-process model outputs into bounding boxes, keypoints, and scores.

        @param outputs (dict): Raw model outputs in FP32 after inference.
        @param img_h (int): Original image height.
        @param img_w (int): Original image width.
        @return tuple:
            - class_ids (np.ndarray, shape: [N])
            - scores (np.ndarray, shape: [N])
            - boxes (np.ndarray, shape: [N, 4])
            - keypoints_xy (np.ndarray, shape: [N, K, 2])
            - keypoints_score (np.ndarray, shape: [N, K])
        """
        # Containers for all scale-level results
        all_dbboxes = []
        all_scores = []
        all_ids = []
        all_kpts_xy = []
        all_kpts_score = []

        # Dequantize model outputs
        fp32_outputs = post_utils.dequantize_outputs(outputs, self.output_quants)

        # Iterate through feature map levels
        for i, (stride, anchor_size) in enumerate(zip(self.strides, self.anchor_sizes)):
            cls_key = self.output_names[3 * i]
            box_key = self.output_names[3 * i + 1]
            kpts_key = self.output_names[3 * i + 2]

            # Filter predictions by confidence
            scores, ids, valid_indices = \
                post_utils.filter_classification(fp32_outputs[cls_key], self.conf_thres_raw)

            # Decode DFL boxes and keypoints
            dbboxes = post_utils.decode_boxes(fp32_outputs[box_key], valid_indices,
                                              anchor_size, stride, self.weights_static)

            kpts_xy, kpts_score = post_utils.decode_kpts(fp32_outputs[kpts_key], valid_indices,
                                                         anchor_size, stride)

            # Accumulate per-level results
            all_dbboxes.append(dbboxes)
            all_scores.append(scores)
            all_ids.append(ids)
            all_kpts_xy.append(kpts_xy)
            all_kpts_score.append(kpts_score)

        # Merge predictions from all feature maps
        dbboxes = np.concatenate(all_dbboxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        ids = np.concatenate(all_ids, axis=0)
        kpts_xy = np.concatenate(all_kpts_xy, axis=0)
        kpts_score = np.concatenate(all_kpts_score, axis=0)

        # Apply Non-Maximum Suppression
        keep = post_utils.NMS(dbboxes, scores, ids, self.nms_thresh)

        # Scale boxes and keypoints to original image space
        xyxy = post_utils.scale_coords_back(dbboxes[keep], img_w, img_h,
                                            self.input_W, self.input_H, self.resize_type)

        kpts_xy, kpts_score = post_utils.scale_keypoints_to_original_image(
            kpts_xy[keep], kpts_score[keep], xyxy,
            img_w, img_h, self.input_W, self.input_H, self.resize_type
        )

        return ids[keep], scores[keep], xyxy, kpts_xy, kpts_score


def main() -> None:
    """
    @brief Entry point for running YOLOv11 pose estimation demo.

    This script loads a YOLOv11 pose model, performs image preprocessing,
    runs inference on a test image, applies postprocessing to extract boxes and keypoints,
    and visualizes the result with bounding boxes and keypoints overlaid.

    @return None
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default='yolo11n_pose_bayese_640x640_nv12.bin',
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""")
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255). 0 is lowest, 255 is highest. Default: 0.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='BPU core indexes to run. e.g. --bpu-cores 0 1')
    parser.add_argument('--test-img', type=str, default='bus.jpg',
                        help='Path to load test image.')
    parser.add_argument('--label-file', type=str, default='coco_classes.names',
                        help='Path to load label file.')
    parser.add_argument('--img-save-path', type=str, default='result.jpg',
                        help='Path to save result image.')
    parser.add_argument('--nms-thres', type=float, default=0.7,
                        help='IoU threshold for NMS.')
    parser.add_argument('--score-thres', type=float, default=0.25,
                        help='Score threshold for object classification.')
    parser.add_argument('--kpt-conf-thres', type=float, default=0.5,
                        help='Confidence threshold for keypoint display.')

    opt = parser.parse_args()

    # Auto-download model if not found
    if not os.path.exists(opt.model_path):
        print(f"File {opt.model_path} does not exist. Downloading YOLOv11 pose model...")
        os.system("wget -c https://archive.d-robotics.cc/downloads/rdk_model_zoo/"
                  "rdk_s100/ultralytics_YOLO/yolo11n_pose_nashe_640x640_nv12.bin")
        opt.model_path = 'yolo11n_pose_nashe_640x640_nv12.bin'

    # Instantiate model wrapper
    yoloV11_pose = YoloV11_Pose(opt)

    # Set runtime scheduling parameters
    yoloV11_pose.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print model meta info
    common.print_model_info(yoloV11_pose.model)

    # Load input image
    img = common.load_image(opt.test_img)
    img_h, img_w = img.shape[:2]

    # Preprocess input
    input_array = yoloV11_pose.pre_process(img)

    # Run inference
    outputs = yoloV11_pose.forward(input_array)

    # Post-process to extract boxes and keypoints
    ids, scores, boxes, kpts_xy, kpt_score = yoloV11_pose.post_process(outputs, img_h, img_w)

    # Load label names
    coco_names = common.load_class_names(opt.label_file)

    # Draw bounding boxes
    draw.draw_boxes(img, boxes, ids, scores, coco_names, common.rdk_colors)

    # Draw keypoints with confidence filter
    draw.draw_keypoints(img, kpts_xy, kpt_score, kpt_conf_thresh=opt.kpt_conf_thres)

    # Save result image
    cv2.imwrite(opt.img_save_path, img)
    print(f"[Saved] Result saved to: {opt.img_save_path}")


if __name__ == "__main__":
    main()

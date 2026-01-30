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
import hbm_runtime
import argparse
import numpy as np
from typing import Optional, Dict, Tuple

# Append parent directory to sys.path to enable local 'utils' module imports
sys.path.append(os.path.abspath("../.."))
import utils.preprocess_utils as preprocess
import utils.postprocess_utils as postprocess
import utils.common_utils as common
import utils.draw_utils as draw


class YoloV11:
    """
    @brief YOLOv11 object detection wrapper using HB_HBMRuntime backend.

    This class supports preprocessing, inference, and postprocessing including
    dequantization, anchor-based decoding, classification filtering, and NMS.
    """

    def __init__(self, opt):
        """
        @brief Initialize the YoloV11 model and parameters.

        @param opt (argparse.Namespace) Parsed options including:
            - model_path: Path to model file
            - score_thres: Confidence threshold
            - reg: Number of regression bins for bounding boxes
        """
        # Load model runtime
        self.model = hbm_runtime.HB_HBMRuntime(opt.model_path)

        # Extract model name and I/O metadata
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]

        # Input shape (H, W)
        self.input_H = self.input_shapes[self.input_names[0]][2]
        self.input_W = self.input_shapes[self.input_names[0]][3]

        # Runtime and decoding parameters
        self.score_thres = opt.score_thres
        self.conf_thres_raw = -np.log(1 / self.score_thres - 1)  # sigmoid inverse
        self.nms_thresh = opt.score_thres
        self.resize_type = 1
        self.classes_num = 80
        self.reg = 16

        # Feature map configuration
        self.strides = [8, 16, 32]              # Corresponding to feature map levels
        self.anchor_sizes = [80, 40, 20]        # Used for anchor generation
        self.weights_static = np.arange(self.reg, dtype=np.float32)[np.newaxis, np.newaxis, :]

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """
        @brief Set runtime scheduling parameters.

        @param priority (int, optional) Priority level (0~255)
        @param bpu_cores (list[int], optional) List of BPU core indices to assign
        @return None
        """
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}

        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    # def pre_process(self,
    #                 img: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    #     """
    #     @brief Preprocess input image into NV12 format.

    #     @param img (np.ndarray) Input image in BGR format.
    #     @return dict: Input tensor dict of shape {model_name: {input_name: tensor}}
    #     """
    #     resize_img = preprocess.resized_image(img, self.input_W, self.input_H, self.resize_type)
    #     y, uv = preprocess.bgr_to_nv12_planes(resize_img)

    #     return {
    #         self.model_name: {
    #             self.input_names[0]: y,
    #             self.input_names[1]: uv
    #         }
    #     }
    def pre_process(self, img):
        resize_img = preprocess.resized_image(img, self.input_W, self.input_H, self.resize_type)
        y, uv = preprocess.bgr_to_nv12_planes(resize_img)
        y = y.astype(np.uint8)
        uv = uv.astype(np.uint8)

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
        @brief Perform forward inference using the model runtime.

        @param input_tensor (dict) Prepared input tensors.
        @return dict: Output tensors indexed by output name.
        """
        outputs = self.model.run(input_tensor)
        return outputs[self.model_name]

    def post_process(self,
                     outputs: Dict[str, np.ndarray],
                     img_w: int,
                     img_h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        @brief Postprocess YOLOv11 model outputs to obtain final detection results.

        @param outputs (dict) Output tensors from inference.
        @param img_w (int) Original image width.
        @param img_h (int) Original image height.
        @return Tuple of:
            - xyxy (np.ndarray): Bounding boxes in [x1, y1, x2, y2] format (float32)
            - class_ids (np.ndarray): Class ID array
            - scores (np.ndarray): Confidence scores for each detection
        """
        all_dbboxes = []  # Decoded bounding boxes
        all_scores = []   # Detection confidence scores
        all_ids = []      # Class IDs

        # Dequantize output to FP32
        fp32_outputs = postprocess.dequantize_outputs(outputs, self.output_quants)

        # Loop through detection branches (for different scales)
        for i, (stride, anchor_size) in enumerate(zip(self.strides, self.anchor_sizes)):
            cls_key = self.output_names[2 * i]      # Classification output
            box_key = self.output_names[2 * i + 1]  # Bounding box output

            # Filter by raw confidence threshold
            scores, ids, valid_indices = postprocess.filter_classification(fp32_outputs[cls_key], self.conf_thres_raw)

            # Decode bounding boxes
            dbboxes = postprocess.decode_boxes(fp32_outputs[box_key], valid_indices,
                                               anchor_size, stride, self.weights_static)

            # Collect per-branch results
            all_dbboxes.append(dbboxes)
            all_scores.append(scores)
            all_ids.append(ids)

        # Concatenate all branches' results
        dbboxes = np.concatenate(all_dbboxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        ids = np.concatenate(all_ids, axis=0)

        # Apply NMS
        keep = postprocess.NMS(dbboxes, scores, ids, self.nms_thresh)

        # Rescale boxes to original image resolution
        xyxy = postprocess.scale_coords_back(
            dbboxes[keep], img_w, img_h, self.input_W, self.input_H, self.resize_type)

        return xyxy, ids[keep], scores[keep]


def main() -> None:
    """
    @brief YOLOv11 object detection sample.

    This function loads the model and input image, runs inference with HB_HBMRuntime,
    postprocesses outputs including NMS and coordinate scaling,
    draws bounding boxes on the image, and saves the result.

    @return None
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        # default='/opt/hobot/model/x5/basic/yolo11n_detect_nashe_640x640_nv12.bin',
                        default='yolo11n_detect_bayese_640x640_nv12.bin',
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""")
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255). 0 is lowest, 255 is highest. Default: 0.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help="List of BPU core indexes to run inference, e.g., --bpu-cores 0 1.")
    parser.add_argument('--test-img', type=str, default='kite.jpg',
                        help='Path to load test image.')
    parser.add_argument('--label-file', type=str,
                        default='coco_classes.names',
                        help='Path to load class label file.')
    parser.add_argument('--img-save-path', type=str, default='result.jpg',
                        help='Path to save output image with detection results.')
    parser.add_argument('--nms-thres', type=float, default=0.45,
                        help='IoU threshold for NMS.')
    parser.add_argument('--score-thres', type=float, default=0.25,
                        help='Confidence threshold to filter detections.')

    opt = parser.parse_args()

    # Download model if missing
    if not os.path.exists(opt.model_path):
        print(f"File {opt.model_path} does not exist. Downloading YOLOv11n model...")
        os.system("wget -c https://archive.d-robotics.cc/downloads/rdk_model_zoo/"
                  "rdk_x5/ultralytics_YOLO/yolo11n_detect_nashe_640x640_nv12.bin")
        opt.model_path = 'yolo11n_detect_nashe_640x640_nv12.bin'

    # Instantiate YOLOv11 model
    yolov11 = YoloV11(opt)

    # Set BPU scheduling (core binding and priority)
    yolov11.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print model metadata (input/output names, shapes, etc.)
    common.print_model_info(yolov11.model)

    # Load input image
    img: np.ndarray = common.load_image(opt.test_img)
    img_h, img_w = img.shape[:2]

    # Preprocess input image into model-compatible NV12 format
    input_array = yolov11.pre_process(img)

    # Run forward inference
    outputs = yolov11.forward(input_array)

    # Postprocess: decode + filter + NMS + rescale to original resolution
    boxes, cls_ids, scores = yolov11.post_process(outputs, img_w, img_h)

    # Load label names (COCO-style)
    coco_names = common.load_class_names(opt.label_file)

    # Draw detected boxes and class names on the image
    image = draw.draw_boxes(img, boxes, cls_ids, scores, coco_names, common.rdk_colors)

    # Save the result image
    cv2.imwrite(opt.img_save_path, image)
    print(f"[Saved] Result saved to: {opt.img_save_path}")


if __name__ == "__main__":
    main()

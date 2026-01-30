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
from typing import Optional, Dict, Tuple

# Append parent directory to sys.path to enable local 'utils' module imports
sys.path.append(os.path.abspath("../.."))
import utils.preprocess_utils as pre_utils
import utils.postprocess_utils as post_utils
import utils.common_utils as common
import utils.draw_utils as draw

# Feature map downsampling strides
STRIDES = np.array([8, 16, 32], dtype=np.int32)

# Anchors for each scale
ANCHORS = np.array([
    [10, 13], [16, 30], [33, 23],
    [30, 61], [62, 45], [59, 119],
    [116, 90], [156, 198], [373, 326]
], dtype=np.float32).reshape(3, 3, 2)


class YoloV5X:
    """
    @brief YOLOv5X object detection wrapper using HB_HBMRuntime.

    This class supports input preprocessing, inference execution,
    and postprocessing including decoding, confidence filtering, and NMS.
    """

    def __init__(self, opt):
        """
        @brief Initialize YOLOv5X model from config options.

        @param opt (argparse.Namespace) Parsed arguments including:
                    model_path, score_thres
        """
        # Load model and extract metadata
        self.model = hbm_runtime.HB_HBMRuntime(opt.model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]

        # # Input resolution

        self.input_H = self.input_shapes[self.input_names[0]][2]
        self.input_W = self.input_shapes[self.input_names[0]][3]

        # Detection parameters
        self.score_thres = opt.score_thres
        self.nms_thres = opt.nms_thres
        # self.resize_type = 1
        self.resize_type = 0
        self.classes_num = 80

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """
        @brief Set scheduling parameters such as BPU core allocation and priority.

        @param priority (int, optional) Inference priority [0-255].
        @param bpu_cores (list[int], optional) BPU core indices to run inference on.
        @return None
        """
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}

        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img):

        resize_img = pre_utils.resized_image(img, self.input_W, self.input_H, self.resize_type)
        y, uv = pre_utils.bgr_to_nv12_planes(resize_img)

        y = y.astype(np.uint8)
        uv = uv.astype(np.uint8)

        nv12 = np.concatenate((y.reshape(-1), uv.reshape(-1)), axis=0)
        nv12 = nv12.reshape((1, self.input_H * 3 // 2, self.input_W, 1))
        return {
            self.model_name: {
                self.input_names[0]: nv12
            }
        }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        @brief Perform inference using the HB_HBMRuntime.

        @param input_tensor (dict) Preprocessed input tensor dictionary.
        @return dict: Output tensors keyed by output name.
        """
        outputs = self.model.run(input_tensor)  
        return outputs[self.model_name]

    def post_process(self,
                    outputs: Dict[str, np.ndarray],
                    img_w: int,
                    img_h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        @brief Postprocess YOLO model outputs into final bounding boxes, scores, and classes.
        """
        
        # Step 1: Convert quantized outputs to float32
        fp32_outputs = post_utils.dequantize_outputs(outputs, self.output_quants)
        # Step 2: Decode YOLO outputs into unified predictions
        pred = post_utils.decode_outputs(self.output_names, fp32_outputs,
                                        STRIDES, ANCHORS, self.classes_num)
        # Step 3: Filter predictions by confidence threshold
        xyxy_boxes, score, cls = post_utils.filter_predictions(pred, self.score_thres)
        # Step 4: Non-Maximum Suppression (NMS)
        keep = post_utils.NMS(xyxy_boxes, score, cls, self.nms_thres)

        # Step 5: Rescale boxes to original image dimensions
        xyxy = post_utils.scale_coords_back(xyxy_boxes[keep], img_w, img_h,
                                            self.input_W, self.input_H, self.resize_type)
        return xyxy, cls[keep], score[keep]

def main() -> None:
    """
    @brief Run YOLOv5X object detection on a single image.

    This function parses command-line arguments, loads the YOLOv5X model,
    preprocesses the image, performs inference, postprocesses the results,
    and saves the output image with bounding boxes.

    @return None
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default='/opt/hobot/model/x5/basic/yolov5s_672x672_nv12.bin',
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""")
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255). 0 is lowest, 255 is highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help="List of BPU core indexes to run inference, e.g., --bpu-cores 0 1.")
    parser.add_argument('--test-img', type=str, default='kite.jpg',
                        help='Path to load test image.')
    parser.add_argument('--label-file', type=str, default='coco_classes.names',
                        help='Path to load COCO label file.')
    parser.add_argument('--img-save-path', type=str, default='result.jpg',
                        help='Path to save output image with detection results.')
    parser.add_argument('--nms-thres', type=float, default=0.45,
                        help='IoU threshold for Non-Maximum Suppression.')
    parser.add_argument('--score-thres', type=float, default=0.25,
                        help='Confidence score threshold for filtering detections.')

    opt = parser.parse_args()

    # Download model if missing (manual step, URL should be provided)
    if not os.path.exists(opt.model_path):
        print(f"[Warning] File {opt.model_path} does not exist. Please download yolov5x model manually.")
        return

    # Instantiate YOLOv5X model
    yolov5x = YoloV5X(opt)

    # Configure runtime scheduling (BPU cores, priority)
    yolov5x.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print basic model info
    common.print_model_info(yolov5x.model)

    # Load input image
    img: np.ndarray = common.load_image(opt.test_img)
    img_h, img_w = img.shape[:2]

    # Preprocess image to match model input
    input_array = yolov5x.pre_process(img)

    # Run inference
    outputs = yolov5x.forward(input_array)

    # Load label names (e.g., COCO class names)
    coco_names = common.load_class_names(opt.label_file)

    # Postprocess outputs to get boxes, class IDs, scores
    boxes, cls_ids, scores = yolov5x.post_process(outputs, img_w, img_h)

    print(f"检测到 {len(boxes)} 个目标: {', '.join([f'{coco_names[cls_id]}({score:.2f})' for cls_id, score in zip(cls_ids, scores)])}")

    # Draw detection results on image
    image = draw.draw_boxes(img, boxes, cls_ids, scores, coco_names, common.rdk_colors)

    # Save the resulting image
    cv2.imwrite(opt.img_save_path, image)
    print(f"[Saved] Result saved to: {opt.img_save_path}")


if __name__ == "__main__":
    main()

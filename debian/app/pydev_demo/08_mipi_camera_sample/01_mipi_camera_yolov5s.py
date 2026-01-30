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
import signal
import argparse
import hbm_runtime
import numpy as np
try:
    from hobot_vio import libsrcampy as srcampy
except ImportError:
    from hobot_vio_rdkx5 import libsrcampy as srcampy
from typing import Optional, Dict, Tuple
import cv2
# Append parent directory to sys.path to enable local 'utils' module imports
sys.path.append(os.path.abspath(".."))
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
        self.resize_type = 1
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

    def pre_process(self, img, original_width=1920, original_height=1080):
        y, uv = pre_utils.split_nv12_bytes(img, original_width, original_height)
        y_resized, uv_resized = pre_utils.resize_nv12_yuv(y, uv, 
                                                          self.input_H, self.input_W)
        y_input = y_resized[..., None][None, ...]  # Shape: (1, H, W, 1)
        uv_input = uv_resized[None, ...] 
        nv12 = np.concatenate((y_input.reshape(-1), uv_input.reshape(-1)), axis=0)
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

def get_display_res() -> tuple[int, int]:
    """
    @brief Get the display resolution from the system HDMI query tool.

    @details If the external tool `/usr/bin/get_hdmi_res` exists, it is executed to retrieve
             the HDMI resolution (width, height). If not found, it falls back to a default resolution (1920x1080).
             Returned resolution values are also clamped to a maximum of 1920x1080 and minimum of 0.

    @return tuple[int, int] Resolution width and height, defaulting to (1920, 1080) if tool is unavailable.
    """
    # Fallback default resolution if tool is not available
    if not os.path.exists("/usr/bin/get_hdmi_res"):
        return 1920, 1080

    import subprocess
    # Run external command to get HDMI resolution
    p = subprocess.Popen(["/usr/bin/get_hdmi_res"], stdout=subprocess.PIPE)
    result = p.communicate()

    # Parse comma-separated bytes result, e.g., b"1080,1920"
    res = result[0].split(b',')

    # Clamp values to valid ranges
    res[1] = max(min(int(res[1]), 1920), 0)  # width
    res[0] = max(min(int(res[0]), 1080), 0)  # height

    return int(res[1]), int(res[0])  # return as (width, height)

is_stop=False


def signal_handler(signal, frame):
    global is_stop
    print("Stopping!\n")
    is_stop=True


if __name__ == '__main__':
    """
    @brief Main entry point for YOLOv5X object detection using NV12 input and BPU backend.
    @details Initializes camera and display, loads BPU model, processes live camera input,
             and performs real-time object detection with overlay results.
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
                        help='Model priority (0~255). 0 is lowest, 255 is highest. Default: 0.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help="BPU core indexes to run. Provide a list of integers (e.g., --bpu-cores 0 1).")
    parser.add_argument('--label-file', type=str,
                        default='coco_classes.names',
                        help='Path to load label file.')
    parser.add_argument('--nms-thres', type=float, default=0.45,
                        help='IoU threshold for Non-Maximum Suppression.')
    parser.add_argument('--score-thres', type=float, default=0.25,
                        help='Confidence score threshold.')

    opt = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    # Initialize camera
    cam = srcampy.Camera()
    disp_w, disp_h = get_display_res()
    # print("==================!!!!!!!!!!!!!!!!!disp_w, disp_h",disp_w, disp_h)
    ### add 0107
    h,w = 1080,1920
    cam.open_cam(0,-1,-1,[w,disp_w],[h,disp_h],1080,1920)


    # Initialize display
    disp = srcampy.Display()
    disp.display(0, disp_w, disp_h)
    srcampy.bind(cam, disp)

    ### add0107
    # disp.display(3,disp_w,disp_h)

    # Auto download if model file not found
    if not os.path.exists(opt.model_path):
        print(f"file {opt.model_path} does not exist. download yolov5x model.")

    # Instantiate model object
    yolov5x = YoloV5X(opt)

    # Configure scheduling parameters for BPU inference
    yolov5x.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print model info (dimensions, I/O names, quant params)
    # common.print_model_info(yolov5x.model)

    # Load label classes
    coco_names = common.load_class_names(opt.label_file)

    while not is_stop:
        # Get frame from camera as NV12 bytes
        nv12_bytes = cam.get_img(2, w, h)

        input_array = yolov5x.pre_process(nv12_bytes, original_width=w, original_height=h)

        # Run inference
        outputs = yolov5x.forward(input_array)

        # Decode and filter predictions
        boxes, cls_ids, scores = yolov5x.post_process(outputs, w, h)        # Draw results on display overlay
        draw.draw_detections_on_disp(disp, boxes, cls_ids, scores, coco_names, 
                                    common.rdk_colors, chn=2,
                                    img_w=w, img_h=h, disp_w=disp_w, disp_h=disp_h)
    # Cleanup
    srcampy.unbind(cam, disp)
    cam.close_cam()
    disp.close()

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

        # Input resolution
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

    def pre_process(self, img: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        @brief Preprocess BGR image to model-required NV12 format.

        @param img (np.ndarray) Input image in BGR format.
        @return dict: Nested input tensor dictionary {model_name: {input_name: tensor}}
        """
        # Resize and convert to NV12
        resize_img = pre_utils.resized_image(img, self.input_W, self.input_H, self.resize_type)
        y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
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

        @param outputs (dict) Raw output tensors from inference.
        @param img_w (int) Width of original input image.
        @param img_h (int) Height of original input image.
        @return Tuple:
            - xyxy (np.ndarray): Final bounding boxes (N, 4) in original image coordinates.
            - cls (np.ndarray): Class indices (N).
            - score (np.ndarray): Detection confidence scores (N).
        """
        # Step 1: Convert quantized outputs to float32
        fp32_outputs = post_utils.dequantize_outputs(outputs, self.output_quants)
        #fp32_outputs = fp32_outputs.astype('float32')
        #print("fp32_outputs",fp32_outputs.dtype)

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


def is_usb_camera(device: str) -> bool:
    """
    @brief Check if a given device path corresponds to a valid USB camera.
    @param device The path to the video device (e.g. "/dev/video0").
    @return True if the device is a usable camera, otherwise False.
    """
    try:
        cap = cv2.VideoCapture(device)  # Try opening the video device
        if not cap.isOpened():
            return False
        cap.release()  # Release the camera if opened successfully
        return True
    except Exception:
        # Any error implies the device is not a usable camera
        return False


def find_first_usb_camera() -> str | None:
    """
    @brief Find the first available USB camera device under /dev.
    @return The device path (e.g. "/dev/video0") if found, otherwise None.
    """
    # List all /dev/video* device nodes
    video_devices = [os.path.join('/dev', dev) for dev in os.listdir('/dev') if dev.startswith('video')]

    for dev in video_devices:
        if is_usb_camera(dev):
            return dev  # Return the first working camera device

    return None  # No valid camera found


if __name__ == '__main__':
    """
    @brief Main entry for real-time object detection using YOLOv5X on a USB camera.
    This script loads a quantized model, opens a USB camera, performs real-time inference,
    and visualizes the results with bounding boxes and class labels.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        default='/app/model/basic/yolov5s_672x672_nv12.bin',
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""")
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255). 0 is lowest, 255 is highest. Default: 0.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help="BPU core indexes to run. Provide a list of integers (e.g., --bpu_cores 0 1). ")
    parser.add_argument('--label-file', type=str,
                        default='coco_classes.names',
                        help='Path to Load Label File.')
    parser.add_argument('--resize-type', type=int, choices=[0, 1], default=1,
                        help='Image resize type: 0 for direct resize, 1 for letterbox, default is 1.')
    parser.add_argument('--classes-num', type=int, default=80,
                        help='Classes Num to Detect.')
    parser.add_argument('--nms-thres', type=float, default=0.45,
                        help='IoU threshold.')
    parser.add_argument('--score-thres', type=float, default=0.25,
                        help='Confidence threshold.')
    opt = parser.parse_args()

    # Detect video device
    video_device = find_first_usb_camera()  # Auto-detect USB camera

    if video_device is None:
        print("No USB camera found.")
        sys.exit(-1)

    print(f"Opening video device: {video_device}")
    cap = cv2.VideoCapture(video_device)

    if not cap.isOpened():
        print(f"Failed to open video device: {video_device}")
        sys.exit(-1)

    print("Open USB camera successfully")

    # Set camera parameters: MJPEG codec, resolution 1920x1080, 30 FPS
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv2.CAP_PROP_FOURCC, codec)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Check model path existence
    if not os.path.exists(opt.model_path):
        print(f"file {opt.model_path} does not exist. download yolov5x model.")

    # Instantiate YOLOv5X model
    yolov5x = YoloV5X(opt)

    # Set runtime scheduling parameters
    yolov5x.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print model information
    common.print_model_info(yolov5x.model)

    # Load class names from label file
    coco_names = common.load_class_names(opt.label_file)

    print("Place the mouse in the display window and press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to get image from USB camera")
            break

        # Preprocess image
        img_h, img_w = frame.shape[:2]
        input_array = yolov5x.pre_process(frame)

        # Run inference
        outputs = yolov5x.forward(input_array)

        # Postprocess results to get boxes, class ids and scores
        boxes, cls_ids, scores = yolov5x.post_process(outputs, img_w, img_h)

        # Draw detection results
        vis = draw.draw_boxes(frame.copy(), boxes, cls_ids, scores, coco_names, common.rdk_colors)

        # Display image in fullscreen
        cv2.namedWindow("YOLOv5X Detection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("YOLOv5X Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("YOLOv5X Detection", vis)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

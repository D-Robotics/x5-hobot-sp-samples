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
import time
import json
import asyncio
import websockets
import x3_pb2
import hbm_runtime
import argparse
import numpy as np
from hobot_vio import libsrcampy as srcampy
from typing import Optional, Dict, Tuple

# Append parent directory to sys.path to enable local 'utils' module imports
sys.path.append(os.path.abspath(".."))
import utils.preprocess_utils as pre_utils
import utils.postprocess_utils as post_utils
import utils.common_utils as common


# Global constant for frames per second
FPS = 30

# Feature map downsampling strides
STRIDES = np.array([8, 16, 32], dtype=np.int32)

# Anchors for each scale
ANCHORS = np.array([
    [10, 13], [16, 30], [33, 23],
    [30, 61], [62, 45], [59, 119],
    [116, 90], [156, 198], [373, 326]
], dtype=np.float32).reshape(3, 3, 2)


def limit_display_coordinates(coordinates):
    """
    Limit bounding box coordinates to legal display range.
    Returns a new list with clamped and rounded integer coordinates.
    
    Args:
        coordinates: List of 4 floats [x1, y1, x2, y2]
    Returns:
        List of 4 integers [x1, y1, x2, y2] within valid image bounds
    """
    # Clamp coordinates to image bounds and round to integers
    x1 = max(0, min(1920, round(coordinates[0])))  # x1: [0, 1920]
    y1 = max(0, min(1072, round(coordinates[1])))  # y1: [0, 1072]
    x2 = max(0, min(1920, round(coordinates[2])))  # x2: [0, 1920]
    y2 = max(0, min(1072, round(coordinates[3])))  # y2: [0, 1072]
    
    # Ensure x2 > x1 and y2 > y1 (minimum box size of 1 pixel)
    if x2 <= x1:
        x2 = min(x1 + 1, 1920)
    if y2 <= y1:
        y2 = min(y1 + 1, 1072)
    
    return [x1, y1, x2, y2]


def serialize_message(frame_message, data, detection_classes):
    """
    Serialize detection results into a Protocol Buffer message.
    """
    if data:
        for result in data:
            target = x3_pb2.Target()
            bbox = result["bbox"]  # Bounding box coordinates
            score = result["score"]  # Detection score
            class_id = int(result["id"])  # Class ID

            # Limit and round bounding box coordinates to valid display range
            # bbox format: [x1, y1, x2, y2] (already in original image coordinates)
            limited_bbox = limit_display_coordinates(bbox)

            target.type_ = detection_classes[class_id]
            box = x3_pb2.Box()
            box.type_ = detection_classes[class_id]
            box.score_ = float(score)
            box.top_left_.x_ = int(limited_bbox[0])
            box.top_left_.y_ = int(limited_bbox[1])
            box.bottom_right_.x_ = int(limited_bbox[2])
            box.bottom_right_.y_ = int(limited_bbox[3])
            target.boxes_.append(box)
            frame_message.smart_msg_.targets_.append(target)

    return frame_message.SerializeToString()


def signal_handler(sig, frame):
    sys.exit(0)


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
                    model_path, score_thres, resize_type, classes_num
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

    def pre_process(self, img, original_width=1920, original_height=1080):
        """
        @brief 预处理 NV12 字节数据
        @param img NV12 字节数据
        @param original_width 原始图像宽度（默认 1920）
        @param original_height 原始图像高度（默认 1080）
        """
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


class CameraManager:
    """
    Manages camera capture and image encoding.

    Note:
    - The image height is set to 1072 instead of 1080 because JPU encoding
      requires dimensions to be 16-byte aligned.
    - Using 1072 ensures compatibility while keeping overhead minimal.
    """

    def __init__(self):
        self.cam = srcampy.Camera()
        # Open camera with 16-aligned height (1072 instead of 1080)
        self.cam.open_cam(0, -1, FPS, [512, 1920], [512, 1072])
        self.encoder = srcampy.Encoder()
        self.encoder.encode(0, 3, 1920, 1072)

    def get_resized_image(self, channel: int, width: int, height: int) -> np.ndarray:
        """
        Capture and return a resized image for inference.
        """
        img = self.cam.get_img(channel, width, height)
        return np.frombuffer(img, dtype=np.uint8)

    def get_original_image(self):
        """
        Capture and return the original full-resolution image.
        """
        return self.cam.get_img(2, 1920, 1072)

    def encode_image(self, image):
        """
        Encode an image using the encoder.
        """
        self.encoder.encode_file(image)
        return self.encoder.get_img()

    def close(self):
        """
        Close the camera.
        """
        self.cam.close_cam()


class WebSocketServer:
    """
    Implements the WebSocket service to send processed frames.
    Each client has its own frame processing loop.
    """

    def __init__(
        self,
        host: str,
        port: int,
        model_handler: YoloV5X,
        camera_manager: CameraManager,
        detection_classes,
    ):
        self.host = host
        self.port = port
        self.model_handler = model_handler
        self.camera_manager = camera_manager
        self.detection_classes = detection_classes

        # Track active connections
        self.active_connections = set()

    async def handler(self, websocket):
        """
        Handle a new WebSocket connection and process frames continuously.
        """
        print(
            "New connection from:",
            websocket.remote_address,
            "path:",
            getattr(websocket, "path", None),
        )

        self.active_connections.add(websocket)

        try:
            while True:
                # Create a new FrameMessage for transmission
                frame_msg = x3_pb2.FrameMessage()
                frame_msg.img_.height_ = 1072
                frame_msg.img_.width_ = 1920
                frame_msg.img_.type_ = "JPEG"

                # Capture resized image for model inference
                original_image = self.camera_manager.get_original_image()
                if original_image is None:
                    print("Warning: failed to get frame from camera")
                    await asyncio.sleep(0.01)
                    continue

                # Prepare input data
                try:
                    input_array = self.model_handler.pre_process(
                        original_image, frame_msg.img_.width_, frame_msg.img_.height_)
                except Exception as e:
                    print("Preprocess error:", e)
                    await asyncio.sleep(0.01)
                    continue

                # Run inference and measure time
                try:
                    outputs = self.model_handler.forward(input_array)
                except Exception as e:
                    print("Inference error:", e)
                    await asyncio.sleep(0.01)
                    continue

                # Run post-processing and measure its duration
                try:
                    t1 = time.time()
                    boxes, cls_ids, scores = self.model_handler.post_process(
                        outputs, frame_msg.img_.width_, frame_msg.img_.height_)
                    t2 = time.time()
                    print("Yolov5xPostProcess time:", t2 - t1)
                except Exception as e:
                    print("Postprocess error:", e)
                    await asyncio.sleep(0.01)
                    continue

                results = []
                for box, cls_id, score in zip(boxes, cls_ids, scores):
                    print(f"检测到 {len(boxes)} 个目标: ")

                    results.append({
                        "id": int(cls_id),
                        "name": self.detection_classes[cls_id],
                        "score": float(score),
                        "bbox": list(map(float, box))
                    })

                frame_msg.smart_msg_.timestamp_ = int(time.time())

                # Encode image
                try:
                    encoded_image = self.camera_manager.encode_image(original_image)
                    frame_msg.img_.buf_ = encoded_image
                except Exception as e:
                    print("Encode image error:", e)

                # Serialize results
                try:
                    serialized_buf = serialize_message(frame_msg, results, self.detection_classes)
                    await websocket.send(serialized_buf)
                except websockets.ConnectionClosedOK:
                    print("Connection closed cleanly:", websocket.remote_address)
                    break
                except websockets.ConnectionClosedError:
                    print("Connection closed with error:", websocket.remote_address)
                    break
                except Exception as e:
                    print("Send error:", e)
                    await asyncio.sleep(0.01)
        finally:
            self.active_connections.discard(websocket)
            print("Connection handler exited for:", websocket.remote_address)

    async def start(self):
        """
        Start the WebSocket server.
        """
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"WebSocket server started on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run indefinitely


def main():
    """
    @brief Main entry for starting YOLOv5X WebSocket detection service.
    @details This function parses arguments, initializes model and camera manager,
             sets scheduling params, and starts an asyncio-based WebSocket server.
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
                        help="BPU core indexes to run. Provide a list of integers (e.g., --bpu_cores 0 1). ")
    parser.add_argument('--label-file', type=str, default='coco_classes.names',
                        help='Path to Load Label File.')
    parser.add_argument('--nms-thres', type=float, default=0.45,
                        help='IoU threshold.')
    parser.add_argument('--score-thres', type=float, default=0.05,
                        help='confidence threshold.')
    opt = parser.parse_args()

    # Register signal handler for graceful termination
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize detection classes
    detection_classes = common.load_class_names(opt.label_file)

    # Initialize camera manager
    camera_manager = CameraManager()

    # Download model if missing
    if not os.path.exists(opt.model_path):
        print(f"file {opt.model_path} does not exist. download yolov5x model.")

    # Instantiate YOLOv5X object detector
    yolov5x = YoloV5X(opt)

    # Set priority and BPU core affinity
    yolov5x.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print model meta info (input/output, shape, quantization)
    common.print_model_info(yolov5x.model)

    # Start asynchronous WebSocket inference server
    server = WebSocketServer(
        "0.0.0.0",                      # Bind to all interfaces
        8080,                           # Listening port
        yolov5x,                        # YOLOv5X detector instance
        camera_manager,                 # Camera input manager
        detection_classes               # COCO label list
    )

    # Run asyncio event loop for server
    asyncio.run(server.start())


if __name__ == "__main__":
    main()

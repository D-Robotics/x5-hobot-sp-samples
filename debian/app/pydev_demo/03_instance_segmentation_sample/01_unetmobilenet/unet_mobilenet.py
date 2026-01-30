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
from typing import Optional, Dict

# Append parent directory to sys.path to enable local 'utils' module imports
sys.path.append(os.path.abspath("../.."))
import utils.preprocess_utils as pre_utils
import utils.postprocess_utils as post_utils
import utils.common_utils as common


class UnetMobileNet:
    """
    @brief Wrapper class for performing semantic segmentation using UNet with MobileNet backbone via HB_HBMRuntime.

    This class handles model loading, input preprocessing, inference, and visualization-based postprocessing.
    """

    def __init__(self, opt):
        """
        @brief Initialize the UNetMobileNet model with runtime parameters.

        @param opt (argparse.Namespace) Parsed arguments with:
            - model_path: str, path to quantized model
            - alpha_f: float, alpha blending factor
        """
        self.model = hbm_runtime.HB_HBMRuntime(opt.model_path)

        # Load model and extract runtime metadata
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]

        # Extract input resolution from first input
        self.input_H = self.input_shapes[self.input_names[0]][2] #1 change 20251211 By Gwen
        self.input_W = self.input_shapes[self.input_names[0]][3] #2 change 20251211 By Gwen

        # Runtime and visualization configuration
        self.alpha_f = opt.alpha_f  # blending factor for visualization
        self.resize_type = 0
        self.classes_num = 19

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """
        @brief Set BPU scheduling parameters such as priority and core assignment.

        @param priority (int, optional) Priority level (0–255).
        @param bpu_cores (list[int], optional) List of BPU core indices.
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
        @brief Preprocess input image to NV12 format required by the model.

        @param img (np.ndarray) Input image in BGR format.
        @return dict: Nested tensor dictionary: {model_name: {input_name: y/uv plane}}
        """
        # Resize and convert to NV12 format
        resize_img = pre_utils.resized_image(img, self.input_W, self.input_H, self.resize_type, interpolation=cv2.INTER_AREA)
        y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        nv12 = np.concatenate((y.reshape(-1), uv.reshape(-1)), axis=0)
        nv12 = nv12.reshape((1, self.input_H * 3 // 2, self.input_W, 1))
        
        return {
            self.model_name: {
                self.input_names[0]: nv12,
            }
        }

    def forward(self,
                input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        @brief Run inference on input tensor.

        @param input_tensor (dict) Prepared input tensor.
        @return dict: Output tensor dictionary {output_name: np.ndarray}
        """
        outputs = self.model.run(input_tensor)
        return outputs[self.model_name]

    def post_process(self,
                     model_output: Dict[str, np.ndarray],
                     origin_image: np.ndarray) -> np.ndarray:
        """
        @brief Postprocess segmentation output to obtain visualization overlay.

        @param model_output (dict) Output tensors from the model.
        @param origin_image (np.ndarray) Original input image in BGR format.
        @return np.ndarray: Image with segmentation mask blended on top.
        """
        img_h, img_w = origin_image.shape[:2]

        # Get logits from model output
        logits = model_output[self.output_names[0]][0]

        # Convert logits to class predictions
        pred_class = np.argmax(logits, axis=-1)  # shape: (H, W)

        # Resize to model input resolution
        seg_mask = cv2.resize(pred_class, (self.input_W, self.input_H), interpolation=cv2.INTER_NEAREST)

        # Recover to original image size
        ori_seg_mask = post_utils.recover_to_original_size(seg_mask, img_w, img_h, self.resize_type)

        # Map class ID to color
        palette_np = np.array(common.rdk_colors, dtype=np.uint8).reshape(-1, 3)

        # Clamp class indices to avoid out-of-range palette index
        seg_masked = np.where(ori_seg_mask < self.classes_num, ori_seg_mask, 0)

        # Convert class IDs to RGB color image
        parsing_img = palette_np[seg_masked]  # shape: (H, W, 3)

        # Blend segmentation result with original image
        blended_img = cv2.addWeighted(origin_image, self.alpha_f, parsing_img, 1 - self.alpha_f, 0.0)

        return blended_img


def main() -> None:
    """
    @brief Run semantic segmentation with UnetMobileNet and visualize the result.

    This function loads the input image, runs segmentation using HB_HBMRuntime,
    applies postprocessing to generate a blended overlay, and saves the result.

    @return None
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default='/app/model/basic/mobilenet_unet_1024x2048_nv12.bin',
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""")
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255). 0 is lowest, 255 is highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='List of BPU core indices to run, e.g., --bpu-cores 0 1')
    parser.add_argument('--test-img', type=str,
                        default='segmentation.png',
                        help='Path to input test image.')
    parser.add_argument('--img-save-path', type=str,
                        default='result.jpg',
                        help='Path to save the result image.')
    parser.add_argument('--alpha-f', type=float, default=0.75,
                        help='Alpha blending factor. 0.0 = only mask, 1.0 = only original image.')

    opt = parser.parse_args()

    # Download fallback: model missing
    if not os.path.exists(opt.model_path):
        print(f"File {opt.model_path} does not exist. Please download the UnetMobileNet model manually.")
        return

    # Instantiate model
    unet_mobilenet = UnetMobileNet(opt)

    # Set model execution priority and BPU core bindings
    unet_mobilenet.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print model I/O and metadata
    common.print_model_info(unet_mobilenet.model)

    # Load input image
    img: np.ndarray = common.load_image(opt.test_img)
    img_h, img_w = img.shape[:2]

    # Prepare model input
    input_array = unet_mobilenet.pre_process(img)

    # Run inference
    outputs = unet_mobilenet.forward(input_array)

    # Postprocess: get blended overlay image
    blended_img = unet_mobilenet.post_process(outputs, img)

    # Save result
    cv2.imwrite(opt.img_save_path, blended_img)
    print(f"[Saved] Result saved to: {opt.img_save_path}")


if __name__ == "__main__":
    main()

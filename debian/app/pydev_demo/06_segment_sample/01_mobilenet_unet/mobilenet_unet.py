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
import utils.common_utils as common
import utils.draw_utils as draw


class Mobilenet_unet:
    """
    @brief Mobilenet_unet lane detection wrapper using HB_HBMRuntime.

    This class includes preprocessing, inference, and postprocessing for
    lane segmentation tasks, including instance mask generation.
    """

    def __init__(self, opt: 'argparse.Namespace') -> None:
        """
        @brief Initialize the Mobilenet_unet model wrapper.

        @param opt (argparse.Namespace): Runtime and model parameters:
            - model_path (str): Path to the quantized BPU model (*.bin)
            - resize_type (int): Resize strategy (0=direct, 1=letterbox)
        """
        # Load BPU model
        self.model = hbm_runtime.HB_HBMRuntime(opt.model_path)

        # Get model metadata
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]

        # Model input resolution
        self.input_H = self.input_shapes[self.input_names[0]][2]
        self.input_W = self.input_shapes[self.input_names[0]][3]

        self.resize_type = 1

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list[int]] = None) -> None:
        """
        @brief Set scheduling parameters for HB runtime.

        @param priority (int, optional): Inference priority (0–255)
        @param bpu_cores (list[int], optional): List of BPU core indexes
        @return None
        """
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}

        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img: np.ndarray) -> np.ndarray:
        """
        @brief Preprocess input image for Mobilenet_unet.

        @param img (np.ndarray): Input BGR image
        @return np.ndarray: Preprocessed tensor, float32 normalized
        """
        resize_img = pre_utils.resized_image(img, self.input_W, self.input_H,resize_type=0,interpolation=cv2.INTER_NEAREST)
        y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        nv12 = np.concatenate((y.reshape(-1), uv.reshape(-1)), axis=0)
        nv12 = nv12.reshape((1, self.input_H * 3 // 2, self.input_W, 1))

        return nv12

    def forward(self, input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """
        @brief Run inference.

        @param input_tensor (np.ndarray): Input tensor of shape
        @return dict: Output tensors indexed by output name
        """
        outputs = self.model.run(input_tensor)
        return outputs[self.model_name]

    def post_process(self, outputs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        @brief Postprocess Mobilenet_unet outputs to generate visual masks.

        @param outputs (dict): Raw model outputs
        @return tuple:
            - pred_result (np.ndarray): Segment result
        """
        pred_result = np.argmax(outputs[self.output_names[0]], axis=-1)

        return pred_result


def main() -> None:
    """
    @brief Entry point for running LaneNet lane segmentation demo.

    This script loads a Mobilenet_unet model, performs image preprocessing,
    runs inference, and visualizes the instance segmentation results.

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
                        help='Model priority (0~255). 0 is lowest, 255 is highest. Default: 0.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='BPU core indexes to run (e.g., --bpu-cores 0 1).')
    parser.add_argument('--test-img', type=str, default='segmentation.png',
                        help='Path to test image.')
    parser.add_argument('--save-path', type=str, default='segmentation_result.png',
                        help='Path to test image.')

    opt = parser.parse_args()

    # Instantiate model
    Seg = Mobilenet_unet(opt)

    # Set inference scheduling options
    Seg.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print model information
    common.print_model_info(Seg.model)

    # Load input image
    img = common.load_image(opt.test_img)
    
    # Preprocess input
    input_array = Seg.pre_process(img)

    # Run inference
    outputs = Seg.forward(input_array)

    # Post-process outputs
    pred_result = Seg.post_process(outputs)
    draw.draw_seg_result(img, pred_result,opt.save_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

################################################################################
# Copyright (c) 2024,D-Robotics.
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
################################################################################

import os
import sys
import cv2

def is_usb_camera(device):
    try:
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            return False
        cap.release()
        return True
    except Exception:
        return False

def find_first_usb_camera():
    video_devices = [os.path.join('/dev', dev) for dev in os.listdir('/dev') if dev.startswith('video')]
    for dev in video_devices:
        if is_usb_camera(dev):
            return dev
    return None

if __name__ == '__main__':
    if len(sys.argv) > 1:
        video_device = sys.argv[1]
    else:
        video_device = find_first_usb_camera()

    if video_device is None:
        print("No USB camera found.")
        sys.exit(-1)

    print(f"Opening video device: {video_device}")
    cap = cv2.VideoCapture(video_device)

    if not cap.isOpened():
        print(f"Failed to open video device: {video_device}")
        sys.exit(-1)

    print("Open USB camera successfully")

    # 设置 USB camera 的输出图像格式为 MJPEG，分辨率 1920 x 1080
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv2.CAP_PROP_FOURCC, codec)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    ret, frame = cap.read()

    if not ret or frame is None:
        print("Failed to get image from USB camera")
        sys.exit(-1)

    cv2.imwrite("img.jpg", frame)
    print("Image saved as img.jpg")
    cap.release()


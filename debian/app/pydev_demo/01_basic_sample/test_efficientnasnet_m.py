'''
// Copyright (c) 2024,D-Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
'''

#!/usr/bin/env python3

from hobot_dnn import pyeasy_dnn as dnn
import numpy as np
import cv2

import time
import ctypes
import json

output_tensors = None

fcos_postprocess_info = None

class hbSysMem_t(ctypes.Structure):
    _fields_ = [
        ("phyAddr",ctypes.c_double),
        ("virAddr",ctypes.c_void_p),
        ("memSize",ctypes.c_int)
    ]

class hbDNNQuantiShift_yt(ctypes.Structure):
    _fields_ = [
        ("shiftLen",ctypes.c_int),
        ("shiftData",ctypes.c_char_p)
    ]

class hbDNNQuantiScale_t(ctypes.Structure):
    _fields_ = [
        ("scaleLen",ctypes.c_int),
        ("scaleData",ctypes.POINTER(ctypes.c_float)),
        ("zeroPointLen",ctypes.c_int),
        ("zeroPointData",ctypes.c_char_p)
    ]

class hbDNNTensorShape_t(ctypes.Structure):
    _fields_ = [
        ("dimensionSize",ctypes.c_int * 8),
        ("numDimensions",ctypes.c_int)
    ]

class hbDNNTensorProperties_t(ctypes.Structure):
    _fields_ = [
        ("validShape",hbDNNTensorShape_t),
        ("alignedShape",hbDNNTensorShape_t),
        ("tensorLayout",ctypes.c_int),
        ("tensorType",ctypes.c_int),
        ("shift",hbDNNQuantiShift_yt),
        ("scale",hbDNNQuantiScale_t),
        ("quantiType",ctypes.c_int),
        ("quantizeAxis", ctypes.c_int),
        ("alignedByteSize",ctypes.c_int),
        ("stride",ctypes.c_int * 8)
    ]

class hbDNNTensor_t(ctypes.Structure):
    _fields_ = [
        ("sysMem",hbSysMem_t * 4),
        ("properties",hbDNNTensorProperties_t)
    ]


class ClassificationPostProcessInfo_t(ctypes.Structure):
    _fields_ = [
        ("height",ctypes.c_int),
        ("width",ctypes.c_int),
        ("ori_height",ctypes.c_int),
        ("ori_width",ctypes.c_int),
        ("score_threshold",ctypes.c_float),
        ("nms_threshold",ctypes.c_float),
        ("nms_top_k",ctypes.c_int),
        ("is_pad_resize",ctypes.c_int)
    ]


libpostprocess = ctypes.CDLL('/usr/lib/libpostprocess.so')

get_Postprocess_result = libpostprocess.ClassificationPostProcess
get_Postprocess_result.argtypes = [ctypes.POINTER(ClassificationPostProcessInfo_t)]
get_Postprocess_result.restype = ctypes.c_char_p

def get_TensorLayout(Layout):
    if Layout == "NCHW":
        return int(2)
    else:
        return int(0)

def bgr2nv12_opencv(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12

def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)


# 对宽度进行16字节对齐
def align_16(value):
    return (value + 15) // 16 * 16

# 分配对齐后的内存，并填充图像数据
def bgr_to_nv12_custom_with_padding(bgr_image, aligned_width, aligned_height):
    height, width = bgr_image.shape[:2]

    # 分离 YUV 分量
    yuv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YUV_I420)

    # 创建对齐后的 Y 平面和 UV 平面
    y_plane = np.zeros((aligned_height, aligned_width), dtype=np.uint8)
    uv_plane = np.zeros((aligned_height // 2, aligned_width), dtype=np.uint8)

    # 提取 Y、U、V 分量
    y_orig = yuv_image[:height, :]
    u_orig = yuv_image[height:height + height // 4].reshape(height // 2, width // 2)
    v_orig = yuv_image[height + height // 4:].reshape(height // 2, width // 2)

    # 填充 Y 分量到对齐后的 Y 平面
    for i in range(height):
        y_plane[i, :width] = y_orig[i, :]

    # 填充 UV 分量到对齐后的 UV 平面 (交错 U 和 V)
    for i in range(height // 2):
        uv_plane[i, 0:width:2] = u_orig[i, :]  # 奇数列填充 U
        uv_plane[i, 1:width:2] = v_orig[i, :]  # 偶数列填充 V

    # 返回对齐后的 Y 和 UV 数据
    return y_plane, uv_plane

# 将 Y 和 UV 数据合并为 NV12 格式
def combine_yuv_to_nv12(y_data, uv_data):
    return np.concatenate((y_data.flatten(), uv_data.flatten()))

# 图像处理函数，传入模型的高、宽，ssd_mobilenetv1_300x300_nv12.bin使用的分辨率是 300x300
def process_image(img_file, models_h , models_w):

    # 使用固定的高和宽
    h, w = (models_h, models_w)
    des_dim = (w, h)

    # 调整图像大小到目标分辨率
    resized_data = cv2.resize(img_file, des_dim, interpolation=cv2.INTER_AREA)

    # 对齐后的宽高
    aligned_width = align_16(w)
    aligned_height = h

    # 使用对齐后的宽高进行 NV12 转换并填充
    y_data, uv_data = bgr_to_nv12_custom_with_padding(resized_data, aligned_width, aligned_height)

    return combine_yuv_to_nv12(y_data, uv_data)

def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]


if __name__ == '__main__':
    # test classification result
    models = dnn.load('../models/efficientnasnet_m_300x300_nv12.bin')
    # test input and output properties
    print("=" * 10, "inputs[0] properties", "=" * 10)
    print_properties(models[0].inputs[0].properties)
    print("inputs[0] name is:", models[0].inputs[0].name)

    print("=" * 10, "outputs[0] properties", "=" * 10)
    print_properties(models[0].outputs[0].properties)
    print("outputs[0] name is:", models[0].outputs[0].name)


    img_file = cv2.imread('./zebra_cls.jpg')
    h, w = get_hw(models[0].inputs[0].properties)
    des_dim = (w, h)
    # resized_data = cv2.resize(img_file, des_dim, interpolation=cv2.INTER_AREA)
    # nv12_data = bgr2nv12_opencv(resized_data)
    nv12_data = process_image(img_file, h, w)

    outputs = models[0].forward(nv12_data)

    t0 = time.time()
    # 获取结构体信息
    classification_postprocess_info = ClassificationPostProcessInfo_t()
    classification_postprocess_info.height = h
    classification_postprocess_info.width = w
    org_height, org_width = img_file.shape[0:2]
    classification_postprocess_info.ori_height = org_height
    classification_postprocess_info.ori_width = org_width
    classification_postprocess_info.score_threshold = 0.3
    classification_postprocess_info.nms_threshold = 0
    classification_postprocess_info.nms_top_k = 1
    classification_postprocess_info.is_pad_resize = 0

    output_tensors = (hbDNNTensor_t * len(models[0].outputs))()
    for i in range(len(models[0].outputs)):
        output_tensors[i].properties.tensorLayout = get_TensorLayout(outputs[i].properties.layout)
        # print(output_tensors[i].properties.tensorLayout)
        if (len(outputs[i].properties.scale_data) == 0):
            output_tensors[i].properties.quantiType = 0
            output_tensors[i].sysMem[0].virAddr = ctypes.cast(outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
        else:
            output_tensors[i].properties.quantiType = 2
            output_tensors[i].properties.scale.scaleData = outputs[i].properties.scale_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            output_tensors[i].sysMem[0].virAddr = ctypes.cast(outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)

        for j in range(len(outputs[i].properties.shape)):
            # output_tensors[i].properties.validShape.numDimensions = len(outputs[i].properties.shape)
            # output_tensors[i].properties.validShape.dimensionSize[j] = outputs[i].properties.shape[j]
            output_tensors[i].properties.validShape = ctypes.cast(outputs[i].properties.validShape, ctypes.POINTER(hbDNNTensorShape_t)).contents
            output_tensors[i].properties.alignedShape = ctypes.cast(outputs[i].properties.alignedShape, ctypes.POINTER(hbDNNTensorShape_t)).contents

        libpostprocess.ClassificationDoProcess(output_tensors[i], ctypes.pointer(classification_postprocess_info), i)

    result_str = get_Postprocess_result(ctypes.pointer(classification_postprocess_info))
    result_str = result_str.decode('utf-8')
    t1 = time.time()
    print("postprocess time is :", (t1 - t0))

    # draw result
    # 解析JSON字符串
    data = json.loads(result_str[25:])

    # 遍历每一个结果
    for result in data:
        prob = result['prob']  # 得分
        label = result['label']  # id
        name = result['class_name']  # 类别名称

        # 打印信息
        print(f"cls id: {label}, Confidence: {prob}, class_name: {name}")
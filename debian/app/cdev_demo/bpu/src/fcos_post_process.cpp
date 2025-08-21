// Copyright (c) 2024，D-Robotics.
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

// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cfloat>

#include "fcos_post_process.hpp"
float score_hold = 0.45;
float iou_threshold = 0.6;
int top_k = 500;
/**
 * Config definition for Fcos
 */

PTQFcosConfig fcos_config_ = {
    {{8, 16, 32, 64, 128}},
    80,
    {"person", "bicycle", "car",
     "motorcycle", "airplane", "bus",
     "train", "truck", "boat",
     "traffic light", "fire hydrant", "stop sign",
     "parking meter", "bench", "bird",
     "cat", "dog", "horse",
     "sheep", "cow", "elephant",
     "bear", "zebra", "giraffe",
     "backpack", "umbrella", "handbag",
     "tie", "suitcase", "frisbee",
     "skis", "snowboard", "sports ball",
     "kite", "baseball bat", "baseball glove",
     "skateboard", "surfboard", "tennis racket",
     "bottle", "wine glass", "cup",
     "fork", "knife", "spoon",
     "bowl", "banana", "apple",
     "sandwich", "orange", "broccoli",
     "carrot", "hot dog", "pizza",
     "donut", "cake", "chair",
     "couch", "potted plant", "bed",
     "dining table", "toilet", "tv",
     "laptop", "mouse", "remote",
     "keyboard", "cell phone", "microwave",
     "oven", "toaster", "sink",
     "refrigerator", "book", "clock",
     "vase", "scissors", "teddy bear",
     "hair drier", "toothbrush"},
    ""};

static int get_tensor_hwc_index(hbDNNTensor *tensor,
                                int *h_index,
                                int *w_index,
                                int *c_index)
{
  if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NHWC)
  {
    *h_index = 1;
    *w_index = 2;
    *c_index = 3;
  }
  else if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NCHW)
  {
    *c_index = 1;
    *h_index = 2;
    *w_index = 3;
  }
  else
  {
    return -1;
  }
  return 0;
}

static void fcos_nms(std::vector<Detection> &input,
                     float iou_threshold,
                     int top_k,
                     std::vector<Detection> &result,
                     bool suppress)
{
  // sort order by score desc
  std::stable_sort(input.begin(), input.end(), std::greater<Detection>());

  std::vector<bool> skip(input.size(), false);

  // pre-calculate boxes area
  std::vector<float> areas;
  areas.reserve(input.size());
  for (size_t i = 0; i < input.size(); i++)
  {
    float width = input[i].bbox.xmax - input[i].bbox.xmin;
    float height = input[i].bbox.ymax - input[i].bbox.ymin;
    areas.push_back(width * height);
  }

  int count = 0;
  for (size_t i = 0; count < top_k && i < skip.size(); i++)
  {
    if (skip[i])
    {
      continue;
    }
    skip[i] = true;
    ++count;

    for (size_t j = i + 1; j < skip.size(); ++j)
    {
      if (skip[j])
      {
        continue;
      }
      if (suppress == false)
      {
        if (input[i].id != input[j].id)
        {
          continue;
        }
      }

      // intersection area
      float xx1 = std::max(input[i].bbox.xmin, input[j].bbox.xmin);
      float yy1 = std::max(input[i].bbox.ymin, input[j].bbox.ymin);
      float xx2 = std::min(input[i].bbox.xmax, input[j].bbox.xmax);
      float yy2 = std::min(input[i].bbox.ymax, input[j].bbox.ymax);

      if (xx2 > xx1 && yy2 > yy1)
      {
        float area_intersection = (xx2 - xx1) * (yy2 - yy1);
        float iou_ratio =
            area_intersection / (areas[j] + areas[i] - area_intersection);
        if (iou_ratio > iou_threshold)
        {
          skip[j] = true;
        }
      }
    }
    result.push_back(input[i]);
  }
}

static void GetBboxAndScoresNHWC(
    hbDNNTensor *tensors,
    bpu_image_info_t *post_info,
    std::vector<Detection> &dets)
{
  int ori_h = post_info->m_ori_height;
  int ori_w = post_info->m_ori_width;
  int input_h = post_info->m_model_h;
  int input_w = post_info->m_model_w;
  float w_scale;
  float h_scale;
  // preprocess action is pad and resize
  w_scale = static_cast<float>(ori_w) / input_w;
  h_scale = static_cast<float>(ori_h) / input_h;

  // fcos stride is {8, 16, 32, 64, 128}
  for (int i = 0; i < 5; i++)
  {
    auto *cls_data = reinterpret_cast<float *>(tensors[i].sysMem[0].virAddr);
    auto *bbox_data =
        reinterpret_cast<float *>(tensors[i + 5].sysMem[0].virAddr);
    auto *ce_data =
        reinterpret_cast<float *>(tensors[i + 10].sysMem[0].virAddr);

    // 同一个尺度下，tensor[i],tensor[i+5],tensor[i+10]出来的hw都一致，64*64/32*32/...
    int *shape = tensors[i].properties.alignedShape.dimensionSize;
    int tensor_h = shape[1];
    int tensor_w = shape[2];
    int tensor_c = shape[3];

    for (int h = 0; h < tensor_h; h++)
    {
      int offset = h * tensor_w;
      for (int w = 0; w < tensor_w; w++)
      {
        // get score
        int ce_offset = offset + w;
        ce_data[ce_offset] = 1.0 / (1.0 + exp(-ce_data[ce_offset]));

        int cls_offset = ce_offset * tensor_c;
        ScoreId tmp_score = {cls_data[cls_offset], 0};
        for (int cls_c = 1; cls_c < tensor_c; cls_c++)
        {
          int cls_index = cls_offset + cls_c;
          if (cls_data[cls_index] > tmp_score.score)
          {
            tmp_score.id = cls_c;
            tmp_score.score = cls_data[cls_index];
          }
        }
        tmp_score.score = 1.0 / (1.0 + exp(-tmp_score.score));
        tmp_score.score = std::sqrt(tmp_score.score * ce_data[ce_offset]);
        if (tmp_score.score <= score_hold)
          continue;

        // get detection box
        Detection detection;
        int index = 4 * (h * tensor_w + w);
        auto &strides = fcos_config_.strides;

        detection.bbox.xmin =
            ((w + 0.5) * strides[i] - bbox_data[index]) * w_scale;
        detection.bbox.ymin =
            ((h + 0.5) * strides[i] - bbox_data[index + 1]) * h_scale;
        detection.bbox.xmax =
            ((w + 0.5) * strides[i] + bbox_data[index + 2]) * w_scale;
        detection.bbox.ymax =
            ((h + 0.5) * strides[i] + bbox_data[index + 3]) * h_scale;

        detection.score = tmp_score.score;
        detection.id = tmp_score.id;
        detection.class_name = fcos_config_.class_names[detection.id].c_str();
        dets.push_back(detection);
      }
    }
  }
}

static void GetBboxAndScoresNCHW(
    hbDNNTensor *tensors,
    bpu_image_info_t *post_info,
    std::vector<Detection> &dets)
{
  int ori_h = post_info->m_ori_height;
  int ori_w = post_info->m_ori_width;
  int input_h = post_info->m_model_h;
  int input_w = post_info->m_model_w;
  float w_scale;
  float h_scale;
  // preprocess action is pad and resize
  w_scale = static_cast<float>(ori_w) / input_w;
  h_scale = static_cast<float>(ori_h) / input_h;

  for (int i = 0; i < 5; i++)
  {
    auto *cls_data = reinterpret_cast<float *>(tensors[i].sysMem[0].virAddr);
    auto *bbox_data =
        reinterpret_cast<float *>(tensors[i + 5].sysMem[0].virAddr);
    auto *ce_data =
        reinterpret_cast<float *>(tensors[i + 10].sysMem[0].virAddr);

    // 同一个尺度下，tensor[i],tensor[i+5],tensor[i+10]出来的hw都一致，64*64/32*32/...
    int *shape = tensors[i].properties.alignedShape.dimensionSize;
    int tensor_c = shape[1];
    int tensor_h = shape[2];
    int tensor_w = shape[3];
    int aligned_hw = tensor_h * tensor_w;

    for (int h = 0; h < tensor_h; h++)
    {
      int offset = h * tensor_w;
      for (int w = 0; w < tensor_w; w++)
      {
        // get score
        int ce_offset = offset + w;
        ce_data[ce_offset] = 1.0 / (1.0 + exp(-ce_data[ce_offset]));

        ScoreId tmp_score = {cls_data[offset + w], 0};
        for (int cls_c = 1; cls_c < tensor_c; cls_c++)
        {
          int cls_index = cls_c * aligned_hw + offset + w;
          if (cls_data[cls_index] > tmp_score.score)
          {
            tmp_score.id = cls_c;
            tmp_score.score = cls_data[cls_index];
          }
        }
        tmp_score.score = 1.0 / (1.0 + exp(-tmp_score.score));
        tmp_score.score = std::sqrt(tmp_score.score * ce_data[ce_offset]);
        if (tmp_score.score <= score_hold)
          continue;

        // get detection box
        auto &strides = fcos_config_.strides;
        Detection detection;
        detection.bbox.xmin =
            ((w + 0.5) * strides[i] - bbox_data[offset + w]) * w_scale;
        detection.bbox.ymin =
            ((h + 0.5) * strides[i] - bbox_data[1 * aligned_hw + offset + w]) *
            h_scale;
        detection.bbox.xmax =
            ((w + 0.5) * strides[i] + bbox_data[2 * aligned_hw + offset + w]) *
            w_scale;
        detection.bbox.ymax =
            ((h + 0.5) * strides[i] + bbox_data[3 * aligned_hw + offset + w]) *
            h_scale;

        detection.score = tmp_score.score;
        detection.id = tmp_score.id;
        detection.class_name = fcos_config_.class_names[detection.id].c_str();
        dets.push_back(detection);
      }
    }
  }
}

static void GetBboxAndScoresScaleNHWC_V2(
  hbDNNTensor *tensors,
  bpu_image_info_t *post_info,
  std::vector<Detection> &dets)
{
  int ori_h = post_info->m_ori_height;
  int ori_w = post_info->m_ori_width;
  int input_h = post_info->m_model_h;
  int input_w = post_info->m_model_w;
  float w_scale = static_cast<float>(ori_w) / input_w;
  float h_scale = static_cast<float>(ori_h) / input_h;

  // 保存所有原始输出数据
  std::vector<std::vector<int32_t>> all_cls_data;
  std::vector<std::vector<int32_t>> all_bbox_data;
  std::vector<std::vector<int32_t>> all_ce_data;

  for (int i = 0; i < 5; i++)  // 遍历5个特征层
  {

      hbDNNTensor *cls_tensor = &tensors[i];
      hbDNNTensor *bbox_tensor = &tensors[i + 5];
      hbDNNTensor *ce_tensor = &tensors[i + 10];

      // 添加张量有效性检查
      if (!cls_tensor || !bbox_tensor || !ce_tensor) {
          printf("Error: Null tensor pointer at i=%d\n", i);
          fflush(stdout);
          continue;
      }

      // 获取量化参数
      float *cls_scale = cls_tensor->properties.scale.scaleData;
      float *bbox_scale = bbox_tensor->properties.scale.scaleData;
      float *ce_scale = ce_tensor->properties.scale.scaleData;
      int *shape = cls_tensor->properties.alignedShape.dimensionSize;
      int tensor_h = shape[1];
      int tensor_w = shape[2];
      int tensor_c = shape[3];
      int stride = fcos_config_.strides[i];
      int32_t bbox_c_stride = bbox_tensor->properties.alignedShape.dimensionSize[3];
      int32_t ce_c_stride = ce_tensor->properties.alignedShape.dimensionSize[3];

      // 获取数据指针
      int32_t *cls_data = reinterpret_cast<int32_t*>(cls_tensor->sysMem[0].virAddr);
      int32_t *bbox_data = reinterpret_cast<int32_t*>(bbox_tensor->sysMem[0].virAddr);
      int32_t *ce_data = reinterpret_cast<int32_t*>(ce_tensor->sysMem[0].virAddr);

      // 计算总元素数
      size_t cls_size = tensor_h * tensor_w * tensor_c;
      size_t bbox_size = tensor_h * tensor_w * 4;
      size_t ce_size = tensor_h * tensor_w;

      // 保存当前层原始数据
      std::vector<int32_t> cls_layer_data(cls_data, cls_data + cls_size);
      std::vector<int32_t> bbox_layer_data(bbox_data, bbox_data + bbox_size);
      std::vector<int32_t> ce_layer_data(ce_data, ce_data + ce_size);

      all_cls_data.push_back(cls_layer_data);
      all_bbox_data.push_back(bbox_layer_data);
      all_ce_data.push_back(ce_layer_data);

      // 完整处理所有点
      for (int h = 0; h < tensor_h; h++)
      {
          for (int w = 0; w < tensor_w; w++)
          {

              int spatial_offset = h * tensor_w + w;

              // 1. 中心度分数
              float ce_val = ce_data[spatial_offset] * ce_scale[0];
              ce_val = 1.0f / (1.0f + std::exp(-ce_val));

              // 2. 分类分数
              int cls_offset = spatial_offset * tensor_c;
              float max_score = -FLT_MAX;
              int max_id = -1;

              for (int c = 0; c < tensor_c; c++) {
                  float cls_val = cls_data[cls_offset + c] * cls_scale[c];
                  if (cls_val > max_score) {
                      max_score = cls_val;
                      max_id = c;
                  }
              }

              float cls_score = 1.0f / (1.0f + std::exp(-max_score));
              float final_score = std::sqrt(cls_score * ce_val);

              if (final_score <= score_hold) {
                  continue;
              }

              Detection detection;
              int index = bbox_c_stride * (h * tensor_w + w);

              // 检查索引是否越界
              if (index + 3 >= bbox_size) {
                  // printf("Error: bbox index out of bounds at i=%d, h=%d, w=%d, index=%d, bbox_size=%zu\n",
                  //        i, h, w, index, bbox_size);
                  // fflush(stdout);
                  continue;
              }

              auto &strides = fcos_config_.strides;

              // 修正边界框计算 - 确保不会出现负值
              float xmin = bbox_data[index] * bbox_scale[0];
              float ymin = bbox_data[index + 1] * bbox_scale[1];
              float xmax = bbox_data[index + 2] * bbox_scale[2];
              float ymax = bbox_data[index + 3] * bbox_scale[3];

              // 确保偏移量非负
              xmin = std::max(0.0f, xmin);
              ymin = std::max(0.0f, ymin);
              xmax = std::max(0.0f, xmax);
              ymax = std::max(0.0f, ymax);

              detection.bbox.xmin = ((w + 0.5) - xmin) * strides[i] * w_scale;
              detection.bbox.ymin = ((h + 0.5) - ymin) * strides[i] * h_scale;
              detection.bbox.xmax = ((w + 0.5) + xmax) * strides[i] * w_scale;
              detection.bbox.ymax = ((h + 0.5) + ymax) * strides[i] * h_scale;

              detection.score = final_score;
              detection.id = max_id;

              // 添加类别ID范围检查
              if (detection.id < 0 || detection.id >= fcos_config_.class_names.size()) {
                  printf("Warning: Invalid class id %d at i=%d, h=%d, w=%d\n",
                         detection.id, i, h, w);
                  fflush(stdout);
                  continue;
              }

              detection.class_name = fcos_config_.class_names[detection.id].c_str();

              dets.push_back(detection);

          }
      }
  }
}

void fcos_post_process(hbDNNTensor* tensors, bpu_image_info_t *post_info, std::vector<Detection> &det_restuls)
{

  std::vector<Detection> dets;

  int h_index, w_index, c_index;
  int ret = get_tensor_hwc_index(&tensors[0], &h_index, &w_index, &c_index);
  if (ret)
  {
    printf(" %s [ERROR]:Invalid tensor,please check your model!\n", __FUNCTION__);
    return ;
  }

  auto quanti_type = tensors->properties.quantiType;
  if(quanti_type == hbDNNQuantiType::SCALE)
  {
    float *cls_scale = tensors->properties.scale.scaleData;
    GetBboxAndScoresScaleNHWC_V2(tensors , post_info , dets);
  } else if(quanti_type == hbDNNQuantiType::NONE){
    if (tensors[0].properties.tensorLayout == HB_DNN_LAYOUT_NHWC)
    {
      GetBboxAndScoresNHWC(tensors, post_info, dets);
    }
    else if (tensors[0].properties.tensorLayout == HB_DNN_LAYOUT_NCHW)
    {
      GetBboxAndScoresNCHW(tensors, post_info, dets);
    }
  }


  else
  {
    printf("tensor layout error.\n");
    return ;
  }
  // 计算交并比来合并检测框，传入交并比阈值和返回box数量
  fcos_nms(dets, iou_threshold, top_k, det_restuls, false);
}

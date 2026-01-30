# Ultralytics YOLOv8 目标检测示例

本示例基于 Ultralytics YOLOv8 模型，通过 `HB_HBMRuntime` 接口完成图像的目标检测。支持图像预处理、推理、后处理（包含解码、置信度过滤、NMS）以及结果图像保存。

## 环境依赖
本样例无特殊环境需求，只需确保安装了pydev中的环境依赖即可。
```bash
pip install -r ../../requirements.txt
```

## 目录结构
```text
.
├── ultralytics_yolov8.py      # 主推理脚本
└── README.md                   # 使用说明
```

## 参数说明

| 参数           | 说明                                                     | 默认值                                      |
|----------------|----------------------------------------------------------|---------------------------------------------|
| `--model-path` | 模型文件路径（.bin 格式）                                  | `/opt/hobot/model/x5/basic/yolov8_640x640_nv12.bin` |
| `--test-img`   | 测试图片路径                                              | `kite.jpg`                     |
| `--label-file` | 类别标签路径（每行一个类别）                                | `coco_classes.names`           |
| `--img-save-path` | 检测结果图像保存路径                                    | `result.jpg`                                |
| `--priority`  | 模型调度优先级（0~255）                                     | `0`                                         |
| `--bpu-cores` | 使用的 BPU 核心编号列表（如 `--bpu-cores 0 1`）              | `[0]`                                      |
| `--nms-thres`   | 非极大值抑制（NMS）阈值                                    | `0.45`                                    |
| `--score-thres` | 置信度阈值                                                | `0.25`                                    |


## 快速运行
- 运行模型
    - 使用默认参数
        ```bash
        python ultralytics_yolov8.py
        ```
    - 指定参数运行
        ```bash
        python ultralytics_yolov8.py \
            --model-path /opt/hobot/model/x5/basic/yolov8_640x640_nv12.bin \
            --test-img kite.jpg \
            --label-file coco_classes.names \
            --img-save-path result.jpg \
            --priority 0 \
            --bpu-cores 0 \
            --nms-thres 0.45 \
            --score-thres 0.25
        ```
- 查看结果

    运行成功后，会将目标检测框绘制在原图上，并保存到 --img-save-path 指定路径
    ```bash
    [Saved] Result saved to: result.jpg
    ```

## 注意事项
- 若指定模型路径不存在，可尝试去`/opt/hobot/model/x5/basic/`查找。

- 输出结果存储为result.jpg，用户可自行查看。

- 如需了解更多部署方式或模型支持情况，请参考官方文档或联系平台技术支持。

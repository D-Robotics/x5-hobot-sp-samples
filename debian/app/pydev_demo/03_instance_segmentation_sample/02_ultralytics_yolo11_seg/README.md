# Ultralytics YOLO11 语义分割示例

本示例展示了如何基于 `HB_HBMRuntime` 在 BPU 上运行 Ultralytics YOLO11 语义分割模型，支持图像预处理、推理、后处理（解析输出并叠加彩色分割掩码）等功能。

## 环境依赖
本样例无特殊环境需求，只需确保安装了pydev中的环境依赖即可。
```bash
pip install -r ../../requirements.txt
```

## 目录结构
```text
.
├── ultralytics_yolo11_seg.py   # 主推理脚本
└── README.md                   # 使用说明
```



## 参数说明
| 参数                | 说明                    | 默认值 |
|--------------------|-----------------------------|--------------------------------------|
| `--model-path`     | 模型文件路径（.bin 格式）     | `yolo11n_seg_bayese_640x640_nv12.bin` |
| `--test-img`       | 测试图片路径                 | `bus.jpg`        |
| `--label-file`     | 分类标签文件                 | `coco_classes.names`     |
| `--img-save-path`  | 输出结果图片保存路径          | `result.jpg`                          |
| `--priority`       | 模型优先级 (0~255)           | `0`                                   |
| `--bpu-cores`      | BPU 核心编号                 | `[0]`                                 |
| `--nms-thres`      | NMS IoU 队值间值             | `0.7`                                 |
| `--score-thres`    | 精度阈值                     | `0.25`                                |
| `--is-open`        | 是否对分割结果进行形态形象处理 | `True`                                |
| `--is-point`       | 是否在边缘处绘制边线上的点     | `True`                                |

## 快速运行
- 运行模型
    - 使用默认参数
        ```bash
        python ultralytics_yolo11_seg.py
        ```
    - 指定参数运行
        ```bash
        python ultralytics_yolo11_seg.py \
        --model-path yolo11n_seg_bayese_640x640_nv12.bin \
        --test-img bus.jpg \
        --label-file coco_classes.names \
        --img-save-path result.jpg \
        --priority 0 \
        --bpu-cores 0 \
        --nms-thres 0.7 \
        --score-thres 0.25 \
        --is-open True \
        --is-point True
        ```
- 查看结果

    运行成功后，会将结果绘制在原图上，并保存到 --img-save-path 指定路径
    ```bash
    [Saved] Result saved to: result.jpg
    ```

## 注意事项
- 若指定模型路径不存在，程序将尝试自动下载模型。

- 输出结果存储为result.jpg，用户可自行查看。

- 如需了解更多部署方式或模型支持情况，请参考官方文档或联系平台技术支持。

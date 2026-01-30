# Ultralytics YOLO11 姿态估计示例

本示例展示了如何基于 `HB_HBMRuntime` 在 BPU 上运行 Ultralytics YOLO11 姿态估计模型，实现人体关键点检测与可视化。支持模型预处理、推理执行与后处理（含关键点解码、边界框绘制、关键点标注）。

## 环境依赖
本样例无特殊环境需求，只需确保安装了pydev中的环境依赖即可。
```bash
pip install -r ../../requirements.txt
```

## 目录结构
```text
.
├── ultralytics_yolo11_pose.py   # 主推理脚本
└── README.md                    # 使用说明
```

## 参数说明
| 参数名                | 说明                                              | 默认值                                |
| ------------------ | --------------------------------------------------- | ------------------------------------- |
| `--model-path`     | 模型文件路径（`.bin` 格式）                           | `yolo11n_pose_bayese_640x640_nv12.bin` |
| `--test-img`       | 测试图像路径                                         | `bus.jpg`                |
| `--label-file`     | 类别标签路径，每行一个类别名称                         | `coco_classes.names`     |
| `--img-save-path`  | 检测结果保存路径                                     | `result.jpg`                          |
| `--priority`       | 模型调度优先级（0\~255，数值越大优先级越高）           | `0`                                   |
| `--bpu-cores`      | 推理所使用的 BPU 核心编号列表（如：`--bpu-cores 0 1`） | `[0]`                                 |
| `--nms-thres`      | 非极大值抑制（NMS）中的 IoU 阈值                      | `0.7`                                 |
| `--score-thres`    | 目标置信度阈值（低于该值的目标将被过滤）               | `0.25`                                |
| `--kpt-conf-thres` | 关键点可视化置信度阈值（低于该值的关键点将不显示）      | `0.5`                                 |

## 快速运行
- 运行模型
    - 使用默认参数
        ```bash
        python ultralytics_yolo11_pose.py
        ```
    - 指定参数运行
        ```bash
        python ultralytics_yolo11_pose.py \
        --model-path yolo11n_pose_bayese_640x640_nv12.bin \
        --test-img bus.jpg \
        --label-file coco_classes.names \
        --img-save-path result.jpg \
        --priority 0 \
        --bpu-cores 0 \
        --score-thres 0.25 \
        --nms-thres 0.7 \
        --kpt-conf-thres 0.5
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

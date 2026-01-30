# YOLOv5X USB Camera Inference 推理示例

基于 HB_HBMRuntime 的 YOLOv5X 实时推理示例，支持通过 USB 摄像头读取画面并进行目标检测，并以全屏方式可视化检测结果。

## 环境依赖
- 确保安装了pydev中的环境依赖
    ```bash
    pip install -r ../requirements.txt
    ```

## 目录结构
```text
.
├── usb_camera_yolov5x.py       # 主程序
└── README.md                   # 使用说明
```

## 参数说明
| 参数名           | 说明                              | 默认值                                                    |
| --------------- | --------------------------------- | ------------------------------------------------------ |
| `--model-path`  | BPU 量化模型路径（`.bin`）          | `/app/model/basic/yolov5x_672x672_nv12.bin` |
| `--priority`    | 推理优先级（0\~255，255为最高）     | `0`                                                    |
| `--bpu-cores`   | BPU 核心索引列表（如 `0 1`）        | `[0]`                                                  |
| `--label-file`  | 类别标签文件路径                    | `coco_classes.names`                         |
| `--nms-thres`   | 非极大值抑制的 IoU 阈值             | `0.45`                                                 |
| `--score-thres` | 检测置信度阈值                      | `0.25`                                                 |


## 快速运行
注意：该程序需运行在桌面环境。
- 运行模型
    - 使用默认参数
        ```bash
        python usb_camera_yolov5.py
        ```
    - 指定参数运行
        ```bash
        python usb_camera_yolov5x.py \
        --model-path /app/model/basic/yolov5x_672x672_nv12.bin \
        --priority 0 \
        --bpu-cores 0 \
        --label-file coco_classes.names \
        --nms-thres 0.45 \
        --score-thres 0.25
        ```
- 退出运行

    将鼠标放置在显示框内，按Q键退出

- 查看结果

    运行成功后，屏幕会实时显示目标检测图像

## 注意事项
- 该程序需运行在桌面环境。

- 若指定模型路径不存在，可尝试去`/app/model/basic`查找。

- 如需了解更多部署方式或模型支持情况，请参考官方文档或联系平台技术支持。

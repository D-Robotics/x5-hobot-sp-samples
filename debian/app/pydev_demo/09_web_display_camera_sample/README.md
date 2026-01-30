# YOLOv5X WebSocket 推理示例

本示例展示了如何在含有 HBM 加速器和 VIO 摄像头模块的嵌入式平台（如 RDK X5）上，使用 YOLOv5X 模型进行目标检测，并通过 WebSocket 实时推送 JPEG 图像和检测框。

## 环境依赖
- 确保安装了pydev中的环境依赖
    ```bash
    pip install -r ../requirements.txt
    ```
- 安装WebSocket的包
    ```bash
    pip install websockets==15.0.1 protobuf==3.20.3
    ```

## 硬件环境
- mipi camera的接口使用的自动检测模式，该sample运行时只能接入一个mipi摄像头（任意mipi接口都可以），同时接入多个会报错。

## 目录结构
```text
.
├── mipi_camera_web_yolov5x.py      # 主程序
└── README.md                       # 使用说明
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
- 启动服务
    ```bash
    # 1. 进入webservice目录
    cd webservice/

    # 2. 启动服务
    sudo ./sbin/nginx -p .
    ```
- 运行模型
    - 回到当前目录
        ```bash
        cd ..
        ```
    - 使用默认参数
        ```bash
        python mipi_camera_web_yolov5x.py
        ```
    - 指定参数运行
        ```bash
        python mipi_camera_web_yolov5x.py \
        --model-path /app/model/basic/yolov5x_672x672_nv12.bin \
        --priority 0 \
        --bpu-cores 0 \
        --label-file coco_classes.names \
        --nms-thres 0.45 \
        --score-thres 0.25
        ```

- 查看结果

    运行成功后，通过访问web展示端：http://IP

- 退出运行

    在命令行输入Ctrl C

## 注意事项
- 若指定模型路径不存在，可尝试去`/app/model/basic`查找。

- 如需了解更多部署方式或模型支持情况，请参考官方文档或联系平台技术支持。

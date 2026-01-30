# YOLOv5X MIPI Camera Inference 推理示例

本目录包含基于 MIPI 摄像头的多个 Python 示例，支持图像采集、缩放、裁剪、推理等功能；
## 环境依赖
- 确保安装了pydev中的环境依赖
    ```bash
    pip install -r ../requirements.txt
    ```

## 硬件环境
- mipi camera的接口使用的自动检测模式，该sample运行时只能接入一个mipi摄像头（任意mipi接口都可以），同时接入多个会报错。

## 目录结构
```text
.
├── 01_mipi_camera_yolov5x.py       # 使用 YOLOv5X 模型进行摄像头实时目标检测与显示
├── 02_mipi_camera_dump.py          # 将摄像头捕获的图像帧以 YUV 格式保存为文件
├── 03_mipi_camera_scale.py         # 对本地 YUV 图像进行分辨率缩放处理
├── 04_mipi_camera_crop_scale.py    # 对本地 YUV 图像裁剪并缩放处理
├── 05_mipi_camera_streamer.py      # 将摄像头图像实时显示至 HDMI 屏幕（推流测试）
└── README.md                       # 当前文件，包含脚本功能说明、参数介绍及使用方法
```

### 各示例说明

#### 01_mipi_camera_yolov5x.py

##### 功能简介

基于 HB_HBMRuntime 的 YOLOv5X 实时推理示例，支持通过 MIPI 摄像头读取画面并进行目标检测，并以全屏方式可视化检测结果。


##### 参数说明
| 参数名           | 说明                              | 默认值                                                    |
| --------------- | --------------------------------- | ------------------------------------------------------ |
| `--model-path`  | BPU 量化模型路径（`.bin`）          | `/app/model/basic/yolov5x_672x672_nv12.bin` |
| `--priority`    | 推理优先级（0\~255，255为最高）     | `0`                                                    |
| `--bpu-cores`   | BPU 核心索引列表（如 `0 1`）        | `[0]`                                                  |
| `--label-file`  | 类别标签文件路径                    | `coco_classes.names`                         |
| `--nms-thres`   | 非极大值抑制的 IoU 阈值             | `0.45`                                                 |
| `--score-thres` | 检测置信度阈值                      | `0.25`                                                 |


##### 快速运行
注意：该程序需运行在桌面环境。
- 运行模型
    - 使用默认参数
        ```bash
        python 01_mipi_camera_yolov5x.py
        ```
    - 指定参数运行
        ```bash
        python 01_mipi_camera_yolov5x.py \
        --model-path /app/model/basic/yolov5x_672x672_nv12.bin \
        --priority 0 \
        --bpu-cores 0 \
        --label-file coco_classes.names \
        --nms-thres 0.45 \
        --score-thres 0.25
        ```
- 退出运行

    在命令行输入Ctrl C

- 查看结果

    运行成功后，屏幕会实时显示目标检测图像

##### 注意事项
- 该程序需运行在桌面环境。

- 若指定模型路径不存在，可尝试去`/app/model/basic/`查找。

- 如需了解更多部署方式或模型支持情况，请参考官方文档或联系平台技术支持。

#### 02_mipi_camera_dump.py

##### 功能简介

初始化 MIPI 摄像头
- 设置采集帧率、分辨率、帧数
- 持续抓拍指定数量的图像帧并保存为 `YUV` 文件（raw dump）

##### 参数说明

| 参数 | 含义             | 类型   | 示例       |
|------|------------------|--------|------------|
| `-f` | 帧率（FPS）       | int    | `-f 30`    |
| `-c` | 采集帧数（count） | int    | `-c 10`    |
| `-w` | 图像宽度         | int    | `-w 1920`  |
| `-h` | 图像高度         | int    | `-h 1080`  |

##### 快速运行
- 运行

    ```bash
    python 02_mipi_camera_dump.py -f 30 -c 10 -w 1920 -h 1080
    ```

- 查看结果

    运行成功后，脚本所在目录会存放多个yuv文件。

#### 03_mipi_camera_scale.py

##### 功能简介

- 支持输入 YUV 数据（通常为 NV12 格式）
- 支持设置输入和输出图像的分辨率
- 使用硬件 VPS 完成图像缩放
- 将缩放后的图像保存为新的 YUV 文件

##### 参数说明

| 参数             | 说明                                   | 示例                |
|------------------|---------------------------------------|---------------------|
| `-i`             | 输入文件路径                           | `-i input.yuv`      |
| `-o`             | 输出文件路径（默认 `output_scale.yuv`） | `-o result.yuv`     |
| `-w`             | 输出图像宽度                           | `-w 640`            |
| `-h`             | 输出图像高度                           | `-h 360`            |
| `--iwidth`       | 输入图像宽度                           | `--iwidth 1920`     |
| `--iheight`      | 输入图像高度                           | `--iheight 1080`    |

##### 快速运行
- 准备一个原始 YUV 图像（NV12 格式）作为输入，例如：`input.yuv`
- 执行缩放脚本：

    ```bash
    python 03_mipi_camera_scale.py -i input.yuv -o output_640x360.yuv -w 640 -h 360 --iwidth 1920 --iheight 1080
    ```

- 查看结果

    运行成功后，脚本所在目录会存放缩放后的yuv文件。

#### 04_mipi_camera_crop_scale.py

##### 功能简介

- 支持从输入图像中裁剪指定区域
- 将裁剪区域缩放为目标分辨率
- 使用硬件 VPS 处理图像，效率高、速度快
- 将结果保存为 YUV 文件（NV12 格式）

##### 参数说明

| 参数            | 说明                                       | 示例                   |
|----------------|--------------------------------------------|------------------------|
| `-i`           | 输入 YUV 文件路径（NV12 格式）               | `-i input_1080p.yuv`   |
| `-o`           | 输出文件路径（默认 `output_crop_scale.yuv`） | `-o result.yuv`        |
| `-w`           | 输出图像宽度                                | `-w 640`               |
| `-h`           | 输出图像高度                                | `-h 480`               |
| `--iwidth`     | 输入图像原始宽度                             | `--iwidth 1920`        |
| `--iheight`    | 输入图像原始高度                             | `--iheight 1080`       |
| `-x`           | 裁剪区域左上角 X 坐标                        | `-x 300`               |
| `-y`           | 裁剪区域左上角 Y 坐标                        | `-y 300`               |
| `--crop_w`     | 裁剪区域宽度                                | `--crop_w 900`         |
| `--crop_h`     | 裁剪区域高度                                | `--crop_h 600`         |


##### 快速运行
- 准备一个原始 YUV 图像（NV12 格式）作为输入，例如：`input.yuv`
- 执行缩放脚本：

    ```bash
       python 04_mipi_camera_crop_scale.py \
        -i input.yuv -o output_640x480.yuv \
        -w 640 -h 480 --iwidth 1920 --iheight 1080 \
        -x 304 -y 304 --crop_w 896 --crop_h 592
    ```

- 查看结果

    运行成功后，脚本所在目录会存放缩放后的yuv文件。

##### 注意事项
- 裁剪宽度必须是 16 的整数倍（即对齐到 16 字节）

#### 05_mipi_camera_streamer.py

##### 功能简介

该脚本通过 hobot_vio.libsrcampy 接口采集 MIPI 摄像头图像，并将图像通过 HDMI 实时回显至屏幕，用于测试摄像头与显示通路的连通性。

##### 参数说明

| 参数            | 说明                                       | 示例                   |
|----------------|--------------------------------------------|------------------------|
| `-w`           | 输出图像宽度                                | `-w 1920`               |
| `-h`           | 输出图像高度                                | `-h 1080`               |


##### 快速运行
注意：该程序需运行在桌面环境。

- 执行缩放脚本：

    ```bash
    python 05_mipi_camera_streamer.py -w 1920 -h 1080
    ```

- 查看结果

    屏幕会显示实时画面；

# Mobilenet_unet 路面分割检测示例

本示例基于 HB_HBMRuntime 运行 Mobilenet_unet 模型，实现人、车辆、路面、路标等类别分割，并将结果图像保存到本地。

## 环境依赖
本样例无特殊环境需求，只需确保安装了pydev中的环境依赖即可。
```bash
pip install -r ../../requirements.txt
```

## 目录结构
```text
.
├── mobilenet_unet.py                  # 主推理脚本
└── README.md                   # 使用说明
```

## 参数说明
| 参数名            | 说明                                      | 默认值                      |
| -------------- | ------------------------------------------- | ------------------------ |
| `--model-path` | 模型文件路径，`.bin` 格式                     | `/app/model/basic/mobilenet_unet_1024x2048_nv12.bin`     |
| `--priority`   | 模型运行优先级，范围 0\~255，数值越大优先级越高 | `0`                      |
| `--bpu-cores`  | 指定用于运行模型的 BPU 核心编号                | `[0]`                    |
| `--test-img`   | 测试图像路径                                  | `segmentation.png` |

## 快速运行
- 运行模型
    - 使用默认参数
        ```bash
        python mobilenet_unet.py
        ```
    - 指定参数运行
        ```bash
        python mobilenet_unet.py \
        --model-path /app/model/basic/mobilenet_unet_1024x2048_nv12.bin \
        --priority 0 \
        --bpu-cores 0 \
        --test-img segmentation.png
        ```
- 查看结果

    运行成功后，会将结果绘制出来，保存到 segmentation_result.png
    ```bash
    Results saved to: segmentation_result.png
    ```

## 注意事项
- 若指定模型路径不存在，请在 `/app/model/basic`目录下寻找。

- 输出结果存储为segmentation_result.png，用户可自行查看。

- 如需了解更多部署方式或模型支持情况，请参考官方文档或联系平台技术支持。

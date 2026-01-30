# UnetMobileNet 语义分割示例

本示例展示了如何基于 `HB_HBMRuntime` 在 BPU 上运行 UNet-MobileNet 语义分割模型，支持图像预处理、推理、后处理（解析输出并叠加彩色分割掩码）等功能。

## 环境依赖
本样例无特殊环境需求，只需确保安装了pydev中的环境依赖即可。
```bash
pip install -r ../../requirements.txt
```

## 目录结构
```text
.
├── unet_mobilenet.py           # 主推理脚本
└── README.md                   # 使用说明
```

## 参数说明

| 参数名               | 说明                                      | 默认值                                 |
| ----------------- | --------------------------------------- | ----------------------------------- |
| `--model-path`    | 模型文件路径（.bin 格式）                              | `/app/model/basic/mobilenet_unet_1024x2048_nv12.bin` |
| `--test-img`      | 输入测试图像路径                                      | `segmentation.png`     |
| `--img-save-path` | 推理后结果图像保存路径                                  | `result.jpg`                        |
| `--priority`      | 模型优先级（0\~255，越大优先级越高）                    | `0`                                 |
| `--bpu-cores`     | 指定运行模型的 BPU 核心编号列表（如 `--bpu-cores 0 1`） | `[0]`                               |
| `--alpha-f`       | 可视化融合系数，`0.0=仅显示掩码`，`1.0=仅原图`           | `0.75`                              |


## 快速运行
- 运行模型
    - 使用默认参数
        ```bash
        python unet_mobilenet.py
        ```
    - 指定参数运行
        ```bash
        python unet_mobilenet.py \
        --model-path unet_mobilenet_1024x2048_nv12.bin \
        --test-img segmentation.png \
        --img-save-path result.jpg \
        --alpha-f 0.75 \
        --priority 0 \
        --bpu-cores 0
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

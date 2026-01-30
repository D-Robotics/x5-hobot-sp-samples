# Ultralytics YOLOE11 实例分割示例

本示例展示了如何使用 HB_HBMRuntime 在 BPU 上运行 Ultralytics YOLOE11 实例分割模型。程序实现了从输入图像的预处理、模型推理、后处理到结果可视化的完整流程。

## 环境依赖
本样例无特殊环境需求，只需确保安装了pydev中的环境依赖即可。
```bash
pip install -r ../../requirements.txt
```

## 目录结构
```text
.
├── yoloe11_seg.py              # 主推理脚本
└── README.md                   # 使用说明
```

## 参数说明
| 参数名               | 说明                                     | 默认值                                       |
| ----------------- | ------------------------------------------ | ----------------------------------------- |
| `--model-path`    | BPU 量化模型路径（\*.bin）                  | `/opt/hobot/model/x5/basic/yoloe_11s_seg_pf_nashe_640x640_nv12.bin` |
| `--test-img`      | 输入测试图像路径                            | `/app/res/assets/office_desk.jpg`            |
| `--label-file`    | 类别标签文件路径（每行一个类别）             | `/app/res/labels/coco_extended.names`        |
| `--img-save-path` | 推理结果图像保存路径                        | `result.jpg`                              |
| `--priority`      | 模型调度优先级（0\~255）                    | `0`                                       |
| `--bpu-cores`     | 使用的 BPU 核心编号（如 `--bpu-cores 0 1`） | `[0]`                                     |
| `--nms-thres`     | 非极大值抑制（NMS）的 IoU 阈值              | `0.7`                                     |
| `--score-thres`   | 目标检测置信度阈值                          | `0.25`                                    |
| `--is-open`       | 是否对掩码进行形态学操作（开操作）           | `False`                                   |
| `--is-point`      | 是否绘制掩码边缘轮廓点                      | `False`                                   |


## 快速运行
- 运行模型
    - 使用默认参数
        ```bash
        python yoloe11_seg.py
        ```
    - 指定参数运行
        ```bash
        python yoloe11_seg.py \
        --model-path yoloe_11s_seg_pf_nashe_640x640_nv12.bin \
        --priority 0 \
        --bpu-cores 0 \
        --test-img /app/res/assets/office_desk.jpg \
        --label-file /app/res/labels/coco_extended.names \
        --img-save-path result.jpg \
        --nms-thres 0.7 \
        --score-thres 0.25 \
        --is-open False \
        --is-point False
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

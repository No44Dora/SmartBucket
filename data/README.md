# data 目录说明

本目录用于存放训练数据及其预处理产物，建议组织如下：

```text
data/
├── raw/
│   └── images/                # 原始线稿图片
├── processed/
│   └── train_256/             # 预处理后的训练输入（如 256x256）
└── README.md
```

## 预处理脚本说明

已提供脚本：`scripts/prepare_image_inputs.py`。

用途：
- 将输入图片预处理到模型训练的固定输入尺寸；
- 对接近正方形的图片直接缩放；
- 对长方形图片先检测主体区域并进行尽量保留主体的裁剪，再缩放到目标尺寸。

示例命令：

```bash
python scripts/prepare_image_inputs.py \
  --input-dir data/raw/images \
  --output-dir data/processed/train_256 \
  --target-size 256
```

可调参数：
- `--square-tolerance`：判定“接近正方形”的阈值；
- `--foreground-threshold`：主体检测的灰度阈值（线稿通常黑线白底）；
- `--bbox-margin-ratio`：主体框外扩比例，避免裁剪过紧。

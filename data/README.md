# data 目录说明

本目录用于存放原始数据与裁剪缩放后的训练输入。

```text
data/
├── raw/
│   └── color/                 # 原始彩色图像
└── processed/
    └── color_256/             # 裁剪缩放后的训练输入图
```

## 预处理脚本

脚本：`scripts/prepare_image_inputs.py`

用途：
- 对原始彩色图像进行预裁剪并缩放（优先裁剪避免拉伸）；
- 输出统一输入尺寸的训练图片。

示例：

```bash
python scripts/prepare_image_inputs.py \
  --input-dir data/raw/color \
  --output-dir data/processed/color_256 \
  --target-size 256
```

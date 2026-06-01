# CNN 农产品成熟度检测

基于 CNN 的农产品成熟度三分类（未熟、半熟、熟透），支持标准模型和轻量化模型。

## 项目结构

```
├── config.py          # 配置参数
├── dataset.py         # 数据加载与预处理
├── model.py           # CNN 模型定义
├── train.py           # 训练脚本
├── predict.py         # 推理脚本
├── requirements.txt
├── datasets/          # 数据集
│   ├── train/
│   ├── val/
│   └── test/
├── models/            # 模型权重
├── results/           # 评估结果
│   ├── evaluate.py
│   └── evaluate_saved_model.py
└── docs/              # 辅助脚本
    ├── demo_presentation.py
    ├── generate_data.py
    └── split_data.py
```

## 安装

```bash
pip install -r requirements.txt
```

Python 3.8+, PyTorch 2.0+.

## 数据准备

按类别组织图像到 `datasets/` 目录：

```
datasets/
├── train/
│   ├── 未熟76/       # BBCH 71-76
│   ├── 半熟77/       # BBCH 77
│   └── 熟透78/       # BBCH 78+
├── val/
│   └── ... (同上)
└── test/
    └── ... (同上)
```

可使用 `docs/split_data.py` 从原始数据切分训练/验证集：

```bash
python docs/split_data.py --source ./test --train ./datasets/train --val ./datasets/val --ratio 0.8
```

## 使用

**训练：**

```bash
python train.py
```

**评估：**

```bash
python results/evaluate.py
```

**预测：**

```python
from predict import MaturityDetector
detector = MaturityDetector('models/best_model.pth')
result = detector.detect_image('path/to/image.jpg')
```

批量预测：

```python
results = detector.batch_detect('datasets/test/熟透78/', save_results=True)
```

## 模型

### MaturityCNN

5 个卷积块（3→32→64→128→256→512）+ 3 层全连接分类器（512→256→128→3），含 BatchNorm、Dropout(0.5)、自适应池化，Kaiming/Xavier 初始化。

### LightweightCNN

深度可分离卷积版本，参数量更少，适合边缘设备。

## 配置

在 `config.py` 中调整，主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| IMAGE_SIZE | 224 | 输入尺寸 |
| BATCH_SIZE | 32 | 批次大小 |
| EPOCHS | 5 | 训练轮数（正式训练建议 50+） |
| LEARNING_RATE | 0.001 | 学习率 |
| DROPOUT_RATE | 0.5 | Dropout 比例 |
| MOMENTUM | 0.9 | SGD 动量 |

## 数据增强

训练时使用随机裁剪、水平翻转、旋转(±15°)、颜色抖动、ImageNet 标准化。

## 技术栈

PyTorch / torchvision / scikit-learn / matplotlib / seaborn / OpenCV

## License

MIT


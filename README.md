# CNN 农产品成熟度检测

基于 CNN 的农产品成熟度三分类（未熟、半熟、熟透），支持标准模型和轻量化模型。

## 项目结构

```
深度学习/
├── config.py              # 配置参数
├── dataset.py             # 数据加载与预处理
├── model.py               # CNN 模型定义
├── train.py               # 标准模型训练 (~37MB)
├── train_lightweight.py   # 轻量化训练 (~0.2MB)
├── predict.py             # 推理脚本
├── requirements.txt       # 依赖列表
├── README.md              # 项目说明
├── .gitignore             # Git 忽略规则
├── results/               # 评估脚本
│   ├── evaluate.py
│   └── evaluate_saved_model.py
└── docs/                  # 文档
    └── TRAINING_GUIDE.md  # 训练指南
```

## 环境配置

### 1. Python 环境

**要求**: Python 3.8+

检查 Python 版本：
```bash
python --version
```

### 2. 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. GPU 支持（可选，但强烈推荐）

**NVIDIA GPU 用户**：
```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"
```

如果返回 `True`，PyTorch 会自动使用 GPU 加速训练。

**CPU only 用户**：
训练可以正常运行，但速度较慢。代码已自动检测并使用 CPU。

### 5. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

预期输出示例：
```
PyTorch: 2.0.0
CUDA: True
```

---

## 常见问题

### ❌ 报错：ModuleNotFoundError

**原因**: 缺少依赖包

**解决**:
```bash
pip install -r requirements.txt
```

### ❌ 报错：num_samples=0

**原因**: 数据目录为空或路径错误

**解决**:
1. 确认数据集目录已放置图片
2. 检查 `config.py` 中的路径配置
3. 目录结构应为：
   ```
   train_dataset/
   ├── 未熟/
   ├── 半熟/
   └── 熟透/
   ```

### ❌ 报错：CUDA out of memory

**原因**: GPU 显存不足

**解决**:
1. 减小 `config.py` 中的 `BATCH_SIZE`（如从 32 改为 16）
2. 或使用 LightweightCNN：`python train_lightweight.py`

### ❌ 报错：No module named 'torch'

**原因**: 未安装 PyTorch 或环境未激活

**解决**:
```bash
# 激活虚拟环境
venv\Scripts\activate

# 重新安装
pip install torch torchvision
```

### ❌ 报错：Permission denied

**原因**: 权限问题

**解决**:
```bash
# Windows: 以管理员身份运行终端
# Linux/Mac: 使用 sudo
sudo pip install -r requirements.txt
```

---

## 快速开始

```bash
# 1. 克隆项目
git clone <你的仓库地址>
cd <项目目录>

# 2. 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 准备数据
# 将图片放入 train_dataset/未熟、train_dataset/半熟、train_dataset/熟透 目录

# 5. 开始训练
python train.py
```

---

## 安装

```bash
pip install -r requirements.txt
```

Python 3.8+, PyTorch 2.0+.

## 数据准备

按类别组织图像到数据集目录：

```
train_dataset/
├── 未熟/       # 未成熟
├── 半熟/       # 半成熟
└── 熟透/       # 熟透

val_dataset/
└── ... (同上)

test_dataset/
└── ... (同上)
```

可使用 `docs/split_data.py` 从原始数据切分训练/验证集：

```bash
python docs/split_data.py --source ./test --train ./train_dataset --val ./val_dataset --ratio 0.8
```

## 训练

本项目支持 **2 种模型训练**：

| 模型 | 脚本 | 模型大小 | 说明 |
|------|------|----------|------|
| MaturityCNN | `train.py` | ~37MB | 标准高精度模型 |
| LightweightCNN | `train_lightweight.py` | **~0.2MB** | 轻量化压缩模型 |

**快速开始：**

```bash
# 标准模型训练 (~37MB)
python train.py

# 轻量化模型训练 (~0.2MB)
python train_lightweight.py
```

详细训练指南请查看 [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)

## 评估

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
results = detector.batch_detect('test_dataset/熟透/', save_results=True)
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

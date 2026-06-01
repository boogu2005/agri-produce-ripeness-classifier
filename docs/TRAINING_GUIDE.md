# 训练指南

## 可用的训练脚本

本项目提供 **4 种训练方式**，适用于不同场景：

| 脚本 | 模型 | 模型大小 | 适用场景 |
|------|------|----------|----------|
| `train.py` | MaturityCNN | ~37MB | 追求最高精度，服务器训练 |
| `train_lightweight.py` | LightweightCNN | **~0.2MB** | 边缘设备部署，模型压缩 |

---

## 方式一：标准模型训练 (MaturityCNN)

### 特点
- 精度最高
- 参数量：4,882,787
- 模型大小：~37MB
- 适合：服务器、GPU 环境

### 使用方法
```bash
python train.py
```

---

## 方式二：轻量化模型训练 (LightweightCNN)

### 特点
- 模型极小，便于部署
- 参数量：48,259
- 模型大小：~0.2MB
- 使用深度可分离卷积
- 适合：边缘设备、移动端、嵌入式

### 使用方法
```bash
python train_lightweight.py
```

---

## 方式三：自定义超参数训练

### 修改 config.py

在 `config.py` 中调整以下参数：

```python
class Config:
    IMAGE_SIZE = 224       # 输入图片尺寸
    BATCH_SIZE = 32        # 批次大小（显存不足时减小）
    EPOCHS = 50            # 训练轮数（正式训练建议 50+）
    LEARNING_RATE = 0.001   # 学习率
    WEIGHT_DECAY = 1e-4     # 权重衰减
    MOMENTUM = 0.9          # SGD 动量
    DROPOUT_RATE = 0.5     # Dropout 比例
```

### 选择模型

在 `train.py` 或 `train_lightweight.py` 中修改：

```python
# 标准模型（高精度）
model = get_model(model_name='cnn', num_classes=config.NUM_CLASSES)

# 轻量模型（高压缩）
model = get_model(model_name='lightweight', num_classes=config.NUM_CLASSES)
```

---

## 方式四：使用已有模型继续训练

### 加载预训练权重

修改 `train.py` 中的模型加载部分：

```python
# 加载已有模型继续训练
checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
# 继续训练...
```

---

## 训练输出

训练完成后会在以下位置生成文件：

```
models/
├── best_model.pth           # 最佳模型权重
├── checkpoint_epoch_10.pth   # 第10轮检查点
└── checkpoint_epoch_20.pth   # 第20轮检查点

results/training/
└── training_history.png      # 训练曲线图
```

---

## 训练结果评估

```bash
# 使用评估脚本
python results/evaluate_saved_model.py
```

---

## 常见问题

### 1. 显存不足 (CUDA Out of Memory)
- 减小 `config.py` 中的 `BATCH_SIZE`
- 使用 LightweightCNN

### 2. 训练效果不佳
- 增加训练数据量
- 调整学习率
- 增加训练轮数 (EPOCHS)
- 使用数据增强

### 3. 模型过拟合
- 增加 Dropout 比例
- 添加数据增强
- 减小模型复杂度

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MaturityCNN(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32), ConvBlock(32, 32),
            nn.MaxPool2d(2, 2), nn.BatchNorm2d(32),

            ConvBlock(32, 64), ConvBlock(64, 64),
            nn.MaxPool2d(2, 2), nn.BatchNorm2d(64),

            ConvBlock(64, 128), ConvBlock(128, 128),
            nn.MaxPool2d(2, 2), nn.BatchNorm2d(128),

            ConvBlock(128, 256), ConvBlock(256, 256),
            nn.MaxPool2d(2, 2), nn.BatchNorm2d(256),

            ConvBlock(256, 512), ConvBlock(512, 512),
            nn.MaxPool2d(2, 2), nn.BatchNorm2d(512),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


class LightweightCNN(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64), nn.ReLU6(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128), nn.ReLU6(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256), nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout_rate), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_model(model_name='cnn', num_classes=3, dropout_rate=0.5):
    if model_name == 'cnn':
        return MaturityCNN(num_classes, dropout_rate)
    elif model_name == 'lightweight':
        return LightweightCNN(num_classes, dropout_rate)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, input_size=(3, 224, 224)):
    print("=" * 60)
    print("Model Structure")
    print("=" * 60)
    print(model)
    print("=" * 60)
    print(f"Trainable parameters: {count_parameters(model):,}")
    print("=" * 60)

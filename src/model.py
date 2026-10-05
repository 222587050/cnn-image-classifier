"""Küçük, CPU'da da eğitilebilen bir CNN. BatchNorm ve Dropout açılıp kapatılabilir."""
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int, use_bn: bool) -> nn.Sequential:
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
    return nn.Sequential(*layers)


class SmallCNN(nn.Module):
    def __init__(self, use_bn: bool = False, dropout: float = 0.0, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32, use_bn),    # 32x32 -> 16x16
            conv_block(32, 64, use_bn),   # 16x16 -> 8x8
            conv_block(64, 128, use_bn),  # 8x8  -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

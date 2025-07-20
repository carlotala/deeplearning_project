"""
src/models/VGGTransferModel.py

Implementation of a transfer learning model using VGG16 as the backbone.
"""
import torch
import torch.nn as nn
import torchvision.models as models

from src.config.config_file import NUM_CLASSES


class VGGTransferModel(nn.Module):
    """
    Transfer Learning model using VGG16 backbone for classification.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # Load pretrained VGG16 and freeze features
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        for param in self.features.parameters():
            param.requires_grad = False

        # Adaptive pooling to fixed spatial size
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        flatten_dim = 512 * 7 * 7

        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # [B, 512*7*7]
        class_logits = self.classifier(x)
        return class_logits

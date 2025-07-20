"""
src/models/ResNetTransferModel.py

Implementation of a transfer learning model using ResNet50 as the backbone.
"""
import torch
import torch.nn as nn
import torchvision.models as models

from src.config.config_file import NUM_CLASSES


class ResNetTransferModel(nn.Module):
    """
    Transfer Learning model using ResNet50 backbone.
    Outputs:
      - bbox: Tensor of shape (batch_size, 4), values in [0,1]
      - class_logits: Tensor of shape (batch_size, num_classes)
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # Load pretrained ResNet50 and freeze layers
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # all conv layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        flatten_dim = resnet.fc.in_features  # 2048

        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor):
        # x: [B, 3, H, W]
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # [B, 2048]
        class_logits = self.classifier(x)
        return class_logits

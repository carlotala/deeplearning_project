
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetTransferModel(nn.Module):
    """
    Transfer Learning model using ResNet50 backbone.
    Outputs:
      - bbox: Tensor of shape (batch_size, 4), values in [0,1]
      - class_logits: Tensor of shape (batch_size, num_classes)
    """
    def __init__(self, num_classes: int):
        super().__init__()
        # Load pretrained ResNet50 and freeze layers
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # all conv layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        flatten_dim = resnet.fc.in_features  # 2048

        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(flatten_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

        # Classification head
        self.class_head = nn.Sequential(
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
        bbox = self.bbox_head(x)
        class_logits = self.class_head(x)
        return bbox, class_logits

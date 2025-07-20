import torch
import torch.nn as nn
import torchvision.models as models


class VGGTransferModel(nn.Module):
    """
    Transfer Learning model using VGG16 backbone.
    Outputs:
      - bbox: Tensor of shape (batch_size, 4), values in [0,1]
      - class_logits: Tensor of shape (batch_size, num_classes)
    """
    def __init__(self, num_classes: int):
        super().__init__()
        # Load pretrained VGG16 and freeze features
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        for param in self.features.parameters():
            param.requires_grad = False

        # Adaptive pooling to fixed spatial size
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        flatten_dim = 512 * 7 * 7

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
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # [B, 512*7*7]
        bbox = self.bbox_head(x)
        class_logits = self.class_head(x)
        return bbox, class_logits

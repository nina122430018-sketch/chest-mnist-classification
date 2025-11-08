# model.py

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights

class SimpleCNN(nn.Module):
    """Model CNN sederhana untuk backward compatibility"""
    def _init_(self, in_channels=1, num_classes=10):
        super()._init_()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2)
        self.pool = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1 if num_classes == 2 else num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ImprovedCNN(nn.Module):
    """
    Pretrained EfficientNet-B0 for Medical Imaging
    Using ImageNet pretrained weights with transfer learning
    """
    def _init_(self, in_channels=1, num_classes=2, use_pretrained=True):
        super()._init_()
        
        # Load pretrained EfficientNet-B0
        if use_pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Modify first conv layer untuk grayscale input (1 channel)
        if in_channels != 3:
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels, 
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            # Copy pretrained weights untuk grayscale (average RGB channels)
            if use_pretrained:
                with torch.no_grad():
                    self.backbone.features[0][0].weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )
        
        # Get feature dimension
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1 if num_classes == 2 else num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


class ResNetMedical(nn.Module):
    """
    Pretrained ResNet18 for Medical Imaging
    Lighter than EfficientNet, good for small datasets
    """
    def _init_(self, in_channels=1, num_classes=2, use_pretrained=True):
        super()._init_()
        
        # Load pretrained ResNet18
        if use_pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Modify first conv untuk grayscale
        if in_channels != 3:
            original_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            # Copy pretrained weights
            if use_pretrained:
                with torch.no_grad():
                    self.backbone.conv1.weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )
        
        # Replace classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1 if num_classes == 2 else num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# --- Bagian untuk pengujian ---
if _name_ == '_main_':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    BATCH_SIZE = 8
    
    print("=" * 70)
    print("PERBANDINGAN MODEL CNN")
    print("=" * 70)
    
    dummy_input = torch.randn(BATCH_SIZE, IN_CHANNELS, 28, 28)
    
    # Test SimpleCNN
    print("\nSimpleCNN (Legacy)")
    print("-" * 70)
    model_simple = SimpleCNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    params_simple = sum(p.numel() for p in model_simple.parameters())
    output = model_simple(dummy_input)
    print(f"Parameters: {params_simple:,}")
    print(f"Output: {output.shape}")
    
    # Test EfficientNet
    print("\nEfficientNet-B0 (Pretrained)")
    print("-" * 70)
    model_eff = ImprovedCNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    params_eff = sum(p.numel() for p in model_eff.parameters())
    output = model_eff(dummy_input)
    print(f"Parameters: {params_eff:,}")
    print(f"Output: {output.shape}")
    
    # Test ResNet18
    print("\nResNet18 (Pretrained)")
    print("-" * 70)
    model_resnet = ResNetMedical(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    params_resnet = sum(p.numel() for p in model_resnet.parameters())
    output = model_resnet(dummy_input)
    print(f"Parameters: {params_resnet:,}")
    print(f"Output: {output.shape}")
    
    print("\n" + "=" * 70)
    print("All models tested successfully!")
    print("=" * 70)
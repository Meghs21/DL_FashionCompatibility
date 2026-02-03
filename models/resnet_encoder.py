import torch
import torch.nn as nn
from models.resnet18_model import resnet18

class ResNetEncoder(nn.Module):
    """
    Feature extractor using ResNet-18 pretrained on ImageNet.
    
    This module:
    1. Loads pretrained ResNet-18 weights
    2. Removes the final classification layer
    3. Returns 512-dimensional feature vectors for each image
    
    Why this works:
    - ResNet learned visual patterns from 1.2M ImageNet images
    - Features are transferable to our fashion domain
    - Much better than random initialization
    - Saves training time and improves performance
    """
    
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        
        # Load pretrained ResNet-18
        # pretrained=True downloads weights from PyTorch's server
        self.resnet = resnet18(pretrained=True)
    
    def forward(self, x):
        """
        Extract features from clothing images.
        
        Args:
            x: Tensor of shape [batch_size, 3, 224, 224]
               - batch_size: how many images processed together
               - 3: RGB channels
               - 224, 224: image dimensions (ResNet standard)
        
        Returns:
            features: Tensor of shape [batch_size, 512]
                     Each row is a feature vector for one image
        
        Example:
            5 clothing items → [5, 3, 224, 224]
            ↓ Through ResNet
            5 feature vectors → [5, 512]
        """
        # Pass through ResNet conv layers (feature extraction)
        x = self.resnet.conv1(x)          # [B, 64, 112, 112]
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)        # [B, 64, 56, 56]
        
        # ResNet blocks progressively increase feature depth
        x = self.resnet.layer1(x)         # [B, 64, 56, 56]
        x = self.resnet.layer2(x)         # [B, 128, 28, 28]
        x = self.resnet.layer3(x)         # [B, 256, 14, 14]
        x = self.resnet.layer4(x)         # [B, 512, 7, 7]
        
        # Global average pooling (compress spatial dimensions)
        x = self.resnet.avgpool(x)        # [B, 512, 1, 1]
        
        # Flatten to 1D feature vector
        features = torch.flatten(x, 1)    # [B, 512]
        
        # Return 512-dimensional feature vector
        # NOT passing through fc layer (which would give 1000 dims for ImageNet)
        return features
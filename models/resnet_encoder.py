import torch.nn as nn
from models.resnet18_model import resnet18  # adjust this if your path is diff

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        self.resnet = resnet18(pretrained=True)

    def forward(self, x):
        features = self.resnet(x)  # [B, 512]
        return features
# cnn.py
import torch
import torch.nn as nn
from torchvision import models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        # Load a pretrained ResNet50 model
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove the last fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        return x

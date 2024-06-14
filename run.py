# run.py
import torch
import torch.nn as nn
from torchvision import models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

class DogSimilarityClassifier(nn.Module):
    def __init__(self, feature_extractor):
        super(DogSimilarityClassifier, self).__init__()
        self.features = feature_extractor

    def forward(self, x):
        x = self.features(x)
        return x

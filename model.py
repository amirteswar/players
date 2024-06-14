# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return self.features(x)

class DogSimilarityClassifier(nn.Module):
    def __init__(self, feature_extractor):
        super(DogSimilarityClassifier, self).__init__()
        self.features = feature_extractor
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 512)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

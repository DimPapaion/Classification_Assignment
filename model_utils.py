import torch
import torch.nn as nn
import torchvision.models as models

class MyVGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        vgg = models.vgg16(weights=None)

        vgg.classifier[6] = nn.Linear(4096, num_classes)

        self.features = vgg.features
        self.avg_pool = vgg.avgpool
        self.classif = vgg.classifier

    def forward(self, x):
        x = self.features(x)

        feat_last_conv = x

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classif(x)
        return feat_last_conv, x 


import torch.nn as nn

import timm


class ConvNeXt(nn.Module):
    def __init__(self):
        super(ConvNeXt, self).__init__()

        self.backbone = timm.create_model("convnextv2_femto", pretrained=False)
        self.backbone.head.fc = nn.Linear(self.backbone.head.fc.in_features, 18)

    def forward(self, x):
        x = self.backbone(x)
        return x


class EVATiny(nn.Module):
    def __init__(self):
        super(EVATiny, self).__init__()

        self.backbone = timm.create_model("eva02_tiny_patch14_224.mim_in22k", pretrained=False)
        self.clf = nn.Linear(192, 18)

    def forward(self, x):
        x = self.backbone(x)
        x = self.clf(x)
        return x


class Mobile(nn.Module):
    def __init__(self):
        super(Mobile, self).__init__()

        self.backbone = timm.create_model("mobilenetv4_conv_small", pretrained=False)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, 18)

    def forward(self, x):
        x = self.backbone(x)
        return x

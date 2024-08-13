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


class ColorClassifier(nn.Module):
    def __init__(self):
        super(ColorClassifier, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (224 - 4 + 2 * 1) / 2 + 1 = 112
            nn.MaxPool2d(kernel_size=4, stride=4),  # (112 - 4) / 4 + 1 = 28
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1),  # (28 - 2 + 2 * 1) / 2 + 1 = 15
            nn.MaxPool2d(kernel_size=3, stride=3),  # (15 - 3) / 3 + 1 = 5
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1),  # (5 - 2 + 2 * 1) / 1 + 1 = 6
            nn.MaxPool2d(kernel_size=6, stride=1),  # (6 - 6) / 1 + 1 = 1
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(256, 18)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
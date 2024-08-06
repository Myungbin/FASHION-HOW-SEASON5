import torch.nn as nn
import timm

class EVATiny(nn.Module):
    def __init__(self):
        super(EVATiny, self).__init__()

        self.backbone = timm.create_model("eva02_tiny_patch14_224", pretrained=True)
        self.daily = nn.Linear(192, 6)
        self.gender = nn.Linear(192, 5)
        self.embellishment = nn.Linear(192, 3)

    def forward(self, x):
        x = self.backbone(x)
        daily = self.daily(x)
        gender = self.gender(x)
        embellishment = self.embellishment(x)
        return daily, gender, embellishment
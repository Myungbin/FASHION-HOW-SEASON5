import torch.nn as nn
import timm

class ConvNeXt(nn.Module):
    def __init__(self):
        super(ConvNeXt, self).__init__()

        self.backbone = timm.create_model("convnextv2_femto", pretrained=False)
        self.clf = nn.Linear(1000, 18)

    def forward(self, x):
        x = self.backbone(x)
        x = self.clf(x)
        return x
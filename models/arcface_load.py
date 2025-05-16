# models/arcface_load.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.pool(x).view(b, c)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out)).view(b, c, 1, 1)
        return x * out.expand_as(x)

class BottleneckIRSE(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckIRSE, self).__init__()
        self.shortcut_layer = nn.Sequential()
        if stride == 2 or in_channels != out_channels:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            SEBlock(out_channels),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Backbone(nn.Module):
    def __init__(self, drop_ratio=0.6):
        super(Backbone, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        self.body = nn.Sequential(
            # stage 1
            BottleneckIRSE(64, 64, 1),
            BottleneckIRSE(64, 64, 1),
            BottleneckIRSE(64, 64, 1),
            BottleneckIRSE(64, 64, 2),
            # stage 2
            BottleneckIRSE(64, 128, 1),
            BottleneckIRSE(128, 128, 1),
            BottleneckIRSE(128, 128, 1),
            BottleneckIRSE(128, 128, 2),
            # stage 3
            BottleneckIRSE(128, 256, 1),
            BottleneckIRSE(256, 256, 1),
            BottleneckIRSE(256, 256, 1),
            BottleneckIRSE(256, 256, 2),
            # stage 4
            BottleneckIRSE(256, 512, 1),
            BottleneckIRSE(512, 512, 1),
            BottleneckIRSE(512, 512, 1),
        )
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(drop_ratio),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x, F.normalize(x)

def load_arcface_model(weight_path: str, device='cuda'):
    model = Backbone()
    ckpt = torch.load(weight_path, map_location=device)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model

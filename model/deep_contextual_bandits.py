import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class DeepContextualBandits(nn.Module):
    def __init__(self, feat_num):
        super(DeepContextualBandits, self).__init__()
        self.feat_num = feat_num

        # self.resnet = models.resnet34(pretrained=True)
        # for para in self.resnet.parameters():
        #     para.requires_grad = False
        #
        # self.resnet.fc = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.2)
        # )

        self.fc = nn.Sequential(
            nn.Linear(feat_num, 58),
            nn.ReLU(),
            nn.Linear(58, 56),
            nn.ReLU(),
            nn.Linear(56, 56),
            nn.ReLU(),
            nn.Linear(56, 28),
            nn.ReLU(),
            nn.Linear(28, 28),
            nn.ReLU(),
            nn.Linear(28, 1),
        )

    def forward(self, feat):
        out = self.fc(feat)
        return out





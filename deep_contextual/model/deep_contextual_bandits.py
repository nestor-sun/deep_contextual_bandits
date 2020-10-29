import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class DeepContextualBandits(nn.Module):
    def __init__(self, user_feat_num):
        super(DeepContextualBandits, self).__init__()
        self.user_feat_num = user_feat_num

        self.resnet = models.resnet34(pretrained=True)
        for para in self.resnet.parameters():
            para.requires_grad = False

        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256 + user_feat_num, 128),
            nn.Linear(128, 1)
        )

    def forward(self, image, user):
        out = self.resnet(image).squeeze(0)
        out = torch.cat((out, user), dim=-1)
        out = self.fc1(out)
        return out


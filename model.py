import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model

nclasses = 250


class Deit(nn.Module):
    def __init__(self):
        super(Deit, self).__init__()
        self.deit=create_model('deit_base_distilled_patch16_384',pretrained='true')
        self.dropout=nn.Dropout(0.5)
        self.fc=nn.Linear(1000,nclasses)
    def forward(self, x):
        x = self.deit(x)
        x = self.dropout(x)
        return self.fc(x)


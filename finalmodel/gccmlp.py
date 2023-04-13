import torch
import torch.nn.functional as F
import torch.nn as nn

print("MPL3 MODEL FOR GCC-PHAT-on our dataset-hri-gcc-mlp")


class Model(nn.Module):
    def __init__(self,time_frame=None):
        super(Model, self).__init__()

        self.MLP3 = nn.Sequential(
            nn.Linear(126, 1024), # input layer
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),

        )

        self.DoALayer = nn.Linear(1024, 360, bias=True)


    def forward(self, x):

        x = self.MLP3(x)

        y_pred = self.DoALayer(x)

        return y_pred

    def get_DoA_feature(self, x):
        return self.MLP3(x)

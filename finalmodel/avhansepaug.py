import torch
import torch.nn.functional as F
import torch.nn as nn
from .model_utilities import ConvBlock, init_gru, init_layer
import numpy as np



print("TF MODEL V2 -on AMV2020")

class HANLayer(nn.Module):
    #    only use 1 layer

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(HANLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask)[0]  # multi-head attention
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]  # multi-head attention
        src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)

class Model(nn.Module):
    def __init__(self,
                 time_frame=32):
        super(Model, self).__init__()

        self.d_model = 128
        print("d_model="+str(self.d_model))

        self.norm = nn.BatchNorm2d(10, affine=True)
        self.time_frame = time_frame
        self.nhead = 4
        self.drop = 0.3

        print("[param]:", time_frame, self.d_model, self.drop)
        # mel extraction  mel B * 4 * T * 21
        # input   B * T * 82
        self.gcc_ext = nn.Sequential(
                        nn.Linear(6*21, self.d_model, bias=True),
                        nn.BatchNorm1d(time_frame, self.d_model),  # 64, 128
                        # nn.LayerNorm(self.d_model),
                        # nn.ReLU(),
                        nn.ELU(),
                        nn.Dropout(p=0.1)
                       )

        # mel_ext
        self.mel_ext = nn.Sequential(
                        nn.Linear(4*21, self.d_model, bias=True),
                        nn.BatchNorm1d(time_frame,self.d_model),
                        # nn.LayerNorm(self.d_model),
                        # nn.ReLU(),
                        nn.ELU(),
                        nn.Dropout(p=0.1)
                       )

        # face_ext
        self.face_ext = nn.Sequential(
                        nn.Linear(2*21, self.d_model, bias=True),
                        nn.BatchNorm1d(time_frame, self.d_model),
                        # nn.LayerNorm(self.d_model),
                        # nn.ReLU(),
                        nn.ELU(),
                        nn.Dropout(p=0.1)
                       )

        self.HANlayer1 = HANLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=4*self.d_model, dropout=self.drop)
        self.HANlayer2 = HANLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=4*self.d_model, dropout=self.drop)
        self.HANlayer3 = HANLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=4*self.d_model, dropout=self.drop)
        self.HANlayer4 = HANLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=4*self.d_model, dropout=self.drop)
        self.HANlayer5 = HANLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=4*self.d_model, dropout=self.drop)
        self.HANlayer6 = HANLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=4*self.d_model, dropout=self.drop)

        self.MLP3 = nn.Sequential(
            nn.Linear(6*self.d_model, 1024, bias=True),
            nn.BatchNorm1d(time_frame, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 1024, bias=True),
            nn.BatchNorm1d(time_frame, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),

        )

        self.DoALayer = nn.Linear(1024, 360, bias=True)


        self.tah = nn.Tanh()
        # self.drop = nn.Dropout(p=0.3)
        self.norm = nn.LayerNorm(6*self.d_model)



    def forward(self, x):
        #

        x_gcc = x[:,:, 0:6,:]  # ([32, 64, 6, 21])
        x_gcc = x_gcc.contiguous().view(-1, self.time_frame, 6*21)  # B * T * F
        x_gcc = self.gcc_ext(x_gcc)  # ([32, 64, 128])

        x_mel = x[:,:, 6:10,:]
        x_mel = x_mel.contiguous().view(-1, self.time_frame, 4*21)  # B * T * F
        x_mel = self.mel_ext(x_mel)

        x_face = x[:,:, 10:12,:]
        x_face = x_face.contiguous().view(-1, self.time_frame, 2*21)  # B * T * F
        x_face = self.face_ext(x_face)

        # feature fusion
        fusion1 = self.HANlayer1(x_face, x_gcc)
        fusion2 = self.HANlayer2(x_gcc, x_face)
        fusion3 = self.HANlayer3(x_face, x_mel)
        fusion4 = self.HANlayer4(x_mel, x_face)
        fusion5 = self.HANlayer5(x_mel, x_gcc)
        fusion6 = self.HANlayer6(x_gcc, x_mel) # ([32, 64, 128])-> [bs, time, dim]

        # T * N * (6*512)
        fusion_out = torch.cat((fusion1, fusion2, fusion3, fusion4, fusion5, fusion6), dim=2) # [bs, time, dim] ([32, 64, 768])
        fusion_out = self.norm(fusion_out)
        # fusion_out = self.drop(fusion_out) # add one dropout

        azimuth_output = self.MLP3(fusion_out)
        azimuth_output = self.DoALayer(azimuth_output)

        return azimuth_output

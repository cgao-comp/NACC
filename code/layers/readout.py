import torch
import torch.nn as nn

# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            # print("torch.mean(seq, 1):{}".format(torch.mean(seq, 1)))
            # print("torch.mean(seq, 1)的size为:{}".format(torch.mean(seq, 1).size()))
            return torch.mean(seq, 1) ## 返回的维度为torch.Size([1, 128])
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)


import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True): ## in_ft 为输入的特征维数（cora为1433），out_ft 为输出的embedding的输出维数
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)  ##nn.linear 是维度转换，将in_ft维度通过全连接层输出为out_ft维度。
        self.act = nn.PReLU() if act == 'prelu' else act
        self.weight = nn.Parameter(torch.FloatTensor(in_ft, out_ft))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules(): ## modules()返回一个包含 当前模型 所有模块的迭代器。
            self.weights_init(m)


    def weights_init(self, m):
        if isinstance(m, nn.Linear): ##判断m是不是nn.Linear 类型。
            torch.nn.init.xavier_uniform_(m.weight.data) ##xavier_uniform 初始化（ 参数初始化方法）
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0) ##此处的seq_fts维度为[1, 2708, 128]， 所以要用squeeze降维，spmm为torch的稀疏矩阵乘法
            # out = torch.spmm(adj, torch.squeeze(seq_fts, 0))
            # output = torch.mm(out,self.weight)
        else:
            out = torch.bmm(adj, seq_fts)   ##两个三维矩阵相乘
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)


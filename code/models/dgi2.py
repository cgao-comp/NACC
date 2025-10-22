import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator
import community
# from models.gcnLayers import GraphConvolution
import torch.nn.functional as F


class DGI_node2(nn.Module):  ##DGI类继承自torch.nn.Module类 生成的特征是无向图
    def __init__(self, n_in, n_h, activation):
        super(DGI_node2, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)  ##n_in是输入特征的维数，n_h是embedding设定的维度
        # self.gcn2 = GCN(n_h, 128, activation)  ##n_in是输入特征的维数，n_h是embedding设定的维度
        self.read = AvgReadout()#创建一个AvgReadout层，用于图级别的读出操作
        # self.drop = dropout
        self.sigm = nn.Sigmoid() # 创建一个Sigmoid激活函数，用于二分类任务

    def forward(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)# 使用GCN层进行图的节点嵌入，得到节点特征h_1

        # h_11 = self.gcn2(h_1, adj, sparse)
        # h_11 = F.dropout(h_11, self.drop, training=self.training)
        c = self.read(h_1, msk)# 使用AvgReadout层对节点特征进行图级别的读出操作，得到图级别的特征c

        return h_1, c

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)

        c = self.read(h_1, msk)
        # print("h_1的size为")

        return h_1, c


class DGI_node22(nn.Module):  ##DGI类继承自torch.nn.Module类
    def __init__(self, n_in, n_h, activation, dropout):
        super(DGI_node22, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)  ##n_in是输入特征的维数，n_h是embedding设定的维度
        self.gcn2 = GCN(n_h, 128, activation)  ##n_in是输入特征的维数，n_h是embedding设定的维度
        self.read = AvgReadout()
        self.drop = dropout
        self.sigm = nn.Sigmoid()


    def forward(self, seq, adj, sparse, msk): 

        h_1 = self.gcn(seq, adj, sparse)

        h_11 = self.gcn2(h_1, adj, sparse)
        h_11 = F.dropout(h_11, self.drop, training=self.training)
        c = self.read(h_11, msk)
        
        return h_11, c

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)

        c = self.read(h_1, msk)
        # print("h_1的size为")


        return h_1, c



class gcn_network2(nn.Module):  ##DGI类继承自torch.nn.Module类
    def __init__(self, n_in, n_h, activation, dropout):
        super(gcn_network2, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)  ##n_in是输入特征的维数，n_h是embedding设定的维度
        self.gcn2 = GCN(n_h, 128, activation)   ##n_in是输入特征的维数，n_h是embedding设定的维度
        self.drop = dropout

    def forward(self, seq1, adj, sparse): 
        node_num = seq1.size()[1]
        ret_1 = self.gcn(seq1, adj, sparse)
        ret_1 = F.dropout(ret_1, self.drop, training=self.training)
        ret11 = self.gcn2(ret_1, adj, sparse)

        return ret11
    #print params
    def outputparameter(self):
        print("run output_parameter")
        print(type(self.named_parameters()))
        # 遍历模型参数
        for name, param in self.named_parameters():
            print(name, param.size())


class gcn_network3(nn.Module):  ##DGI类继承自torch.nn.Module类
    def __init__(self, n_in, n_h, activation, dropout, n_layer):
        super(gcn_network3, self).__init__()
        w_dict = {}
        self.gcn = GCN(n_in, n_h, activation)  ##n_in是输入特征的维数，n_h是embedding设定的维度
        self.gcn2 = GCN(n_h, 128, activation)  ##n_in是输入特征的维数，n_h是embedding设定的维度
        self.drop = dropout
        for lay in range(n_layer):
            self.w = torch.nn.Parameter(torch.rand(1))
            w_dict[lay] = self.w
        self.w_dict = w_dict
        print('www')

    def forward(self, seq1, adj, sparse):
        numLay = len(seq1)
        node_num = seq1[0].size()[1]
        ret_total = 0
        for lay in range(numLay):
            ret_1 = self.gcn(seq1[lay], adj[lay], sparse)
            ret_1 = F.dropout(ret_1, self.drop, training=self.training)
            ret11 = self.gcn2(ret_1, adj[lay], sparse)
            ret_total = ret11 * self.w_dict[lay] + ret_total
        print('w', self.w_dict)
        return ret_total
    #print params
    def outputparameter(self):
        print("run output_parameter")
        print(type(self.named_parameters()))
        # 遍历模型参数
        for name, param in self.named_parameters():
            print(name, param.size())
import torch
from torch.nn import Sequential, Linear, ReLU
import numpy as np

class MLP (torch.nn.Module):
    def __init__(self, input_dim, mlp_edge_model_dim=32):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        #连通多个网络层
        self.mlp_edge_model = Sequential(
            Linear(self.input_dim * 2, mlp_edge_model_dim),
            ReLU(),
            # torch.tanh(),
            Linear(mlp_edge_model_dim, 1)
        )
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    # 使用Xavier初始化方法初始化权重，偏置项初始化为0
    def forward(self, node_emb, node_num, edge):
        edge_index1 = []
        edge_index2 = []
        # for node in range(node_num):
        #     for secnod in range(node_num):
        #         edge_index1.append(node)
        #         edge_index2.append(secnod)
        # edge_index1 = np.array(edge_index1)
        # edge_index2 = np.array(edge_index2)
        # edge_index = np.vstack([edge_index1, edge_index2])
        # 构建边的索引
        edge = edge.T
        edge_index = torch.tensor(edge, dtype=torch.long)
        src, dst = edge_index[0], edge_index[1]
        # 获取节点嵌入
        node_emb = node_emb.squeeze()
        emb_src = node_emb[src]
        emb_dst = node_emb[dst]
        # 将源节点嵌入和目标节点嵌入拼接在一起作为边的特征
        edge_emb = torch.cat([emb_src, emb_dst], 1)
        # 通过MLP模型计算边的logits
        edge_logits = self.mlp_edge_model(edge_emb)

        return edge_logits, edge_index
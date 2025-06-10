import torch
from torch import Tensor
from torch.nn import Linear

from torch_geometric.utils import spmm

"""
Residual Aggregation Graph Encoder (RAGE)：
"""


class RAGEConv(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_l = Linear(in_channels, out_channels, bias=True)
        self.lin_r = Linear(in_channels, out_channels, bias=True)

        # self.lin_t = Linear(in_channels, in_channels)
        # self.lin_p = Linear(in_channels, in_channels)
        # self.dropout = Dropout(0.25)

    def forward(self, x: Tensor, edge_index) -> Tensor:
        # x = torch.sigmoid(self.lin_t(x)) * x
        x = [x, x]
        # 邻居聚合操作 mean self.dropout(x[0])
        out = spmm(edge_index, x[0], reduce="mean")
        out = self.lin_l(out) + self.lin_r(x[1])

        return out

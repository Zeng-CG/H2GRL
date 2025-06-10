from typing import List

import torch
from torch.nn import Embedding, Linear
from torch_geometric.nn.conv import LGConv
from torch_geometric.utils import to_torch_csr_tensor, subgraph

from cl_loss import InfoNCE
from my_conv import RAGEConv
from re_loss import BPRLoss


class H2GRL(torch.nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_layers: int,
            cat_layers: List,
            num_users: int,
            num_items: int,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.cat_layers = cat_layers

        self.embedding = Embedding(num_users + num_items, embedding_dim)
        self.conv = LGConv()  # 主图卷积层
        self._init_encoders(embedding_dim)
        self.reset_parameters()

        self.lin_g = Linear(embedding_dim * len(self.cat_layers), embedding_dim * len(self.cat_layers))
        self.lin = Linear(embedding_dim * len(self.cat_layers), embedding_dim * len(self.cat_layers), bias=True)

    def _init_encoders(self, emb_dim):
        """初始化同构图编码器"""
        enc_dim = emb_dim * len(self.cat_layers)
        self.encoders_u = RAGEConv(enc_dim, emb_dim)
        self.encoders_i = RAGEConv(enc_dim, emb_dim)

    def reset_parameters(self):
        # xavier_uniform初始化节点embedding
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.conv.reset_parameters()

    def forward(self, edge_index, **kwargs):
        # num_layers层 LGConv  每层保留 embs_layer
        self.main_emb, embs_layer = self._compute_main_embeddings(edge_index)
        if not kwargs.get('is_CL', True):  # 评估时候使用
            return self.main_emb

        # 同构图嵌入
        self.user_cl, self.item_cl = self._compute_cl_embeddings(
            embs_layer,
            kwargs['u_edge_index'], kwargs['i_edge_index'],
            kwargs['u_edge_value'], kwargs['i_edge_value'],
            kwargs['users'], kwargs['items']
        )

        return self.main_emb, (self.user_cl, self.item_cl)

    def _compute_main_embeddings(self, edge_index):
        """（LGConv部分）"""
        x = self.embedding.weight
        embeddings = [x]
        for _ in range(self.num_layers):
            x = self.conv(x, edge_index)
            embeddings.append(x)
        return torch.stack(embeddings[1:], dim=1).mean(dim=1), embeddings

    def _compute_cl_embeddings(self, embs_layer,
                               u_edge, i_edge,
                               u_val, i_val,
                               users, items):
        """处理同构图嵌入"""
        # 拼接所需的层
        emb_cat = torch.cat([embs_layer[i] for i in self.cat_layers], dim=1)
        emb_cat = torch.sigmoid(self.lin_g(emb_cat)) * emb_cat

        # 用户子图处理
        user_emb = emb_cat[users]  # 提取当前批次用户嵌入
        u_sparse = self._build_sparse_subgraph(u_edge, u_val, users, self.num_users)  # 提取子图

        item_global_ids = items
        item_local_ids = items - self.num_users  # 转换为原始物品ID
        item_emb = emb_cat[items]  # 提取当前批次物品嵌入
        i_sparse = self._build_sparse_subgraph(i_edge, i_val, item_local_ids, self.num_items)
        # 同构图编码
        user_cl = self.encoders_u(user_emb, u_sparse)
        item_cl = self.encoders_i(item_emb, i_sparse)

        return user_cl, item_cl

    def _build_sparse_subgraph(self, edge_index, edge_val, idx, num_nodes):
        sub_edge, sub_val = subgraph(subset=idx,
                                     edge_index=edge_index,
                                     edge_attr=edge_val,
                                     num_nodes=num_nodes,
                                     relabel_nodes=True)

        return to_torch_csr_tensor(sub_edge, sub_val, (len(idx), len(idx)))

    def recommendation_loss(self, pos_edge_rank, neg_edge_rank, node_id=None, lambda_reg=1e-4):
        loss_fn = BPRLoss(lambda_reg)
        emb = self.embedding.weight
        emb = emb if node_id is None else emb[node_id]
        return loss_fn(pos_edge_rank, neg_edge_rank, emb)

    def cal_cl_loss(self, user_view1, user_view2, item_view1, item_view2, temperature: float = 0.2):
        loss_fn = InfoNCE(temperature)
        user_cl_loss = loss_fn(user_view1, user_view2)
        item_cl_loss = loss_fn(item_view1, item_view2)
        return user_cl_loss + item_cl_loss

    def cal_cluster_loss(self, u_idx, i_idx,
                         node_centroids, node_2cluster,
                         temperature=0.2):

        node_2cluster_u = node_2cluster[u_idx]
        node_2cluster_i = node_2cluster[i_idx]

        unique_u_clusters, inverse_indices = torch.unique(node_2cluster_u, return_inverse=True)
        unique_users = torch.zeros_like(unique_u_clusters)
        unique_users[inverse_indices] = u_idx

        unique_i_clusters, inverse_indices = torch.unique(node_2cluster_i, return_inverse=True)
        unique_items = torch.zeros_like(unique_i_clusters)
        unique_items[inverse_indices] = i_idx

        user_embeddings = self.main_emb[unique_users]
        item_embeddings = self.main_emb[unique_items]

        cl_cluster_loss = self.cal_cl_loss(user_embeddings, node_centroids[unique_u_clusters],
                                           item_embeddings, node_centroids[unique_i_clusters], temperature)

        return cl_cluster_loss

    # def cal_cluster_cl_loss(self, user_view1, user_cluster_view2,
    #                         item_view1, item_cluster_view2,
    #                         temperature: float = 0.2):
    #     loss_fn = InfoNCE(temperature)
    #     user_cl_loss = loss_fn(user_view1, user_cluster_view2)
    #     item_cl_loss = loss_fn(item_view1, item_cluster_view2)
    #     return user_cl_loss + item_cl_loss

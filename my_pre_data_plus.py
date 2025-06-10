from collections import Counter
from typing import List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_torch_coo_tensor, to_edge_index, remove_self_loops


class MyDataset(InMemoryDataset):
    """处理用户-物品交互图数据的自定义数据集"""

    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], data_cls=Data)

    @property
    def raw_file_names(self) -> List[str]:
        return ['user_list.txt', 'item_list.txt', 'train.txt', 'test.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        data = Data()
        self._read_node_counts(data)  # 步骤1：读取基础数据
        self._process_train_test(data)  # 步骤2：处理训练/测试数据

        if self.pre_transform:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    # ----------------------- 核心逻辑分界线 -----------------------
    def _read_node_counts(self, data: Data):
        """读取用户和物品数量"""
        for path, key in zip(self.raw_paths[:2], ['user_num', 'item_num']):
            df = pd.read_csv(path, sep=' ', header=0)
            data[key] = len(df)

    def _process_train_test(self, data: Data):
        """统一处理训练集和测试集"""
        # 训练集需要构建相似度图
        train_index, item_weights, user_weights = self._process_file(
            self.raw_paths[2], is_train=True
        )

        # 构建用户-用户和物品-物品相似度图
        self._build_similarity_graphs(
            data, train_index, item_weights, user_weights
        )

        # 处理测试集（仅需边索引）
        test_index = self._process_file(
            self.raw_paths[3], is_train=False
        )
        data.edge_index_test = self._shift_item_ids(test_index, data.user_num)

    def _process_file(self, path: str, is_train: bool) -> torch.Tensor:
        """处理单个文件的核心逻辑"""
        rows, cols = [], []
        item_count = Counter()
        user_counts = []

        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                user_id = int(parts[0])
                item_ids = list(map(int, parts[1:]))

                # 记录交互关系
                rows.extend([user_id] * len(item_ids))
                cols.extend(item_ids)

                # 训练集需要统计信息
                if is_train:
                    item_count.update(map(str, item_ids))
                    user_counts.append(len(item_ids))

        index = torch.tensor([rows, cols])

        if is_train:
            # 计算权重参数
            item_weights = self._calc_weights(
                counts=item_count,
                num_nodes=self[0].item_num,
                prefix="item"
            )
            user_weights = self._calc_weights(
                counts=user_counts,
                num_nodes=self[0].user_num,
                prefix="user"
            )
            return index, item_weights, user_weights
        else:
            return index

    def _calc_weights(self, counts, num_nodes: int, prefix: str) -> torch.Tensor:
        """通用权重计算函数"""
        if prefix == "item":
            # 物品权重：log2(最大次数/当前次数) + 1
            max_count = max(counts.values())
            scores = [counts.get(str(i), 1) for i in range(num_nodes)]
        else:
            # 用户权重：log2(最大交互数/当前交互数) + 1
            max_count = max(counts)
            scores = counts

        weights = np.log2(max_count / np.array(scores)) + 1
        return torch.tensor(weights, dtype=torch.float32)

    def _build_similarity_graphs(self, data: Data, index: torch.Tensor,
                                 item_weights: torch.Tensor, user_weights: torch.Tensor):
        """构建用户和物品相似度图"""
        # 构建加权邻接矩阵
        ui_adj = self._build_weighted_adj(
            index,
            weights=item_weights[index[1]],
            shape=(data.user_num, data.item_num)
        )
        iu_adj = self._build_weighted_adj(
            index[[1, 0]],
            weights=user_weights[index[0]],
            shape=(data.item_num, data.user_num)
        )

        # 计算相似度矩阵
        user_sim = torch.sparse.mm(ui_adj, ui_adj.t())
        item_sim = torch.sparse.mm(iu_adj, iu_adj.t())

        # 后处理（过滤+归一化）
        data.u_edge_index, data.u_edge_value = self._postprocess_edges(
            user_sim, data.user_num
        )
        data.i_edge_index, data.i_edge_value = self._postprocess_edges(
            item_sim, data.item_num
        )

        # 原始边索引调整物品ID偏移
        data.edge_index = self._shift_item_ids(index, data.user_num)

    def _build_weighted_adj(self, index: torch.Tensor, weights: torch.Tensor,
                            shape: tuple) -> torch.Tensor:
        """构建带权重的稀疏邻接矩阵"""
        return to_torch_coo_tensor(
            edge_index=index,
            edge_attr=weights,
            size=shape
        )

    def _postprocess_edges(self, sim_matrix: torch.Tensor, num_nodes: int):
        """边后处理：去自环+过滤+归一化"""
        edge_index, edge_value = remove_self_loops(*to_edge_index(sim_matrix))
        mask = edge_value > torch.median(edge_value)
        edge_index, edge_value = edge_index[:, mask], edge_value[mask]
        return gcn_norm(edge_index, edge_value, num_nodes)

    def _shift_item_ids(self, index: torch.Tensor, user_num: int) -> torch.Tensor:
        """调整物品ID偏移以避免冲突"""
        shifted_index = index.clone()
        shifted_index[1] += user_num
        return shifted_index
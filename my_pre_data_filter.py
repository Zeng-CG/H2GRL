import warnings
from collections import Counter
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_torch_coo_tensor, to_edge_index, remove_self_loops

warnings.filterwarnings("ignore")


class MyDataset(InMemoryDataset):
    """用户-物品交互图数据集处理"""

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], data_cls=Data)

    @property
    def raw_file_names(self) -> List[str]:
        return ['user_list.txt', 'item_list.txt', 'train.txt', 'test.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass  # 无需下载数据

    def process(self):
        data = Data()

        # 读取用户和物品数量
        user_df = pd.read_csv(self.raw_paths[0], sep=' ', header=0)
        item_df = pd.read_csv(self.raw_paths[1], sep=' ', header=0)
        data['user_num'] = len(user_df)
        data['item_num'] = len(item_df)

        item_count = Counter()  # 统计每种商品被购买的数量
        user_interaction_count = []  # 用户购买商品的数量

        def process_split(path: str, is_train: bool):
            """处理单个数据分割（训练/测试）"""
            rows, cols = [], []
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1:  # 有效交互记录
                        u = int(parts[0])
                        items = list(map(int, parts[1:]))
                        rows.extend([u] * len(items))
                        cols.extend(items)
                        if is_train:
                            # 计算 每个物品被交互次数、每个用户交互物品数量
                            item_count.update(items)
                            # yelp2018数据集 用户是按照顺序排列的
                            user_interaction_count.append(len(items))

            edge_index = torch.tensor([rows, cols])

            if is_train:
                # 构建共同购买关系
                # ui_matrix = to_torch_coo_tensor(
                #     edge_index,
                #     torch.ones(edge_index.size(1)),
                #     size=(data['user_num'], data['item_num'])
                # )
                # 按照itemId升序
                int_key_count_pairs = [(int(key), count) for key, count in item_count.items()]
                sorted_int_key_count_pairs = sorted(int_key_count_pairs, key=lambda x: x[0])
                item_interaction_count = [count for _, count in sorted_int_key_count_pairs]
                # 评分因子
                (item_interaction_count_score,
                 user_interaction_count_score) = self._compute_interaction_scores(item_interaction_count,
                                                                                  user_interaction_count)
                # 重构带有权重的ui、iu的邻接矩阵
                ui_values = item_interaction_count_score[edge_index[1]]
                weighted_ui_adj = to_torch_coo_tensor(edge_index=edge_index,
                                                      edge_attr=ui_values,
                                                      size=(data['user_num'], data['item_num']))
                iu_values = user_interaction_count_score[edge_index[0]]
                weighted_iu_adj = to_torch_coo_tensor(edge_index=edge_index[[1, 0], :],
                                                      edge_attr=iu_values,
                                                      size=(data['item_num'], data['user_num']))
                # 构建用户-用户或物品-物品的同构图
                uu_matrix = torch.sparse.mm(weighted_ui_adj, weighted_ui_adj.T)
                ii_matrix = torch.sparse.mm(weighted_iu_adj, weighted_iu_adj.T)
                # 处理同构图边
                u_edge_index, u_edge_weight = self._process_edges(uu_matrix, data['user_num'])
                i_edge_index, i_edge_weight = self._process_edges(ii_matrix, data['item_num'])

                # 调整物品节点ID偏移（用户数 + 原始物品ID）,用于LGConv
                edge_index[1] += data['user_num']
                data.update({
                    "u_edge_index": u_edge_index,
                    "u_edge_value": u_edge_weight,
                    "i_edge_index": i_edge_index,
                    "i_edge_value": i_edge_weight,
                    "edge_index": edge_index  # 原始UI边索引
                })

            return edge_index

        # 处理训练和测试数据
        _ = process_split(self.raw_paths[2], is_train=True)
        test_edges = process_split(self.raw_paths[3], is_train=False)
        data['edge_index_test'] = test_edges

        # 用户自定义预处理
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def _process_edges(self, co_matrix: torch.Tensor, num_nodes: int) -> tuple:

        """处理边：去自环、阈值过滤、GCN归一化"""
        edge_index, edge_weight = remove_self_loops(*to_edge_index(co_matrix))

        # mask = edge_weight > 1  # 共同购买阈值
        # edge_index = edge_index[:, mask]
        # edge_weight = edge_weight[mask]

        # mask = edge_weight > torch.median(edge_weight)  # 中位数阈值
        # mask = edge_weight > torch.mean(edge_weight)

        mean = torch.mean(edge_weight)
        std = torch.std(edge_weight)
        mask = edge_weight > (mean + 1.0 * std)  # 保留偏强的边

        edge_weight = edge_weight[mask]
        edge_index = edge_index[:, mask]

        # edge_index, edge_weight = gcn_norm(edge_index, edge_weight, num_nodes)

        return edge_index, edge_weight

    def _compute_interaction_scores(self, item_counts, user_counts):
        max_item = max(item_counts)
        max_user = max(user_counts)
        item_scores = np.log2(max_item / np.array(item_counts)) + 1
        user_scores = np.log2(max_user / np.array(user_counts)) + 1
        return (
            torch.tensor(item_scores, dtype=torch.float32),
            torch.tensor(user_scores, dtype=torch.float32)
        )

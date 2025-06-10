from collections import Counter
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_torch_coo_tensor, to_edge_index, remove_self_loops


class MyDataset(InMemoryDataset):
    """自定义数据集类，用于处理用户-物品交互图数据"""

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], data_cls=Data)

    @property
    def raw_file_names(self) -> List[str]:
        """返回原始数据文件名"""
        return ['user_list.txt', 'item_list.txt', 'train.txt', 'test.txt']

    @property
    def processed_file_names(self) -> str:
        """返回处理后的数据文件名"""
        return 'data.pt'

    def download(self):
        """无需下载数据"""
        pass

    def process(self):
        """核心数据处理流程"""
        data = Data()

        # 统计物品交互次数和用户交互次数
        item_count = Counter()
        user_interaction_count = []

        # 读取用户和物品数量
        node_types_nums = ['user_num', 'item_num']
        for path, node_type_num in zip(self.raw_paths, node_types_nums):
            df = pd.read_csv(path, sep=' ', header=0)
            data[node_type_num] = len(df)

        def get_weighted_adj_by_count(index, item_interaction_count_sore, adj_type: str, add_weighted=True):
            """构建加权邻接矩阵"""
            values = torch.ones(index.size(1), dtype=torch.float32)
            if add_weighted:
                # 根据邻接类型计算权重值
                if adj_type == 'UI':
                    values = item_interaction_count_sore[index[1]]  # 物品权重
                elif adj_type == 'IU':
                    values = item_interaction_count_sore[index[0]]  # 用户权重

            # 构建稀疏矩阵并处理转置
            if adj_type == 'UI':
                return to_torch_coo_tensor(index, values, (data['user_num'], data['item_num']))
            else:
                return to_torch_coo_tensor(index[[1, 0], :], values, (data['item_num'], data['user_num']))

        def process_file(path, data, is_train):
            """处理单个文件（训练/测试）"""
            rows, cols, dst_s_s = [], [], []
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip().split(' ')
                    if len(line) > 1:  # 过滤空交互记录
                        if is_train:
                            # 训练集：统计物品交互次数和用户交互数量
                            item_count.update(line[1:])
                            user_interaction_count.append(len(line[1:]))

                        dst_s = list(map(int, line[1:]))  # 物品ID列表
                        dst_s_s.append(dst_s)
                        rows.extend([int(line[0])] * len(dst_s))  # 用户ID重复扩展
                        cols.extend(dst_s)  # 物品ID列表

            index = torch.tensor([rows, cols])  # 转换为边索引格式

            if is_train:

                # === 训练集特殊处理 ===
                # 计算物品和用户的权重分数（缓解流行度偏差）
                item_interaction_count_max = max(item_count.values())
                user_interaction_count_max = max(user_interaction_count)

                # 物品权重公式：log2(最大交互次数/当前物品交互次数) + 1
                item_scores = [
                    item_count[str(i)] for i in range(data['item_num'])
                ]  # 确保按物品ID顺序获取计数
                item_interaction_count_sore = np.log2(
                    item_interaction_count_max / np.array(item_scores)) + 1
                item_interaction_count_sore = torch.tensor(
                    item_interaction_count_sore, dtype=torch.float32)

                # 用户权重公式：log2(最大交互次数/当前用户交互次数) + 1
                user_interaction_count_sore = np.log2(
                    user_interaction_count_max / np.array(user_interaction_count)) + 1
                user_interaction_count_sore = torch.tensor(
                    user_interaction_count_sore, dtype=torch.float32)

                # 构建加权邻接矩阵
                weighted_ui_adj = get_weighted_adj_by_count(
                    index, item_interaction_count_sore, adj_type="UI")
                weighted_iu_adj = get_weighted_adj_by_count(
                    index, user_interaction_count_sore, adj_type="IU")

                # 通过矩阵乘法得到用户-用户和物品-物品相似度矩阵
                weighted_u_adj = torch.sparse.mm(weighted_ui_adj, weighted_ui_adj.T)
                weighted_i_adj = torch.sparse.mm(weighted_iu_adj, weighted_iu_adj.T)

                # 移除自环并过滤弱连接边
                u_edge_index, u_edge_value = remove_self_loops(*to_edge_index(weighted_u_adj))
                i_edge_index, i_edge_value = remove_self_loops(*to_edge_index(weighted_i_adj))

                # 基于中位数阈值过滤边
                u_mask = u_edge_value > torch.median(u_edge_value)
                u_edge_index = u_edge_index[:, u_mask]
                u_edge_value = u_edge_value[u_mask]

                i_mask = i_edge_value > torch.median(i_edge_value)
                i_edge_index = i_edge_index[:, i_mask]
                i_edge_value = i_edge_value[i_mask]

                # GCN归一化处理
                u_edge_index, u_edge_value = gcn_norm(u_edge_index, u_edge_value, data['user_num'])
                i_edge_index, i_edge_value = gcn_norm(i_edge_index, i_edge_value, data['item_num'])

                # 调整物品节点ID偏移（避免与用户ID冲突）
                index[1] += data['user_num']

                # 保存处理结果
                data.update({
                    "u_edge_index": u_edge_index,
                    "u_edge_value": u_edge_value,
                    "i_edge_index": i_edge_index,
                    "i_edge_value": i_edge_value,
                    "edge_index": index  # 原始UI边索引
                })


            else:
                # 测试集仅调整ID偏移
                # index[1] += data['user_num']
                data['edge_index_test'] = index

        # 处理训练和测试文件
        for path in self.raw_paths[2:]:
            is_train = 'train.txt' in path
            process_file(path, data, is_train)

        # 执行用户自定义的预处理
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

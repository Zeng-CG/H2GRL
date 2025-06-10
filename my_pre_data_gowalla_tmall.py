import pickle
from collections import Counter
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import coo_matrix
from torch import sparse_coo_tensor
from torch_geometric.data import HeteroData, InMemoryDataset, Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_edge_index, remove_self_loops


class MyDataset(InMemoryDataset):

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return ['../trnMat.pkl', '../tstMat.pkl']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):

        def get_weighted_adj_by_count(index, item_interaction_count_sore, adj_type: str, add_weighted=True):
            values = torch.ones(index.size(1), dtype=torch.float32)
            if add_weighted:
                # index[0] : 用户 ；index[1] : 物品
                if adj_type == 'UI':
                    values = item_interaction_count_sore[index[1]]  # UI和IU都要权重化
                elif adj_type == 'IU':
                    values = item_interaction_count_sore[index[0]]

            weighted_adj = sparse_coo_tensor(indices=index.to(torch.int64),
                                             values=values.to(torch.float32),
                                             size=(data['user_num'], data['item_num']))

            if adj_type == 'IU':
                weighted_adj = sparse_coo_tensor(indices=index[[1, 0], :].to(torch.int64),
                                                 values=values.to(torch.float32),
                                                 size=(data['item_num'], data['user_num']))
            return weighted_adj.coalesce()

        data = Data()

        with open(self.raw_paths[0], 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)

        # 计算用户和物品的数量
        data['user_num'] = ret.shape[0]
        data['item_num'] = ret.shape[1]

        item_count = Counter()  # 统计每种商品被购买的数量
        user_count = Counter()  # 用户购买商品的数量

        edge_index = torch.tensor([ret.row, ret.col], dtype=torch.int64)

        user_count.update(edge_index[0].tolist())
        item_count.update(edge_index[1].tolist())

        user_interaction_count = np.array([user_count.get(i, 1) for i in range(data['user_num'])])
        item_interaction_count = np.array([item_count.get(i, 1) for i in range(data['item_num'])])

        item_interaction_count_max = np.max(item_interaction_count)
        user_interaction_count_max = np.max(user_interaction_count)

        item_interaction_count_sore = np.log2(item_interaction_count_max / np.array(item_interaction_count))
        item_interaction_count_sore = torch.tensor(item_interaction_count_sore)

        user_interaction_count_sore = np.log2(user_interaction_count_max / np.array(user_interaction_count))
        user_interaction_count_sore = torch.tensor(user_interaction_count_sore)

        # 重构带有权重的ui、iu的邻接矩阵
        weighted_ui_adj = get_weighted_adj_by_count(edge_index, item_interaction_count_sore, adj_type="UI")
        weighted_iu_adj = get_weighted_adj_by_count(edge_index, user_interaction_count_sore, adj_type="IU")

        weighted_u_adj = torch.sparse.mm(weighted_ui_adj, weighted_ui_adj.T)
        weighted_i_adj = torch.sparse.mm(weighted_iu_adj, weighted_iu_adj.T)

        u_edge_index, u_edge_value = remove_self_loops(*to_edge_index(weighted_u_adj))
        i_edge_index, i_edge_value = remove_self_loops(*to_edge_index(weighted_i_adj))

        # mask = u_edge_value > torch.median(u_edge_value)
        # u_edge_value = u_edge_value[mask]
        # u_edge_index = u_edge_index[:, mask]
        #
        # mask = i_edge_value > torch.median(i_edge_value)
        # i_edge_value = i_edge_value[mask]
        # i_edge_index = i_edge_index[:, mask]

        u_edge_index, u_edge_value = gcn_norm(u_edge_index, u_edge_value, data['user_num'])
        i_edge_index, i_edge_value = gcn_norm(i_edge_index, i_edge_value, data['item_num'])

        mask = u_edge_value > torch.median(u_edge_value)
        u_edge_value = u_edge_value[mask]
        u_edge_index = u_edge_index[:, mask]

        mask = i_edge_value > torch.median(i_edge_value)
        i_edge_value = i_edge_value[mask]
        i_edge_index = i_edge_index[:, mask]



        data["u_edge_index"] = u_edge_index
        data["u_edge_value"] = u_edge_value
        data["i_edge_index"] = i_edge_index
        data["i_edge_value"] = i_edge_value

        with open(self.raw_paths[1], 'rb') as fs:
            ret_tset = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret_tset) != coo_matrix:
            ret_tset = sp.coo_matrix(ret_tset)

        edge_index_test = torch.tensor([ret_tset.row, ret_tset.col], dtype=torch.int64)

        edge_index_test[1] += data['user_num']  # 区分user和item
        edge_index[1] += data['user_num']

        data['edge_index'] = edge_index
        data['edge_index_test'] = edge_index_test

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

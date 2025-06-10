import torch
from torch_geometric.loader import DataLoader

from my_pre_data_filter import MyDataset


def load_data(config):
    dataset = MyDataset(config.data_path)
    data = dataset[0].to(config.device)

    # 数据预处理
    num_users, num_items = data['user_num'], data['item_num']
    train_edge_label_index = data.edge_index

    train_loader = DataLoader(
        range(train_edge_label_index.size(1)),
        shuffle=True,
        batch_size=config.batch_size,
    )

    # 处理测试数据
    test_edge_index = data['edge_index_test']
    # 用于测试
    train_edge_index = train_edge_label_index.clone()
    train_edge_index[1] -= num_users

    # 转换 lgcn 需要使用的边格式
    edge_index = torch.cat((train_edge_label_index, train_edge_label_index.flip([0])), dim=1)

    return {
        "num_users": num_users,
        "num_items": num_items,

        "u_edge_index": data["u_edge_index"],
        "i_edge_index": data["i_edge_index"],
        "u_edge_value": data["u_edge_value"],
        "i_edge_value": data["i_edge_value"],

        "train_loader": train_loader,

        "test_edge_index": test_edge_index,
        "train_edge_label_index": train_edge_label_index,
        "train_edge_index": train_edge_index,

        "edge_index": edge_index,
    }

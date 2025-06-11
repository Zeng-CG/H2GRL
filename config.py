import torch


class Config:
    def __init__(self):
        self.dataset_name = "amazon-book"
        self.cl_weight = 0.8
        self.cl_weight_epochs = 47
        self.cl_weight_low = 0.8
        self.batch_size = 4096
        self.epochs = 60

        # self.dataset_name = "yelp2018"
        # self.cl_weight = 0.5
        # self.cl_weight_epochs = 14
        # self.cl_weight_low = 0.4
        # self.batch_size = 1024
        # self.epochs = 50
        # self.cluster_loss_start = 20

        # 数据集配置
        self.data_path = f"./dataset/{self.dataset_name}"
        self.test_batch_size = 2048

        # 模型参数
        self.embedding_dim = 64
        self.num_layers = 2
        self.cat_layers = [1, 2]

        # 训练参数
        self.lr = 0.001
        self.temperature_cl = 0.2

        self.cluster_loss_start = 50
        self.cl_cluster_weight = 0.01
        self.temperature_cluster = 0.3

        # 评估参数
        self.top_k = 20

        # 保存配置
        self.model_save_dir = f"./model_save/{self.dataset_name}/"
        self.embedding_save_dir = f"./embedding_save/{self.dataset_name}/"

        self.is_save = False
        if self.dataset_name == "amazon-book":
            self.sava_start_epochs = 40
        elif self.dataset_name == "yelp2018":
            self.sava_start_epochs = 10

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

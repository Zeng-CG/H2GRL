import faiss
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, data, config, device):
        self.model = model.to(device)
        self.data = data
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    def _train_epoch(self, epoch):
        self.model.train()
        total = {k: 0.0 for k in ["rec_loss", "cl_loss", "cluster_loss"]}
        batch_num = 0
        # KNN
        if epoch > self.config.cluster_loss_start:
            with torch.no_grad():
                embeddings = self.model(self.data["edge_index"], is_CL=False)
            node_centroids, node_2cluster = self._get_centroids(embeddings)

        for batch in tqdm(self.data["train_loader"], desc=f"Epoch-{epoch}"):
            # for batch in self.data["train_loader"]:
            # 数据准备阶段(负采样，用户物品去重)
            edge_label, users, items = self._prepare_batch(batch)
            # 计算 LGCN 嵌入，同构图 RAGEConv 嵌入
            emb, cl_emb = self.model(
                self.data["edge_index"],
                u_edge_index=self.data["u_edge_index"],
                i_edge_index=self.data["i_edge_index"],
                u_edge_value=self.data["u_edge_value"],
                i_edge_value=self.data["i_edge_value"],
                users=users, items=items)

            # 只将batch内的节点做梯度回传
            src = emb[edge_label[0]]
            dst = emb[edge_label[1]]
            pos, neg = (src * dst).sum(dim=-1).chunk(2)
            # 包括了BPRLoss L2正则化 的推荐损失Loss
            rec_loss = self.model.recommendation_loss(pos, neg, node_id=edge_label.unique())

            # 同构异构对比损失
            user_cl_emb, item_cl_emb = cl_emb
            cl_loss = self.model.cal_cl_loss(emb[users],
                                             user_cl_emb,
                                             emb[items],
                                             item_cl_emb,
                                             temperature=self.config.temperature_cl)
            cl_weight_1 = self.config.cl_weight
            if epoch > self.config.cl_weight_epochs:
                cl_weight_1 = self.config.cl_weight_low

            # 聚类对比损失 TODO 优化
            cluster_loss = 0
            if epoch > self.config.cluster_loss_start:
                cluster_loss = self.model.cal_cluster_loss(
                    users, items,
                    node_centroids, node_2cluster,
                    temperature=0.3)
            cl_weight_2 = self.config.cl_cluster_weight

            loss = rec_loss + cl_weight_1 * cl_loss + cl_weight_2 * cluster_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # 损失统计
            total["rec_loss"] += rec_loss
            total["cl_loss"] += cl_loss
            total["cluster_loss"] += cluster_loss
            batch_num += 1
        return {k: v / batch_num for k, v in total.items()}

    def _prepare_batch(self, batch):
        """处理单个batch的数据拼接"""
        pos_edges = self.data["train_edge_label_index"][:, batch]
        neg_edges = self._sample_negatives(pos_edges)
        return (
            torch.cat([pos_edges, neg_edges], dim=1),
            torch.unique(pos_edges[0]),
            torch.unique(pos_edges[1])
        )

    def _sample_negatives(self, pos_edges):
        """负样本采样（可扩展不同采样策略）"""
        return torch.stack([
            pos_edges[0],
            torch.randint(
                self.data["num_users"],
                self.data["num_users"] + self.data["num_items"],
                size=pos_edges[0].shape,
                device=self.device
            )
        ], dim=0)

    def _get_centroids(self, node_emb):
        node_centroids, node_2cluster = self._run_kmeans(node_emb, 1000)
        return node_centroids, node_2cluster

    def _run_kmeans(self, x, num_cluster):
        """
        使用K-means算法对输入的张量x进行聚类。
        参数:
        - x: 输入的张量，例如用户或物品的嵌入向量
        返回:
        - centroids: 聚类中心的张量
        - node2cluster: 每个节点（用户或物品）所属的聚类索引
        """
        # 初始化faiss库的Kmeans对象，设置维度、聚类数和是否使用GPU
        kmeans = faiss.Kmeans(d=x.shape[1], k=num_cluster)
        kmeans.cp.min_points_per_centroid = 20
        # 使用K-means算法对输入x进行训练
        x = x.detach().cpu().numpy()
        kmeans.train(x)
        # 获取聚类中心
        cluster_cents = kmeans.centroids
        # 对输入x进行聚类索引搜索，即找到每个点最近的聚类中心
        _, I = kmeans.index.search(x, 1)
        # 将聚类中心转换为CUDA张量，并进行L2归一化，以便在后续的计算中使用
        centroids = torch.Tensor(cluster_cents).to(self.device)
        node2cluster = torch.LongTensor(I).to(self.device).squeeze(-1)
        return centroids, node2cluster

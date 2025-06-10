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
        total = {k: 0.0 for k in ["rec_loss", "cl_loss"]}
        batch_num = 0

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
            cl_weight = self.config.cl_weight
            if epoch > self.config.cl_weight_epochs:
                cl_weight = self.config.cl_weight_low
            loss = rec_loss + cl_weight * cl_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # 损失统计
            total["rec_loss"] += rec_loss
            total["cl_loss"] += cl_loss
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

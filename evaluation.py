import torch
from torch_geometric.metrics import LinkPredPrecision, LinkPredRecall, LinkPredNDCG
from torch_geometric.nn import MIPSKNNIndex


class RecommenderEvaluator:
    def __init__(self, config, num_users: int, num_items: int, device: torch.device):
        """
        推荐系统评估器

        参数:
            config: 包含评估参数的配置对象
            num_users: 用户数量
            num_items: 物品数量
            device: 计算设备
        """
        self.config = config
        self.num_users = num_users
        self.num_items = num_items
        self.device = device

        # 初始化评估指标
        self.metrics = {
            'precision': LinkPredPrecision(k=config.top_k),
            'recall': LinkPredRecall(k=config.top_k),
            'ndcg': LinkPredNDCG(k=config.top_k)
        }
        for metric in self.metrics.values():
            metric.to(device)

    def _batch_users(self, total: int) -> range:
        """生成用户批次范围"""
        return range(0, total, self.config.test_batch_size)

    def _reset_metrics(self):
        """重置所有指标状态"""
        for metric in self.metrics.values():
            metric.reset()

    @torch.no_grad()
    def evaluate(
            self,
            model: torch.nn.Module,
            full_graph,
            test_edges: torch.Tensor,
            exclude_edges: torch.Tensor
    ) -> dict:
        """
        执行完整评估流程

        参数:
            model: 要评估的模型
            full_graph: 全量图数据
            test_edges: 测试集边数据
            exclude_edges: 需要排除的边数据（通常为训练集）

        返回:
            包含各项指标的字典
        """
        # 获取全量嵌入
        with torch.no_grad():
          embeddings = model(full_graph, is_CL=False)

        user_emb = embeddings[:self.num_users]
        item_emb = embeddings[self.num_users:self.num_users + self.num_items]
        # 初始化MIPS索引
        mips_index = MIPSKNNIndex(item_emb)
        # 重置指标
        self._reset_metrics()

        # 分批次处理用户
        for start in self._batch_users(self.num_users):
            end = min(start + self.config.test_batch_size, self.num_users)
            user_ids = torch.arange(start, end, device=self.device)
            # 当前批次的用户嵌入
            batch_user_emb = user_emb[start:end]
            # 排除已知边
            exclude_mask = torch.isin(exclude_edges[0], user_ids)
            exclude_batch = exclude_edges[:, exclude_mask].clone()
            exclude_batch[0] -= start  # 调整用户ID偏移
            # 执行搜索
            _, preds = mips_index.search(batch_user_emb, self.config.top_k, exclude_links=exclude_batch)
            # 获取真实标签
            test_mask = torch.isin(test_edges[0], user_ids)
            ground_truth = test_edges[:, test_mask].clone()
            ground_truth[0] -= start  # 调整用户ID偏移

            # 更新指标
            for metric in self.metrics.values():
                metric.update(preds, ground_truth)

        # 组装结果
        return {name: float(metric.compute()) for name, metric in self.metrics.items()}

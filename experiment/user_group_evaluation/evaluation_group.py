import torch
from torch_geometric.metrics import LinkPredPrecision, LinkPredRecall, LinkPredNDCG
from torch_geometric.nn import MIPSKNNIndex


class RecommenderEvaluator:
    def __init__(self, device: torch.device):
        self.device = device
        self.metric_classes = {
            'precision': LinkPredPrecision,
            'recall': LinkPredRecall,
            'ndcg': LinkPredNDCG
        }

    def _init_metrics(self):
        return {
            name: metric_class(k=20).to(self.device)
            for name, metric_class in self.metric_classes.items()
        }

    @torch.no_grad()
    def evaluate_user_group(
            self,
            user_emb: torch.Tensor,
            item_emb: torch.Tensor,
            group_users: dict[str, list[int]],  # 分组名 -> 用户ID列表
            test_edges: torch.Tensor,
            exclude_edges: torch.Tensor
    ) -> dict:
        # 初始化 MIPS 索引
        mips_index = MIPSKNNIndex(item_emb)
        group_results = {}

        for group_name, user_id_list in group_users.items():
            if not user_id_list:
                continue  # 跳过空组

            user_ids = torch.tensor(user_id_list, device=self.device)
            emb = user_emb[user_ids]
            metrics = self._init_metrics()

            # 构造当前组的 ground truth 和排除边（训练边）
            exclude_mask = torch.isin(exclude_edges[0], user_ids)
            exclude_batch = exclude_edges[:, exclude_mask].clone()
            exclude_batch[0] = torch.bucketize(exclude_batch[0], user_ids)

            test_mask = torch.isin(test_edges[0], user_ids)
            ground_truth = test_edges[:, test_mask].clone()
            ground_truth[0] = torch.bucketize(ground_truth[0], user_ids)

            # 搜索 + 评估（不分批）
            _, preds = mips_index.search(emb, 20, exclude_links=exclude_batch)
            for metric in metrics.values():
                metric.update(preds, ground_truth)

            # 保存当前组的评估结果
            group_results[group_name] = {k: float(m.compute()) for k, m in metrics.items()}

        return group_results

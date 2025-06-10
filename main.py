import os

import torch

from config import Config
from data_loader import load_data
from evaluation import RecommenderEvaluator
from my_model import H2GRL
from trainer import Trainer

print("数据加载中...")
config = Config()
data = load_data(config)
print("数据加载完毕...")

print("======= Training Configuration =======")
for key, value in config.__dict__.items():
    print(f"{key}: {value}")
print("======================================")

# 初始化模型
model = H2GRL(
    embedding_dim=config.embedding_dim,
    num_layers=config.num_layers,
    cat_layers=config.cat_layers,
    num_users=data["num_users"],
    num_items=data["num_items"],
)
# 创建训练器
trainer = Trainer(
    model=model,
    data=data,
    config=config,
    device=config.device,
)
# 初始化评估器
evaluator = RecommenderEvaluator(
    config=config,
    num_users=data["num_users"],
    num_items=data["num_items"],
    device=config.device,
)
# 训练循环
for epoch in range(1, config.epochs + 1):
    metrics = trainer._train_epoch(epoch)
    results = evaluator.evaluate(
        model=model,
        full_graph=data["edge_index"],
        test_edges=data["test_edge_index"],
        exclude_edges=data["train_edge_index"]
    )
    print(f"Epoch-{epoch}: 评估[ P@{config.top_k}={results['precision']:.4f}, "
          f"R@{config.top_k}={results['recall']:.4f}, "
          f"N@{config.top_k}={results['ndcg']:.4f}]"
          f",损失[ rec_loss={metrics['rec_loss'].item():.4f},"
          f"cl_loss={metrics['cl_loss'].item():.4f}"
          f"{', cluster_loss=' + format(metrics['cluster_loss'].item(), '.4f') if epoch > config.cluster_loss_start else ''}]"
          )


    # 嵌入保存
    if config.is_save and epoch >= config.sava_start_epochs:
        with torch.no_grad():
            embeddings = model(data["edge_index"], is_CL=False)
            os.makedirs(config.embedding_save_dir, exist_ok=True)
            save_path = os.path.join(config.embedding_save_dir, f"{epoch}.pt")
            torch.save(embeddings, save_path)

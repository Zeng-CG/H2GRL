import csv
import os
from datetime import datetime
from itertools import product

import torch

from config import Config
from data_loader import load_data
from evaluation import RecommenderEvaluator
from my_model import H2GRL
from trainer import Trainer

# 定义网格搜索参数空间
param_grid = {
    "cl_weight": [0.45, 0.5, 0.55],
    "cl_weight_epochs": [15, 16, 17, 18, 19, 20],
    "cl_weight_low": [0.2, 0.3, 0.35],
}

# 生成参数组合
param_combinations = [
    dict(zip(param_grid.keys(), values))
    for values in product(*param_grid.values())
]


def setup_logging(config):
    """创建日志目录和文件"""
    log_dir = f"./logs/{config.dataset_name}.log/"
    os.makedirs(log_dir, exist_ok=True)

    log_filename = f"hparam_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_path = os.path.join(log_dir, log_filename)

    # 写入CSV头
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'cl_weight', 'cl_weight_epochs', 'cl_weight_low',
            'epoch', 'precision', 'recall', 'ndcg',
            'rec_loss', 'cl_loss'
        ])
    return log_path


# 主调参流程
log_path = None
for params in param_combinations:
    config = Config()  # 每次创建新配置实例
    config.cl_weight = params["cl_weight"]
    config.cl_weight_epochs = params["cl_weight_epochs"]
    config.cl_weight_low = params["cl_weight_low"]

    # 延迟创建日志文件（确保dataset_name可用）
    if not log_path:
        log_path = setup_logging(config)

    print(f"\n=== 开始训练 {params} ===")

    # 加载数据
    data = load_data(config)

    # 初始化模型
    model = H2GRL(
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        cat_layers=[1, 2],
        num_users=data["num_users"],
        num_items=data["num_items"],
    ).to(config.device)

    # 创建训练器和评估器
    trainer = Trainer(
        model=model,
        data=data,
        config=config,
        device=config.device,
    )
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

        # 写入日志
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                params["cl_weight"],
                params["cl_weight_epochs"],
                params["cl_weight_low"],
                epoch,
                round(results['precision'], 5),
                round(results['recall'], 5),
                round(results['ndcg'], 5),
                round(metrics['rec_loss'].item(), 5),
                round(metrics['cl_loss'].item(), 5)
            ])

    # 资源清理
    del model, trainer, evaluator
    torch.cuda.empty_cache()

print("\n=== 网格搜索完成 ===")

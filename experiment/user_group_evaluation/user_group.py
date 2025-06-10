## 将用户按照交互量的大小进行分组
import torch

from config import Config
from data_loader import load_data
from experiment.user_group_evaluation.evaluation_group import RecommenderEvaluator

dataset_name = "amazon-book"
data_path = f"../../dataset/{dataset_name}/raw/train.txt"

# 存储用户 -> 交互数量
u_dic = {}
with open(data_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # 跳过空行
        parts = line.split()
        if len(parts) <= 1:
            continue  # 无交互物品
        u = int(parts[0])
        items = parts[1:]
        if u in u_dic:
            print(f"Warning: 用户 {u} 多次出现，可能数据有问题")
        u_dic[u] = len(items)
# 分组定义
groups = {
    "g1_≤17": [],
    "g2_18-20": [],
    "g3_21-25": [],
    "g4_26-40": [],
    "g5_>40": []
}
# 用户分配到各组
for u, count in u_dic.items():
    if count <= 17:
        groups["g1_≤17"].append(u)
    elif count <= 20:
        groups["g2_18-20"].append(u)
    elif count <= 25:
        groups["g3_21-25"].append(u)
    elif count <= 40:
        groups["g4_26-40"].append(u)
    else:
        groups["g5_>40"].append(u)

# 输出每组用户数量
for g, users in groups.items():
    print(f"{g}: {len(users)} users")

# 根据保存的embedding 进行指标计算
# 初始化评估器
config = Config()

config.top_k = 20
config.data_path = f"../../dataset/{config.dataset_name}"
data = load_data(config)

# embedding的加载
all_embedding = torch.load(f"../../embedding_save/{config.dataset_name}/54.pt")

evaluator = RecommenderEvaluator(device='cuda')

results = evaluator.evaluate_user_group(
    user_emb=all_embedding[:data["num_users"]],
    item_emb=all_embedding[data["num_users"]:],
    group_users=groups,
    test_edges=data["test_edge_index"],
    exclude_edges=data["train_edge_index"]
)

for group, metrics in results.items():
    print(f"{group}: {metrics}")


def merge_group_metrics(group_results: dict[str, dict[str, float]],
                        group_users: dict[str, list[int]]) -> dict[str, float]:
    total_users = sum(len(users) for users in group_users.values())
    merged = {metric: 0.0 for metric in next(iter(group_results.values())).keys()}

    for group_name, metrics in group_results.items():
        weight = len(group_users[group_name]) / total_users
        for metric, value in metrics.items():
            merged[metric] += weight * value
    return merged


merged = merge_group_metrics(results, groups)
print("Merged from groups:", merged)

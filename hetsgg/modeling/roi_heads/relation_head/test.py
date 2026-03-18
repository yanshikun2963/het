import torch
import torch.nn as nn
# a = torch.tensor([1,8,4,0,2,3,6,5,7])
# sort = torch.argsort(a)
# print(a)
# print(sort)
# print(a[sort])
gate = nn.Linear(2048,3)
x = torch.randn(874,2048)
n = x.shape[0]
device = x.device
        
        # 步骤1：计算理论分配基数
base_size = n // 3
remainder = n % 3
target_counts = torch.tensor( #[n1,n2,n3]
    [base_size + 1 if i < remainder else base_size 
     for i in range(3)],
    device=device
)
        
    # 步骤2：门控计算
logits = gate(x)
probs = torch.softmax(logits, dim=1)
        
        # 步骤3：初始分配
expert_choice = torch.argmax(probs, dim=1)
        
        # 步骤4：动态平衡调整 -------------------------------------------------
        # 创建分配池
allocation_pool = [[] for _ in range(3)]
for idx, expert in enumerate(expert_choice):
    allocation_pool[expert].append(idx)
        
        # 调整超额专家
for expert in range(3):
    current_count = len(allocation_pool[expert])
    if current_count <= target_counts[expert]:
        continue
                
    # 需要转移的样本数
    overflow = current_count - target_counts[expert]
            
    # 选择该专家中概率最低的样本进行转移
    expert_probs = probs[allocation_pool[expert], expert]
    sorted_indices = torch.argsort(expert_probs)[:overflow]
    to_redistribute = [allocation_pool[expert][i] for i in sorted_indices]
            
    # 从原专家移除
    allocation_pool[expert] = [idx for idx in allocation_pool[expert] 
                                      if idx not in to_redistribute]
            
    # 重新分配到其他专家
    for idx in to_redistribute:
        # 寻找最需要样本且概率次高的专家
        candidate_experts = torch.argsort(probs[idx], descending=True)[1:]
        for candidate in candidate_experts:
            if len(allocation_pool[candidate]) < target_counts[candidate]:
                allocation_pool[candidate].append(idx)
                break

# 步骤5：最终分配验证
expert_data = []
expert_indices = []
total_allocated = 0
for i in range(3):
    indices = torch.tensor(allocation_pool[i], device=device)
    expert_indices.append(indices)
    expert_data.append(x[indices])
    total_allocated += len(indices)
        
        # 完整性检查
assert total_allocated == n, f"分配错误：{total_allocated}/{n}"
assert torch.all(torch.bincount(torch.cat(expert_indices)) == 1), "存在重复分配"
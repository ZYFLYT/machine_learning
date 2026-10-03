import torch

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

print(x)
print(x.tolist())

import warnings

warnings.filterwarnings(
    "ignore", message=r".*(owner does not match|Permission mismatch|TASK_QUEUE_ENABLE)"
)

# 可以看到，维度每增加一维，就多套一层方括号。
# `shape` 告诉你每个维度有多大，`dim()` 告诉你一共几维——这就是张量的"尺码标签"。

# 0 维张量（标量）
scalar = torch.tensor(5)
print(f"标量: {scalar}, shape: {scalar.shape}, dim: {scalar.dim()}")

# 1 维张量（向量）
vector = torch.tensor([1, 2, 3])
print(f"向量: {vector}, shape: {vector.shape}, dim: {vector.dim()}")

# 2 维张量（矩阵）
matrix = torch.tensor([[1, 2], [3, 4]])
print(f"矩阵:\n{matrix}, shape: {matrix.shape}, dim: {matrix.dim()}")

# 3 维张量（比如 2 张 2×2 的图片）
tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3维张量:\n{tensor3d}, shape: {tensor3d.shape}, dim: {tensor3d.dim()}")

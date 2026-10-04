import torch

# ========== 第 1 题: 在 CUDA 上计算 z = x² + y ==========
# Step 1: 创建 x 和 y (CPU 上, FP32, 形状 (1000,), x 全为2.0, y 全为3.0)
x = torch.full((1000,), 2.0, dtype=torch.float32)
y = torch.full((1000,), 3.0, dtype=torch.float32)

# Step 2: 搬到 GPU(CUDA)
x_cuda = x.cuda()
y_cuda = y.cuda()

# Step 3: 在 CUDA 上计算 z = x² + y
z_cuda = torch.pow(x_cuda, 2) + y_cuda

# Step 4: 搬回 CPU
z = z_cuda.cpu()

# 验证
expected = torch.full((1000,), 7.0)
print(f"z 的前 5 个元素: {z[:5].tolist()}")
print(f"z 的设备: {z_cuda.device}")
print(f"验证结果: {'√ 通过' if torch.allclose(z, expected) else 'X 失败'}")

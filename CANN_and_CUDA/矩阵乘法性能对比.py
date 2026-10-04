import time

import torch

# ========== 第 2 题（进阶）：CPU vs CUDA 矩阵乘法性能对比 ==========
# 原题目：NPU 矩阵乘法性能对比，改造为本地 RTX CUDA 版本

sizes = [128, 512, 2048]

print(
    f"{'规模':>8} | {'CPU 耗时':>12} | {'CUDA 耗时':>12} | {'加速比':>8} | {'结果一致':>8}"
)
print("-" * 65)

for N in sizes:
    # Step 1: 在 CPU 上创建两个 N×N 随机矩阵
    a_cpu = torch.randn(N, N, dtype=torch.float32)
    b_cpu = torch.randn(N, N, dtype=torch.float32)

    # Step 2: CPU 矩阵乘法计时
    t0_cpu = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    t_cpu = time.time() - t0_cpu

    # Step 3: 把数据拷贝到 CUDA GPU
    a_cuda = a_cpu.cuda()
    b_cuda = b_cpu.cuda()

    # ---------- WARMUP 预热 ----------
    # 第一次运行会触发 CUDA kernel 编译/缓存，开销很大
    # 所以正式测速前先跑一遍，并等待它完成
    c_cuda_warm = torch.matmul(a_cuda, b_cuda)
    torch.cuda.synchronize()

    # 正式计时开始前，再同步一次，确保 GPU 空闲
    # 多次循环，取平均，避免时间精度不足得到0
    repeat = 20
    torch.cuda.synchronize()
    t0_cuda = time.time()
    for _ in range(repeat):
        c_cuda = torch.matmul(a_cuda, b_cuda)
    torch.cuda.synchronize()
    t_cuda = (time.time() - t0_cuda) / repeat

    # Step 4：验证 CUDA 结果是否和 CPU 基准一致
    # 注意：CUDA 结果要先 .cpu() 搬回 CPU，才能和 c_cpu 比较
    consistent = torch.allclose(c_cpu, c_cuda.cpu(), atol=1e-1)

    # 打印结果
    speedup = t_cpu / t_cuda
    tag = "√" if consistent else "X"
    print(
        f"{N:>8} | {t_cpu * 1000:>10.2f}ms | {t_cuda * 1000:>10.2f}ms | {speedup:>7.1f}x | {tag:>8}"
    )

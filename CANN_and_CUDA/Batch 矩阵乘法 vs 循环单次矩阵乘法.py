import time
import torch

# ========== 第3题：Batch矩阵乘法 vs 循环单次矩阵乘法（CUDA本地版本） ==========
batch_sizes = [8, 32, 128]
M, K, N = 256, 256, 256
reps = 20

print(
    f"{'B':>6} | {'bmm 耗时':>12} | {'循环 耗时':>12} | {'加速比':>8} | {'结果一致':>8}"
)
print("-" * 65)

for B in batch_sizes:
    # Step1: 创建 (B, M, K) 和 (B, K, N) 随机FP32矩阵，搬到GPU
    a = torch.randn(B, M, K, dtype=torch.float32)
    b = torch.randn(B, K, N, dtype=torch.float32)

    # Step2: Warmup预热，提前触发kernel编译
    c_bmm_warm = torch.bmm(a, b)
    c_loop_warm = torch.stack([torch.matmul(a[i], b[i]) for i in range(B)])
    torch.cuda.synchronize()

    # Step3: torch.bmm 批量矩阵乘法计时
    torch.cuda.synchronize()
    t0_bmm = time.time()
    for _ in range(reps):
        c_bmm = torch.bmm(a, b)
    torch.cuda.synchronize()
    t_bmm = (time.time() - t0_bmm) / reps * 1000

    # Step4: 循环matmul逐个矩阵乘法计时
    torch.cuda.synchronize()
    t0_loop = time.time()
    for _ in range(reps):
        c_loop = torch.stack([torch.matmul(a[i], b[i]) for i in range(B)])
    torch.cuda.synchronize()
    t_loop = (time.time() - t0_loop) / reps * 1000

    # Step5: 验证两种计算结果一致
    consistent = torch.allclose(c_bmm, c_loop, atol=1e-1)

    # 计算加速比 + 打印
    speedup = t_loop / t_bmm
    tag = "√" if consistent else "X"
    print(
        f"{B:>6} | {t_bmm:>10.2f}ms | {t_loop:>10.2f}ms | {speedup:>7.1f}x | {tag:>8}"
    )

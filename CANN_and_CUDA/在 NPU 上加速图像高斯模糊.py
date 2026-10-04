import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

# ========== 第4题：GPU上加速图像高斯模糊 ==========
# Step1: 加载图片转为张量 [1, 3, H, W]
img = Image.open("./lena.jpg").convert("RGB")
img_tensor = (
    torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
)
print(f"图片张量形状: {img_tensor.shape}")

# 显示输入图片
plt.figure(figsize=(6, 5))
plt.imshow(img)
plt.title("Input: Lena (512×512)")
plt.axis("off")
plt.show()

# Step2: 构造5×5高斯卷积核
size, sigma = 5, 1.0
coords = torch.arange(size, dtype=torch.float32) - size // 2
g1d = torch.exp(-(coords**2) / (2 * sigma**2))
kernel2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)
kernel2d = kernel2d / kernel2d.sum()
weight = kernel2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)  # [3, 1, 5, 5]

# ========= TODO3 CPU卷积计时 =========
t0_cpu = time.time()
blur_cpu = F.conv2d(img_tensor, weight, padding=2, groups=3)
cpu_time = time.time() - t0_cpu

# ========= TODO4 CUDA卷积计时 =========
# 数据搬到GPU
img_cuda = img_tensor.cuda()
weight_cuda = weight.cuda()

# warmup预热
_ = F.conv2d(img_cuda, weight_cuda, padding=2, groups=3)
torch.cuda.synchronize()

# 正式计时
torch.cuda.synchronize()
t0_cuda = time.time()
blur_cuda = F.conv2d(img_cuda, weight_cuda, padding=2, groups=3)
torch.cuda.synchronize()
cuda_time = time.time() - t0_cuda

# 结果搬回CPU
result = blur_cuda.cpu()

# Step5 打印耗时与加速比
print(f"CPU 卷积耗时: {cpu_time * 1000:.2f} ms")
print(f"GPU 卷积耗时: {cuda_time * 1000:.2f} ms")
print(f"加速比: {cpu_time / cuda_time:.1f}x")

# Step6 显示输出图片
blur_img = Image.fromarray(
    (result.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
)
plt.figure(figsize=(6, 5))
plt.imshow(blur_img)
plt.title(f"Gaussian Blur (GPU, {cuda_time * 1000:.1f} ms)")
plt.axis("off")
plt.show()

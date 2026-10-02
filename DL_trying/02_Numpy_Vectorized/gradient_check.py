import numpy as np


def gradient_check(nn, X, Y, epsilon=1e-7):
    print("--- 执行梯度检验 ---")
    layer_tuple = nn.layers[0]
    layer = layer_tuple[0]
    num_grad = np.zeros_like(layer.W)

    # 计算数值梯度
    it = np.nditer(layer.W, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old_val = layer.W[idx]
        layer.W[idx] = old_val + epsilon
        loss_plus = nn.loss_fn.forward(nn.forward(X), Y)
        layer.W[idx] = old_val - epsilon
        loss_minus = nn.loss_fn.forward(nn.forward(X), Y)
        num_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        layer.W[idx] = old_val
        it.iternext()

    # 计算解析梯度
    nn.forward(X)
    nn.backward(Y)
    analytic_grad = layer.dW

    diff = np.linalg.norm(num_grad - analytic_grad) / (
        np.linalg.norm(num_grad) + np.linalg.norm(analytic_grad) + 1e-8
    )
    print(f"相对误差: {diff:.10e}")
    if diff < 1e-6:
        print("✅ 梯度检验通过！")
    else:
        print("❌ 梯度检验失败，请检查反向传播公式！")

import numpy as np


class ReLU:
    def forward(self, Z):
        # Z shape: [n_out, m]
        self.Z = Z
        self.A = np.maximum(0, Z)
        return self.A

    def backward(self, dA):
        # dA shape: [n_out, m],*这个是逐元素处理
        return dA * (self.Z > 0)


class Sigmoid:
    def forward(self, Z):
        self.Z = Z
        self.A = 1.0 / (1.0 + np.exp(-Z))
        return self.A

    def backward(self, dA):
        return dA * self.A * (1.0 - self.A)


class Tanh:
    def forward(self, Z):
        self.Z = Z
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dA):
        return dA * (1.0 - self.A**2)


class SoftmaxCrossEntropy:
    """Softmax + 交叉熵合并，Z: [n_out, m]，每一列是一个样本的logit"""

    def forward(self, Z, Y):
        max_z = np.max(Z, axis=1, keepdims=True)  # axis=0：每一列（每个样本）取最大值
        exp_z = np.exp(Z - max_z)
        self.A = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        m = Y.shape[0]
        loss = -np.sum(Y * np.log(self.A + 1e-8)) / m
        return loss

    def backward(self, Y):
        # m = Y.shape[0]
        return self.A - Y


def one_hot(y, num_classes=10):
    # np.eye(n)生成N阶单位矩阵
    return np.eye(num_classes)[y]  # 这个需要注重思考一下

import math
import random


class DenseLayer:
    """手搓全连接层"""

    def __init__(self, n_in, n_out, activation="relu"):
        # HE初始化适用于ReLu
        limit_he = math.sqrt(2.0 / n_in)
        # Xavier初始化适用于sigmoid/tanh激活
        limit_xavier = math.sqrt(2.0 / (n_in + n_out))
        if activation == "relu":
            self.W = [
                [random.uniform(-limit_he, limit_he) for _ in range(n_in)]
                for _ in range(n_out)
            ]
        elif activation == "sigmoid" or "tanh":
            self.W = [
                [random.uniform(-limit_xavier, limit_xavier) for _ in range(n_in)]
                for _ in range(n_out)
            ]

        self.B = [0.0] * n_out
        self.activation = activation
        self.inputs = None
        self.Z = None
        self.A = None

    def forword(self, X):
        self.inputs = X
        self.Z = [
            sum(X[j] * self.W[i][j] for j in range(len(X))) + self.B[i]
            for i in range(len(self.B))
        ]

        # 激活函数
        if self.activation == "relu":
            self.A = [max(0.0, z) for z in self.Z]
        elif self.activation == "sigmoid":
            self.A = [1.0 / (1 + math.exp(-z)) for z in self.Z]
        elif self.activation == "softmax":
            # 这里的实现和softmax的原本数学形式不太一样，z-max_z,避免溢出的问题
            max_z = max(self.Z)
            exp_z = [math.exp(z - max_z) for z in self.Z]
            sum_exp = sum(exp_z)
            self.A = [e / sum_exp for e in exp_z]

        return self.A

    def backword(self, dA):
        """注意链式法则,dZ是指L对于Z的偏导,利用的链式法则,并且这里最后的损失函数单独写在训练中"""
        dZ = [0.0] * len(self.Z)
        for i in range(len(self.Z)):
            if self.activation == "relu":
                dZ[i] = dA[i] * 1 if self.Z[i] > 0 else 0.0
            elif self.activation == "sofmax":
                dZ[i]
            elif self.activation == "softmax":
                # Softmax单独激活的雅可比矩阵，dZ = dA @ J_softmax
                # J_ij = a_i*(δ_ij - a_j)
                a_i = self.A[i]
                s = 0.0
                for k in range(len(self.Z)):
                    delta = 1.0 if k == i else 0.0
                    a_k = self.A[k]
                    s += dA[k] * a_k * (delta - a_i)
                dZ[i] = s

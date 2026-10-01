import math
import random

"""先行解释，这个文件的是实现了雅可比完整softmax，单独对softmax求导，这不是主流的方法"""


class DenseLayer:
    """手搓全连接层"""

    """【雅可比版本】全连接层：Softmax单独求导，完整雅可比矩阵实现"""

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
        elif activation in ["softmax", "sigmoid", "tanh"]:
            self.W = [
                [random.uniform(-limit_xavier, limit_xavier) for _ in range(n_in)]
                for _ in range(n_out)
            ]

        self.b = [0.0] * n_out
        self.activation = activation
        self.inputs = None
        self.Z = None
        self.A = None

    def forward(self, X):
        self.inputs = X
        self.Z = [
            sum(X[j] * self.W[i][j] for j in range(len(X))) + self.b[i]
            for i in range(len(self.b))
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

    def backward(self, dA):
        """注意链式法则,dZ是指L对于Z的偏导,利用的链式法则,并且这里最后的损失函数单独写在训练中"""
        dZ = [0.0] * len(self.Z)

        for i in range(len(self.Z)):
            if self.activation == "relu":
                dZ[i] = dA[i] * 1 if self.Z[i] > 0 else 0.0
            elif self.activation == "sigmoid":
                da_z = self.A[i]
                dZ[i] = dA[i] * da_z * (1 - da_z)
            elif self.activation == "softmax":
                s = 0.0
                for j in range(len(self.Z)):
                    alpha = 1.0 if j == i else 0.0
                    s += dA[j] * self.A[j] * (alpha - self.A[i])
                dZ[i] = s

        self.dW = [
            [self.inputs[i] * dZ[j] for i in range(len(self.inputs))]
            for j in range(len(self.Z))
        ]

        self.db = dZ

        dA_prev = [
            sum(self.W[i][j] * dZ[i] for i in range(len(dZ)))
            for j in range(len(self.inputs))
        ]

        return dA_prev

    def update(self, lr):
        for i in range(len(self.W)):
            for j in range(len(self.W[0])):
                self.W[i][j] -= lr * self.dW[i][j]
        for j in range(len(self.b)):
            self.b[j] -= lr * self.db[j]

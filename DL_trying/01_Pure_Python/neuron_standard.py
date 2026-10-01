import math
import random

"""先行解释，这个文件是【主流工程简化版本】
Softmax + 交叉熵损失合并求导，导数在train.py提前算出dZ=A-Y，直接传入backward，层内不再做雅可比计算
也就是PyTorch CrossEntropyLoss底层思路
"""


class DenseLayer:
    """
    W形状： [n_out][n_in]
    Z[i]：第i个输出神经元线性结果 Z_i = sum_j X_j * W[i][j] + b[i]
    """

    """【简化工程版本】Softmax+交叉熵合并求导，无雅可比矩阵循环"""

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
        # Z[i] = sum_{j} X[j] * W[i][j] + b[i]  和jacobi完全相同
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
            # z-max_z，数值稳定，防止指数溢出
            max_z = max(self.Z)
            exp_z = [math.exp(z - max_z) for z in self.Z]
            sum_exp = sum(exp_z)
            self.A = [e / sum_exp for e in exp_z]

        return self.A

    def backward(self, dA):
        """
        ⚠️【关键区分，和雅可比版本最大区别】
        当 activation == "softmax"：
            传入的参数名字虽然叫dA，但**实际是 dZ = ∂L/∂Z = A-Y**
            不再传入 ∂L/∂A，softmax导数提前在train.py合并化简完成
        当 activation == relu/sigmoid：
            dA含义仍然是 ∂L/∂A，逻辑和雅可比版本完全一样
        """
        dZ = [0.0] * len(self.Z)

        for i in range(len(self.Z)):
            if self.activation == "relu":
                dZ[i] = dA[i] * 1 if self.Z[i] > 0 else 0.0
            elif self.activation == "sigmoid":
                da_z = self.A[i]
                dZ[i] = dA[i] * da_z * (1 - da_z)
            elif self.activation == "softmax":
                # 外部train.py已经算出dZ=A-Y，直接赋值，去掉雅可比双层循环
                dZ[i] = dA[i]

        # dW[i][j] = inputs[j] * dZ[i]，和jacobi版本公式、下标完全一致
        self.dW = [
            [self.inputs[j] * dZ[i] for j in range(len(self.inputs))]
            for i in range(len(self.Z))
        ]

        self.db = dZ

        # dA_prev[j] = sum_i W[i][j] * dZ[i] 传回上一层梯度，完全相同
        dA_prev = [
            sum(self.W[i][j] * dZ[i] for i in range(len(dZ)))
            for j in range(len(self.inputs))
        ]

        return dA_prev

    def update(self, lr):
        for i in range(len(self.W)):
            for j in range(len(self.W[0])):
                self.W[i][j] -= lr * self.dW[i][j]
        for i in range(len(self.b)):
            self.b[i] -= lr * self.db[i]

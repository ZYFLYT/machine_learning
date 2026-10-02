from layers import DenseLayer
from utils import ReLU, SoftmaxCrossEntropy


class NumpyNN:
    def __init__(self, layer_dims):
        """layer_dims是依次记录网络每一层的神经元数量的列表"""
        self.layers = []
        # 构建隐藏层，输出层单独拿出来
        for i in range(len(layer_dims) - 2):
            dense = DenseLayer(layer_dims[i], layer_dims[i + 1])
            act = ReLU()  # 每一层单独新建ReLU，独立保存自己的Z
            self.layers.append((dense, act))
        # 输出层
        self.out_layer = DenseLayer(layer_dims[-2], layer_dims[-1])
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, X):
        A = X
        for dense, act in self.layers:
            Z = dense.forward(A)
            A = act.forward(Z)
        return self.out_layer.forward(A)

    def backward(self, Y):
        dZ_out = self.loss_fn.backward(Y)
        dA = self.out_layer.backward(dZ_out)
        for dense, act in reversed(self.layers):
            dZ = act.backward(dA)
            dA = dense.backward(dZ)

    def update(self, lr):
        self.out_layer.W -= lr * self.out_layer.dW
        self.out_layer.b -= lr * self.out_layer.db
        for dense, _ in self.layers:
            dense.W -= lr * dense.dW
            dense.b -= lr * dense.db

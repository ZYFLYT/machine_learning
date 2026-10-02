import numpy as np


class DenseLayer:
    def __init__(self, n_in, n_out):
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / (n_in + n_out))
        self.b = np.zeros((1, n_out))

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(A_prev, self.W) + self.b  # 或者@
        return self.Z

    def backward(self, dZ):
        m = self.A_prev.shape[0]
        self.dW = 1 / m * np.dot(self.A_prev.T, dZ)
        self.db = 1 / m * np.sum(dZ, axis=0, keepdims=True)
        return np.dot(dZ, self.W.T)

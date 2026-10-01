import math
import time

from data_loader import load_data
from neuron_standard import DenseLayer


class PureNN:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.hidden = DenseLayer(64, 64, activation="relu")
        self.output = DenseLayer(64, 10, activation="softmax")

    def forward(self, x):
        h_out = self.hidden.forward(x)
        output = self.output.forward(h_out)
        return output

    def backward(self, y_true, y_pred):
        # 输出层梯度 dZ=A-Y
        dZ = list(y_pred)
        dZ[y_true] -= 1.0
        dA_hidden = self.output.backward(dZ)
        self.hidden.backward(dA_hidden)

    def update(self):
        self.hidden.update(self.lr)
        self.output.update(self.lr)


if __name__ == "__main__":
    X, Y = load_data()
    nn = PureNN(lr=0.01)
    print("【主流简化softmax版本】开始训练")

    train_start = time.time()
    for epoch in range(500):
        epoch_start = time.time()
        total_loss = 0
        correct = 0
        for i in range(len(X)):
            out = nn.forward(X[i])
            loss = -math.log(out[Y[i]] + 1e-8)
            total_loss += loss
            if out.index(max(out)) == Y[i]:
                correct += 1
            nn.backward(Y[i], out)
            nn.update()
        epoch_end = time.time()

        if epoch % 100 == 0:
            avg_loss = total_loss / len(X)
            acc = correct / len(X) * 100
            epoch_cost = epoch_end - epoch_start
            print(
                f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc:{acc:.2f}% | Epoch耗时: {epoch_cost:.4f}s"
            )

    train_end = time.time()
    total_train_time = train_end - train_start
    print(f"\n✅ 简化版本训练全部完成，总训练耗时：{total_train_time:.4f} s")

    import pickle

    with open("model_standard.pkl", "wb") as f:
        pickle.dump(nn, f)
    print("简化版本训练完成，保存 model_standard.pkl")

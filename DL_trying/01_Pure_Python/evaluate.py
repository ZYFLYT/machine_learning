import pickle
import time

from data_loader import load_data
from train_jacobi import PurnNN
from train_standard import PureNN  # 标准版本的网络类

import __main__

# 关键！把类挂到__main__，匹配pickle保存时的记录
__main__.PurnNN = PurnNN
__main__.PureNN = PureNN


def evaluate(model_path):
    X, Y = load_data()
    with open(model_path, "rb") as f:
        nn = pickle.load(f)

    start_time = time.time()
    correct = 0
    for i in range(len(X)):
        pred = nn.forward(X[i])
        pred_idx = pred.index(max(pred))
        if pred_idx == Y[i]:
            correct += 1
    end_time = time.time()

    acc = correct / len(X) * 100
    cost = end_time - start_time
    print(f"模型 {model_path}")
    print(f"  准确率：{acc:.2f}%")
    print(f"  推理耗时：{cost:.4f} s\n")


if __name__ == "__main__":
    evaluate("model_jacobi.pkl")
    evaluate("model_standard.pkl")

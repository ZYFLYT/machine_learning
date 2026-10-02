import json
import matplotlib.pyplot as plt


def plot_curves(history_path="history.json"):
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))
    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    plt.title("Learning Curve (Loss)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    plt.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    plt.title("Learning Curve (Accuracy)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("learning_curve.png")
    print("学习曲线已保存为 learning_curve.png")

    # 吴恩达课程诊断：看是否过拟合
    if history["train_loss"][-1] < history["val_loss"][-1] * 0.5:
        print("诊断：可能存在高方差（过拟合）。建议增大 Dropout 或使用 L2 正则化。")
    else:
        print("诊断：模型拟合良好，未出现严重过拟合。")


if __name__ == "__main__":
    plot_curves()

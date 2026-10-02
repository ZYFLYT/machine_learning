import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import MLP


def run_training(epochs=10, lr=0.001, dropout_prob=0.25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders()

    model = MLP(dropout_prob).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss, train_correct = 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += output.argmax(1).eq(target).sum().item()

        # 验证阶段
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                val_correct += output.argmax(1).eq(target).sum().item()

        # 记录
        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["train_acc"].append(train_correct / 50000)
        history["val_acc"].append(val_correct / 10000)

        print(
            f"Epoch {epoch + 1}: Train Loss: {history['train_loss'][-1]:.4f}, Val Loss: {history['val_loss'][-1]:.4f}, Val Acc: {history['val_acc'][-1] * 100:.2f}%"
        )

    # 测试集最终评估
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_correct += output.argmax(1).eq(target).sum().item()
    print(f"\n最终测试集准确率: {test_correct / 10000 * 100:.2f}%")

    # 保存训练历史以便绘图
    import json

    with open("history.json", "w") as f:
        json.dump(history, f)
    torch.save(model.state_dict(), "mlp_model.pth")


if __name__ == "__main__":
    run_training()

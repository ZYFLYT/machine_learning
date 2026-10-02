# dataset.py

# DataLoader：批量迭代；random_split：数据集随机划分
from torch.utils.data import DataLoader, random_split

# 内置CV数据集、图像预处理工具
from torchvision import datasets, transforms


def get_dataloaders(batch_size=64):
    """
    构建MNIST数据集的DataLoader，拆分训练集/验证集/测试集
    :param batch_size: 训练时每个批次样本数量，默认64，和前面Numpy版本保持一致
    :return: train_loader, val_loader, test_loader
             train_loader: 训练集迭代器
             val_loader:   验证集迭代器（用于观察过拟合、调超参）
             test_loader:  测试集迭代器（最终模型评估，训练全程不能用来调参）
    """
    # 预处理流水线，多个操作按顺序执行
    transform = transforms.Compose(
        [
            # ToTensor：PIL图片 -> torch张量；像素从0~255自动缩放到0~1
            transforms.ToTensor(),
            # 标准化 x_norm = (x - mean) / std
            # (0.1307,) MNIST全局像素均值；(0.3081,) MNIST全局像素标准差
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # 加载MNIST完整原始训练集，共60000张手写数字图片
    # train=True代表使用官方训练划分；download=True不存在就自动下载
    full_train = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )

    # 将60000张训练图片随机切分：50000训练集，10000验证集
    # 吴恩达规范：训练集、验证集必须分开，验证集不参与权重更新
    train_set, val_set = random_split(full_train, [50000, 10000])

    # 加载MNIST官方测试集，固定10000张，训练阶段不能使用
    test_set = datasets.MNIST("./data", train=False, transform=transform)

    # 训练集DataLoader，每个epoch前打乱样本，防止模型记住样本顺序
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # 验证集：不需要打乱，batch调大加速评估
    val_loader = DataLoader(val_set, batch_size=1000, shuffle=False)
    # 测试集：不需要打乱，仅最后一次性评估模型泛化能力
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    return train_loader, val_loader, test_loader

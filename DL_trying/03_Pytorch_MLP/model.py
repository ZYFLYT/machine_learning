import torch.nn as nn

# F是函数式接口，存放relu等无参数的运算函数，直接在forward中调用
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dropout_prob=0.25):
        """
        构造函数：只用来定义网络层，**不做张量计算**
        :param dropout_prob: Dropout丢弃概率，训练阶段每个神经元输出被置0的概率
        """
        # 调用父类nn.Module的构造函数，初始化父类内置参数容器，必须写，否则无法管理网络权重
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # Dropout层1，放在第一层隐藏层之后
        # 训练模式：每个神经元输出有dropout_prob概率置0；评估模式自动失效
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        前向传播函数：定义数据流过网络的计算逻辑
        注意：不要手动调用forward()，执行model(x)时PyTorch内部自动调用，同时构建计算图用于反向求导
        :param x: 输入张量，来自DataLoader，原始shape [batch, channel=1, height=28, width=28]
        :return: logits，张量shape [batch, 10]，各类别原始得分，不经过softmax
        """
        # x.view(-1,784)张量重塑展平
        # -1代表自动推导batch维度；把4维图片张量 [batch,1,28,28]转为2维 [batch,784]
        # 全连接层只能接收一维特征向量，不能直接处理28×28图像矩阵
        x = x.view(-1, 784)
        # 第一步：送入第一层全连接做矩阵运算 Z = XW^T + b；然后使用ReLU激活
        # ReLU：所有负数置0，正数保持原值，引入非线性，网络才能拟合复杂关系
        x = F.relu(self.fc1(x))
        # 第一层dropout，训练时随机屏蔽25%神经元输出，抑制神经元间依赖，防止过拟合
        # 保留的输出会自动 × 1/(1-p)，保证训练与评估阶段数值期望一致
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        # 输出层，得到10个类别的logits原始得分，不添加softmax
        # 训练使用nn.CrossEntropyLoss，内部自带LogSoftmax，额外加softmax会造成数值不稳定
        return self.fc3(x)

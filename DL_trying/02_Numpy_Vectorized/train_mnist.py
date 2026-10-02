import time  # 导入time模块，用于计时

import numpy as np
from gradient_check import gradient_check
from network import NumpyNN
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from utils import one_hot

if __name__ == "__main__":
    # ===================== 数据集加载部分 =====================
    print("加载 MNIST 数据集...")
    # 加载MNIST手写数字数据集，70000张28×28手写图片
    # version=1 固定数据集版本；parser='auto'自动解析数据格式
    mnist = fetch_openml("mnist_784", version=1, parser="auto")

    # mnist.data是pandas表格，to_numpy()转为numpy数组
    # 原始像素范围0~255，除以255.0归一化到0~1
    # 归一化目的：控制数值范围，防止矩阵相乘后数值爆炸，提升训练稳定性
    X = mnist.data.to_numpy() / 255.0
    # 读取标签，转为numpy整数数组，Y里面保存原始数字0~9，不是one-hot编码
    Y = mnist.target.to_numpy().astype(int)

    # 划分数据集：全部7万张图片拆分为训练集、测试集
    # test_size=10000：拿出10000张作为测试集，剩余60000张作为训练集
    # random_state=42固定随机种子，保证每次运行划分结果完全一致，方便复现实验
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=10000, random_state=42
    )

    # 将原始数字标签转为one-hot矩阵，shape (样本数量,10)
    # 每一行：真实类别位置为1，其余位置为0；loss_fn损失函数要求输入one-hot标签
    Y_train_oh = one_hot(Y_train)
    Y_test_oh = one_hot(Y_test)

    # ===================== 构建网络 =====================
    # 网络结构：784输入像素 → 隐藏层1(128神经元,Dense+ReLU) → 隐藏层2(64神经元,Dense+ReLU) →输出层10神经元(Dense输出logits)
    model = NumpyNN([784, 128, 64, 10])

    # ===================== 梯度检验（调试阶段使用，正式训练注释！） =====================
    # 梯度检验：对比【手写反向传播解析梯度】和【数值微分算出的数值梯度】，验证反向传播代码正确性
    # 只取前100样本，梯度检验需要逐个扰动权重、多次前向传播，全量样本速度极慢
    print("\n开始执行梯度检验，用来校验反向传播代码是否存在bug...")
    gradient_check(model, X_train[:100], Y_train_oh[:100])
    print("梯度检验完成，准备开始训练\n")

    # ===================== 训练超参数定义 =====================
    # batch_size：一个批次样本数量，mini-batch梯度下降，每次取64张图片更新权重
    # lr：学习率，控制每次参数更新步长；太大震荡不收敛，太小训练缓慢
    # epochs：训练轮数，1个epoch代表完整遍历一遍全部6万训练样本
    batch_size, lr, epochs = 64, 0.1, 10

    # 记录整体训练开始时间
    train_start_time = time.time()

    # ===================== Epoch循环，多轮训练 =====================
    for epoch in range(epochs):
        epoch_start_time = time.time()  # 记录当前epoch开始时间

        # np.random.permutation：生成0~59999的随机打乱序号
        # 每轮epoch打乱样本顺序，防止模型记住样本顺序，提升泛化能力
        indices = np.random.permutation(X_train.shape[0])
        # 使用打乱后的序号，同步打乱图片和onehot标签，保证图片和标签一一对应
        X_shuf, Y_shuf = X_train[indices], Y_train_oh[indices]

        epoch_loss = 0  # 初始化本轮总损失，用于累加所有batch损失

        # 计算batch总数量，//整数除法，丢弃不足一个batch的剩余样本，简化代码
        batch_num = X_train.shape[0] // batch_size
        for i in range(batch_num):
            # 切片取出第i个batch的数据
            X_batch = X_shuf[i * batch_size : (i + 1) * batch_size]
            Y_batch = Y_shuf[i * batch_size : (i + 1) * batch_size]

            # 前向传播：输入batch图片，返回输出层logits（没有经过softmax）
            Z_out = model.forward(X_batch)
            # 损失函数前向：内部执行softmax + 计算交叉熵损失；累加当前batch损失到总损失
            epoch_loss += model.loss_fn.forward(Z_out, Y_batch)

            # 反向传播：链式求导，从损失向后计算每一层dW、db，梯度保存在层内
            # 注意：必须先forward完成，再执行backward，顺序不能颠倒
            model.backward(Y_batch)

            # 参数更新：梯度下降 W = W - lr*dW，b = b - lr*db
            # 每跑完一个batch，更新一次权重
            model.update(lr)

        # -------------------- 本轮训练结束，评估测试集 --------------------
        # 在测试集跑前向传播，测试集只做预测，不反向传播、不更新权重
        test_Z = model.forward(X_test)
        # np.argmax(test_Z, axis=1)：每行找最大值下标，就是模型预测的数字
        # == Y_test：预测值和真实标签对比，True=1，False=0；np.mean求平均值即为准确率
        test_acc = np.mean(np.argmax(test_Z, axis=1) == Y_test)

        # 计算当前epoch耗时
        epoch_cost = time.time() - epoch_start_time
        # 计算本轮平均损失：总损失 / batch数量
        avg_loss = epoch_loss / batch_num
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Test Acc: {test_acc * 100:.2f}%, Epoch耗时: {epoch_cost:.2f}s"
        )

    # ===================== 全部epoch训练结束 =====================
    total_train_time = time.time() - train_start_time
    print(f"\n✅ 全部训练完成！总训练耗时：{total_train_time:.2f} 秒")

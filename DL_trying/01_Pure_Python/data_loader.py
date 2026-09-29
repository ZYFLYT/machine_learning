from sklearn.datasets import load_digits


def load_data():
    """加载8x8手写数字数据集，返回列表格式以适配纯Python"""
    digits = load_digits()
    X = digits.data / 16.0
    Y = digits.target
    # 转为普通的Python List
    return X.tolist(), Y.tolist()

import numpy as np


def init_network():  # 重みとバイアスの初期化
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network  # ディクショナリを返却


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)  # 活性化関数としてsigmoid関数を使う
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)  # 活性化関数としてsigmoid関数を使う
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)  # 活性化関数として恒等関数を使う

    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_function(x):
    return x


print("伝達を開始する")
network = init_network()  # 重みとバイアスの初期化
x = np.array([1.0, 0.5])  # 最初の入力
y = forward(network, x)
print(y)

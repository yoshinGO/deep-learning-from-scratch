import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_function(x):  # 恒等関数の
    return x


print("入力層から第１層への信号の伝達")
X = np.array([1.0, 0.5])  # 入力
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # 重み
B1 = np.array([0.1, 0.2, 0.3])  # バイアス

print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)  # 活性化関数としてsigmoid関数を使う

print(A1)
print(Z1)

print("\n第１層から第２層への信号の伝達")
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)  # 活性化関数としてsigmoid関数を使う

print(A2)
print(Z2)

print("\n第２層から出力層への信号の伝達")
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)  # 活性化関数として恒等関数を使う

print(Y)

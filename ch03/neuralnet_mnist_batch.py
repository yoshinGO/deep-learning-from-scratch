# ニューラルネットワークの推論処理
# 入力層を784個、出力層を10個のニューロンでネットワークを構築

import sys
import os
sys.path.append(os.pardir)
import pickle
import numpy as np
from dataset.mnist import load_mnist


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c)  # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test  # 学習済みの重みパラメータを使うから訓練用のデータは必要ない


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network  # 今回は学習済みのデータを使う


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)  # 活性化関数としてsigmoid関数を使う
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)  # 活性化関数としてsigmoid関数を使う
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)  # 活性化関数としてソフトマックス関数を使う(分類問題であるため)

    return y


print("mnistデータセットの分類を行います。(バッチ処理ver)")
x, t = get_data()  # 学習済みの重みパラメータを使うから訓練用のデータは必要ない
network = init_network()

batch_size = 100  # バッチの数
accuracy_cnt = 0

for i in range(0, len(x), batch_size):  # iに100毎ずつを入れてループさせる
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # 正解率

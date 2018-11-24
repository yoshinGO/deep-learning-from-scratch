import sys
import os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]  # 60000
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
#  コンストラクタが重みとバイアスの初期化を行なってくれる

for i in range(iters_num):  # 10000回繰り返す
    # ミニバッチの取得
    # 0~59999の中からランダムに100個選ばれた数字がbatch_maskに格納される
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算, gradには損失関数の平均を減らすために必要な情報が格納される
    grad = network.numerical_gradient(x_batch, t_batch)

    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 学習経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

for i in range(len(train_loss_list), step=10000):
    print(train_loss_list[i])

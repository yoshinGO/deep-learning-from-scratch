import sys
import os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)  # (60000, 784) 画像
print(t_train.shape)  # (60000, 10) ラベル

train_size = x_train.shape[0]  # 60000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
# すなわち0~60000未満の数字の中からランダムに10個の数字を選び出し、batch_maskと命名する。
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

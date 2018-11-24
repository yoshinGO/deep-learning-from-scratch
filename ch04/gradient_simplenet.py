import sys
import os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # ガウス分布で初期化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)  # 演算
        y = softmax(z)  # 活性化関数としてソフトマックス関数を用いて確率を求める
        loss = cross_entropy_error(y, t)  # 損失関数の平均値

        return loss

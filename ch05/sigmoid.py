import numpy as np


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out  # 出力を保持

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

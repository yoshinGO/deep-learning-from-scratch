class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 損失
        self.y = None  # softmax関数の出力
        self.t = None  # 教師データ（one-hot vector）

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)  # 確率
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size  # データが100個あれば(100, n)の行列

        return dx

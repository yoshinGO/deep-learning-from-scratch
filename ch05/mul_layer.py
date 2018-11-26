class MulLayer:
    def __init__(self):
        #  順伝播時の入力値を保持するために用いるx, yを初期化
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y  # xとyをひっくり返す
        dy = dout * self.x

        return dx, dy

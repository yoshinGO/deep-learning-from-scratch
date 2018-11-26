class AddLayer:
    def __init__(self):
        pass  # 逆伝播時に順伝播の入力を使うことはないので、pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy  # 加算レイヤでは上流から伝わってきた微分をそのまま下流に流すだけ

class Relu:
    def __init__(sef):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return 0

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

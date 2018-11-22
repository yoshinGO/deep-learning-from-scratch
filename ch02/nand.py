import numpy as np


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  # 重みとバイアスだけがANDと違う
    b = 0.7  # バイアス, 発火のしやすさ
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


print(NAND(0, 0))
print(NAND(1, 0))
print(NAND(0, 1))
print(NAND(1, 1))

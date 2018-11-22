import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.8
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.8
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    return_or = OR(x1, x2)
    return_nand = NAND(x1, x2)
    answer = AND(return_or, return_nand)
    return answer


print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))

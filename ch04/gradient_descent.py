import numpy as np
from gradient import numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


def function_2(x):
    return x[0]**2 + x[1]**2


# 勾配法を使って最小値の探索を行う
init_x = np.array([-3.0, 4.0])
final_x = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
print(final_x)  # ほぼ(0, 0)の値が得られる

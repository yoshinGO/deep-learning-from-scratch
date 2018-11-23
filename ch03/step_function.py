import numpy as np
import matplotlib.pylab as plt


def step(x):  # これだと実数の範囲にしか対応しておらずnumpy配列を処理できない
    if x > 0:
        return 0
    else:
        return 0


def step_function(x):
    return np.array(x > 0, dtype=np.int)


x = np.arange(-5.0, 5.0, 0.1)  # -5.0から5.0まで0.1刻みでnumpy配列を定義
y = step_function(x)  # ステップ関数
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # y軸の範囲を指定
plt.show()

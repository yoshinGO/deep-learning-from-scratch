import sys
import os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

# それぞれのデータの形状を出力
print(x_train.shape)  # (60000, 784) 訓練画像
print(t_train.shape)  # (60000) 訓練ラベル
print(x_test.shape)  # (10000, 784) テスト画像
print(t_test.shape)  # (10000) テストラベル

import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('dog.png')  # 画像の読み込み
plt.imshow(img)

plt.show()

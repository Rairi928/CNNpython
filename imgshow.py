import tensorflow as tf
import matplotlib.pyplot as plt #描画用ライブラリ

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train:訓練画像60,000 28x28ピクセル,値0-255,Numpy配列
# y_train:訓練ラベル60,000　画像が表す正解の数字0-9,Numpy配列
# テスト:モデルの実力を測る
# x_test:テスト画像10,000
# y_test:テストラベル10,000

index = 0
index = int(input("表示する画像のインデックスを入力してください (0-59999): "))
plt.imshow(x_train[index], cmap='gray')
plt.title(f"Label: {y_train[index]}")
plt.axis('off')
plt.show()

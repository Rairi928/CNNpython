import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# データの正規化
x_train = x_train / 255.0
x_test = x_test / 255.0

#全結合ニューラルネットワークの構築
model = tf.keras.models.Sequential([
    #28x28ピクセルの画像を1次元のベクトルに変換
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    #128ユニットの全結合層、活性化関数はReLU 
    tf.keras.layers.Dense(128, activation='relu'),
    
    #過学習を防ぐためのドロップアウト層、20%のニューロンをランダムに無効化
    tf.keras.layers.Dropout(0.2),
    
    #10ユニットの全結合層、活性化関数はソフトマックス（多クラス分類用）
    tf.keras.layers.Dense(10, activation='softmax')
])

#モデルのコンパイル
model.compile(
    #最適化アルゴリズム、重みの更新方法
    optimizer='adam',
    #損失関数、間違いの度合いを計算
    loss='sparse_categorical_crossentropy',
    #評価指標、正解率を表示
    metrics=['accuracy']
)

#モデルの訓練
model.fit(x_train, y_train, epochs=5)

#モデルの評価
model.evaluate(x_test, y_test)

#モデルの保存
model.save('mnist_model.h5')

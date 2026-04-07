import tkinter as tk
from PIL import Image, ImageGrab
import tensorflow as tf
import numpy as np

NNmodel = tf.keras.models.load_model('mnist_model.h5')  # 学習済みモデルの読み込み
CNNmodel = tf.keras.models.load_model('mnist_cnn.h5')  # 学習済みCNNモデルの読み込み

CANVAS_SIZE = 280   # 表示用キャンバス（大きく描く）
IMG_SIZE = 28       # 出力画像サイズ（MNISTと同じ）

class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("28x28 手書き入力")

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
        self.canvas.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="クリア", command=self.clear).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="NNによる判定", command=lambda: self.predict(NNmodel)).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="CNNによる判定", command=lambda: self.predict(CNNmodel)).pack(side=tk.LEFT, padx=5)

        self.canvas.bind("<B1-Motion>", self.draw)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8  # ペンの太さ
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")

    def clear(self):
        self.canvas.delete("all")

    def predict(self, model):
        # キャンバスの位置を取得して画像としてキャプチャ
        self.canvas.update()
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + CANVAS_SIZE
        y1 = y + CANVAS_SIZE

        img = ImageGrab.grab((x, y, x1, y1))          # RGB
        img = img.convert("L")                        # グレースケール
        img = img.resize((IMG_SIZE, IMG_SIZE))        # 28x28 に縮小

        img_array = np.array(img) / 255.0            # 正規化
        img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE)  # モデルの入力形状に変換

        # モデルによる予測
        predictions = model.predict(img_array)
        predicted_digit = np.argmax(predictions)
        print(f"予測された数字: {predicted_digit}")
        print(f"予測確率: {predictions}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()
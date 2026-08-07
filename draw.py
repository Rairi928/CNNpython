import tkinter as tk
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ------------------------------------------------------------
# 1. モデル定義（学習時と同じ構造）
# ------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ------------------------------------------------------------
# 2. モデルロード
# ------------------------------------------------------------
device = torch.device("cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("handwritten_cnn_mnist.pth", map_location=device))
model.eval()

# ------------------------------------------------------------
# 3. 画像前処理（MNIST と同じ）
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ------------------------------------------------------------
# 4. Tkinter GUI
# ------------------------------------------------------------
WIDTH = 300
HEIGHT = 300

root = tk.Tk()
root.title("手書き数字判定")

# Canvas を黒背景にする
canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black")
canvas.pack()

# PIL 画像も黒背景で作る
image = Image.new("RGB", (WIDTH, HEIGHT), "black")
draw = ImageDraw.Draw(image)

# ------------------------------------------------------------
# 5. 描画処理（白文字で描く）
# ------------------------------------------------------------
def paint(event):
    x, y = event.x, event.y
    r = 10
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
    draw.ellipse((x-r, y-r, x+r, y+r), fill="white")

canvas.bind("<B1-Motion>", paint)

# ------------------------------------------------------------
# 6. 判定処理（反転なし）
# ------------------------------------------------------------
def predict():
    img = image.convert("L")  # グレースケール化（黒背景・白文字のまま）
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    result_label.config(text=f"判定結果: {pred}")

# ------------------------------------------------------------
# 7. クリア処理（黒背景に戻す）
# ------------------------------------------------------------
def clear():
    canvas.delete("all")
    draw.rectangle((0, 0, WIDTH, HEIGHT), fill="black")
    result_label.config(text="判定結果: ")

# ------------------------------------------------------------
# 8. ボタン
# ------------------------------------------------------------
btn_predict = tk.Button(root, text="判定する", command=predict)
btn_predict.pack()

btn_clear = tk.Button(root, text="クリア", command=clear)
btn_clear.pack()

result_label = tk.Label(root, text="判定結果: ", font=("Arial", 20))
result_label.pack()

root.mainloop()

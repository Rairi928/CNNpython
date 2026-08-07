import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -----------------------------
# 1. データセット読み込み
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# -----------------------------
# 2. 単体画像を表示する関数
# -----------------------------
def show_single(index=0):
    img, label = dataset[index]
    img = img.squeeze(0)  # 1×28×28 → 28×28

    plt.imshow(img, cmap="gray")
    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()

# -----------------------------
# 3. 複数画像を表示する関数
# -----------------------------
def show_multiple(count=10):
    fig = plt.figure(figsize=(10, 4))

    for i in range(count):
        img, label = dataset[i]
        img = img.squeeze(0)

        ax = fig.add_subplot(2, count//2, i+1)
        ax.imshow(img, cmap="gray")
        ax.set_title(label)
        ax.axis("off")

    plt.show()

# -----------------------------
# 4. DataLoader からバッチ表示
# -----------------------------
def show_batch(batch_size=8):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(loader))

    fig = plt.figure(figsize=(10, 4))
    for i in range(batch_size):
        img = images[i].squeeze(0)

        ax = fig.add_subplot(2, batch_size//2, i+1)
        ax.imshow(img, cmap="gray")
        ax.set_title(labels[i].item())
        ax.axis("off")

    plt.show()

# -----------------------------
# 5. 実行部分
# -----------------------------
if __name__ == "__main__":
    print("表示モードを選択してください:")
    print("1: 単体画像を表示")
    print("2: 複数画像を表示")
    print("3: バッチ表示 (DataLoader)")

    mode = input("選択 (1/2/3): ").strip()

    if mode == "1":
        idx = int(input("表示する画像番号 (例: 0): "))
        show_single(idx)

    elif mode == "2":
        count = int(input("表示する枚数 (例: 10): "))
        show_multiple(count)

    elif mode == "3":
        bs = int(input("バッチサイズ (例: 8): "))
        show_batch(bs)

    else:
        print("不正な入力です。終了します。")

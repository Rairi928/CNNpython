import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# =========================
# 1. ハイパーパラメータ
# =========================
batch_size = 64
num_epochs = 5
learning_rate = 1e-3
# CUDA 利用可否はここで確認するが、対話的選択は
# モジュールのインポート時にプロンプトが出ないよう
# __main__ ブロック内で行う。
cuda_available = torch.cuda.is_available()

# =========================
# 2. データセット & DataLoader
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),                 # [0,1] のテンソルに変換
    transforms.Normalize((0.1307,), (0.3081,))  # MNISTの平均・分散で正規化
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# DataLoader / model の初期化は device 選択後に行う（下の __main__ 参照）


# =========================
# 3. モデル定義（シンプルなCNN）
# =========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1x28x28 -> 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 32x14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2)                              # 64x7x7
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

# モデル / 損失 / オプティマイザは実行時に初期化

# =========================
# 5. 学習ループ
# =========================
def train():
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 勾配初期化
            optimizer.zero_grad()

            # 順伝播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 逆伝播 & 更新
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        acc = correct / total * 100
        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {avg_loss:.4f}  Acc: {acc:.2f}%")

# =========================
# 6. 評価ループ
# =========================
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total * 100
    print(f"Test Accuracy: {acc:.2f}%")

# =========================
# 7. 実行
# =========================
if __name__ == "__main__":
    # 対話的にデバイスを選択（インポート時は実行されない）
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 使用可能: {cuda_available}")
    try:
        choice = input("学習デバイスを選択 gpu[g] / cpu[c] (デフォルト: gpu が利用可能なら gpu): ").strip().lower()
    except Exception:
        choice = ""

    if choice == "c":
        device = torch.device("cpu")
    elif choice == "g":
        if cuda_available:
            device = torch.device("cuda")
        else:
            print("CUDA が使用できません。CPU を使用します。")
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if cuda_available else "cpu")

    print(f"使用デバイス: {device}")
    if device.type == "cuda":
        try:
            print(f"CUDA デバイス: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    # DataLoader を device に合わせて初期化
    pin_memory = True if device.type == "cuda" else False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=2)

    # モデル / 損失 / オプティマイザ初期化
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train()
    evaluate()
    # 学習済みモデルを保存
    torch.save(model.state_dict(), "handwritten_cnn_mnist.pth")

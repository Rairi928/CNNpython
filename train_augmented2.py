import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def create_loaders(batch_size):
    # 学習時だけ、数字の回転と拡大縮小をランダムに加える。
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=15,
            scale=(0.9, 1.1),
            fill=0,
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )
    emnist_train_dataset = datasets.EMNIST(
        root="./data",
        split="digits",
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )
    emnist_test_dataset = datasets.EMNIST(
        root="./data",
        split="digits",
        train=False,
        download=True,
        transform=test_transform,
    )

    train_dataset = ConcatDataset([train_dataset, emnist_train_dataset])
    test_dataset = ConcatDataset([test_dataset, emnist_test_dataset])

    pin_memory = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def train_one_epoch(model, device, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def evaluate(model, device, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CNN with rotated and scaled MNIST images"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--output",
        default="handwritten_cnn_mnist_augmented.pth",
        help="path for the new model file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")
    print(f"使用デバイス: {device}")

    train_loader, test_loader = create_loaders(args.batch_size)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss, accuracy = train_one_epoch(
            model, device, train_loader, criterion, optimizer
        )
        test_accuracy = evaluate(model, device, test_loader)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"Loss: {loss:.4f} "
            f"Train Acc: {accuracy:.2f}% "
            f"Test Acc: {test_accuracy:.2f}%"
        )

    torch.save(model.state_dict(), args.output)
    print(f"モデルを保存しました: {args.output}")


if __name__ == "__main__":
    main()

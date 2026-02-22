import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super().__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


def build_googlenet(num_classes=10):
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    b2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    b3 = nn.Sequential(
        Inception(192, 64, (96, 128), (16, 32), 32),
        Inception(256, 128, (128, 192), (32, 96), 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    b4 = nn.Sequential(
        Inception(480, 192, (96, 208), (16, 48), 64),
        Inception(512, 160, (112, 224), (24, 64), 64),
        Inception(512, 128, (128, 256), (24, 64), 64),
        Inception(512, 112, (144, 288), (32, 64), 64),
        Inception(528, 256, (160, 320), (32, 128), 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    b5 = nn.Sequential(
        Inception(832, 256, (160, 320), (32, 128), 128),
        Inception(832, 384, (192, 384), (48, 128), 128),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )
    return nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, num_classes))


def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def run_draw(draw_script, log_path, out_dir, prefix, no_show):
    cmd = [
        sys.executable,
        str(draw_script),
        "--log",
        str(log_path),
        "--out-dir",
        str(out_dir),
        "--prefix",
        prefix,
    ]
    if no_show:
        cmd.append("--no-show")
    subprocess.run(cmd, check=True)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    train_set = datasets.FashionMNIST(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_googlenet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / args.log_name
    model_path = out_dir / args.model_name

    with log_path.open("w", encoding="utf-8") as f:
        for epoch in range(args.num_epochs):
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                batch_size = y.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (logits.argmax(dim=1) == y).sum().item()
                total_samples += batch_size

            train_loss = total_loss / total_samples
            train_acc = total_correct / total_samples
            test_acc = evaluate_accuracy(model, test_loader, device)
            lr = optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "lr": lr,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"Epoch [{epoch + 1}/{args.num_epochs}] "
                f"loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"test_acc={test_acc:.4f} lr={lr:.6f}"
            )

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    print(f"Train log saved to: {log_path}")

    draw_script = Path(__file__).resolve().parents[1] / "draw.py"
    run_draw(draw_script, log_path, out_dir, args.plot_prefix, args.no_show)
    print("Plots generated by draw.py")


def parse_args():
    parser = argparse.ArgumentParser(description="Train GoogLeNet on FashionMNIST.")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./CNN")
    parser.add_argument("--log-name", type=str, default="googlenet_train_log.jsonl")
    parser.add_argument("--model-name", type=str, default="googlenet_fashionmnist.pth")
    parser.add_argument("--plot-prefix", type=str, default="googlenet")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

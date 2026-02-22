import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_jsonl(log_path):
    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No training records found in: {log_path}")
    return rows


def draw_from_log(log_path, out_dir, prefix="train", show=True):
    rows = read_jsonl(log_path)
    epochs = [row["epoch"] for row in rows]

    train_loss = [row.get("train_loss") for row in rows]
    train_acc = [row.get("train_acc") for row in rows]
    test_acc = [row.get("test_acc", row.get("val_acc")) for row in rows]

    out_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, train_loss, label="train_loss", linewidth=2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()
    fig1.tight_layout()
    loss_path = out_dir / f"{prefix}_loss_curve.png"
    fig1.savefig(loss_path, dpi=150)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(epochs, train_acc, label="train_acc", linewidth=2)
    ax2.plot(epochs, test_acc, label="test_acc", linewidth=2)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()
    fig2.tight_layout()
    acc_path = out_dir / f"{prefix}_acc_curve.png"
    fig2.savefig(acc_path, dpi=150)

    print(f"Saved: {loss_path}")
    print(f"Saved: {acc_path}")

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)


def parse_args():
    parser = argparse.ArgumentParser(description="Draw training curves from JSONL log.")
    parser.add_argument("--log", type=str, required=True, help="Path to training jsonl log.")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory.")
    parser.add_argument("--prefix", type=str, default="train", help="Output file name prefix.")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive plot window.")
    return parser.parse_args()


def main():
    args = parse_args()
    draw_from_log(
        log_path=Path(args.log),
        out_dir=Path(args.out_dir),
        prefix=args.prefix,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()

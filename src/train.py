"""Tek bir deneyi eğitir ve sonuçları results/<name>.json içine yazar.

Örnek:
    python src/train.py --name baseline
    python src/train.py --name aug --augment
    python src/train.py --name aug_bn_drop --augment --bn --dropout 0.3
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

from data import get_loaders
from model import SmallCNN


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--bn", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)
    args = p.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}")

    train_loader, test_loader = get_loaders(augment=args.augment)
    model = SmallCNN(use_bn=args.bn, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    history = {"train_loss": [], "test_acc": []}
    best_acc, start = 0.0, time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
        scheduler.step()

        train_loss = running / len(train_loader.dataset)
        acc = evaluate(model, test_loader, device)
        history["train_loss"].append(train_loss)
        history["test_acc"].append(acc)
        print(f"Epoch {epoch:02d}/{args.epochs}  loss={train_loss:.4f}  test_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"models/{args.name}.pt")

    result = {"name": args.name, "args": vars(args), "best_acc": best_acc,
              "seconds": round(time.time() - start), "history": history}
    with open(f"results/{args.name}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Bitti. En iyi doğruluk: {best_acc:.4f}")


if __name__ == "__main__":
    main()

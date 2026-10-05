"""results/*.json dosyalarından karşılaştırma tablosu ve grafik üretir,
ve en iyi modelin confusion matrix'ini çizer.

    python src/evaluate.py
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from data import CLASSES, get_loaders
from model import SmallCNN


def main():
    runs = [json.load(open(f)) for f in sorted(glob.glob("results/*.json"))]
    if not runs:
        raise SystemExit("Önce src/train.py ile en az bir deney çalıştır.")
    runs.sort(key=lambda r: r["best_acc"])

    # 1) README'ye yapıştırılacak tablo
    print("| Deney | Augment | BatchNorm | Dropout | En iyi doğruluk |")
    print("|---|---|---|---|---|")
    for r in runs:
        a = r["args"]
        print(f"| {r['name']} | {a['augment']} | {a['bn']} | {a['dropout']} | {r['best_acc']*100:.2f}% |")

    # 2) Eğri grafiği
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for r in runs:
        ax[0].plot(r["history"]["train_loss"], label=r["name"])
        ax[1].plot([a * 100 for a in r["history"]["test_acc"]], label=r["name"])
    ax[0].set(title="Eğitim kaybı", xlabel="Epoch", ylabel="Loss")
    ax[1].set(title="Test doğruluğu", xlabel="Epoch", ylabel="%")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig("results/curves.png", dpi=150)

    # 3) En iyi modelin confusion matrix'i
    best = runs[-1]
    a = best["args"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN(use_bn=a["bn"], dropout=a["dropout"]).to(device)
    model.load_state_dict(torch.load(f"models/{best['name']}.pt", map_location=device))
    model.eval()
    _, test_loader = get_loaders(augment=False)
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds += model(x.to(device)).argmax(1).cpu().tolist()
            labels += y.tolist()
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Confusion matrix – {best['name']}")
    fig.tight_layout()
    fig.savefig("results/confusion_matrix.png", dpi=150)
    print(f"\nKaydedildi: results/curves.png, results/confusion_matrix.png (en iyi: {best['name']})")


if __name__ == "__main__":
    main()

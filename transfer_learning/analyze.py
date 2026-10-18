"""Recompute all reported metrics from saved test predictions and draw figures."""
import argparse
import csv
import json
import os
from pathlib import Path

from prepare import ROOT, dump
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS = ["Бенгальская", "Бирманская", "Британская к/ш", "Амер. бульдог", "Бигль", "Боксёр"]
STAGES = {"feature": "Feature Extraction", "finetune": "Fine-tuning"}


def read_predictions(out, stage):
    rows = list(csv.DictReader((out / f"{stage}_predictions.csv").open(encoding="utf-8")))
    return rows, np.array([int(r["true"]) for r in rows]), np.array([int(r["pred"]) for r in rows])


def metrics(y, pred, classes):
    report = classification_report(y, pred, labels=list(range(len(classes))), target_names=classes,
                                   output_dict=True, zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "correct": int((y == pred).sum()), "n": len(y),
            "macro": report["macro avg"], "weighted": report["weighted avg"],
            "per_class": {c: report[c] for c in classes},
            "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(classes)))).tolist()}


def main(out):
    config = json.loads((out / "config.json").read_text()); classes = config["classes"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    figures = out / "figures"; figures.mkdir(exist_ok=True)
    summary = {}; all_predictions = {}
    for stage, title in STAGES.items():
        rows, y, pred = read_predictions(out, stage); all_predictions[stage] = (rows, y, pred)
        m = metrics(y, pred, classes); summary[stage] = m
        history = json.loads((out / f"{stage}_history.json").read_text())
        info = json.loads((out / f"{stage}_training.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
        for ax, key, label in zip(axes, ["loss", "accuracy"], ["Cross-entropy", "Accuracy"]):
            for part, name in [("train", "Обучение (аугментации)"), ("val", "Валидация")]:
                ax.plot([h["epoch"] for h in history], [h[f"{part}_{key}"] for h in history], marker=".", label=name)
            ax.axvline(info["best_epoch"], color="gray", linestyle="--", label="Лучший checkpoint")
            ax.set(xlabel="Эпоха", ylabel=label); ax.grid(alpha=.25); ax.legend(fontsize=11)
        fig.suptitle(title); fig.savefig(figures / f"{stage}_learning.png", dpi=200); plt.close(fig)
        cm = np.array(m["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6.7, 5.3), layout="constrained")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j] > 50 else "black")
        ax.set(xticks=range(6), yticks=range(6), xticklabels=LABELS, yticklabels=LABELS,
               xlabel="Предсказанный класс", ylabel="Истинный класс", title=title)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right"); fig.colorbar(im, ax=ax, label="Изображения")
        fig.savefig(figures / f"{stage}_confusion.png", dpi=200); plt.close(fig)
    a, b = all_predictions["feature"], all_predictions["finetune"]
    assert [r["image_id"] for r in a[0]] == [r["image_id"] for r in b[0]] and np.array_equal(a[1], b[1])
    ca, cb = a[1] == a[2], b[1] == b[2]; delta = cb.astype(float) - ca.astype(float)
    rng = np.random.default_rng(config["seed"])
    bootstrap = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(10000)])
    summary["comparison"] = {"accuracy_delta": float(delta.mean()),
        "paired_bootstrap_95_ci": np.quantile(bootstrap, [.025, .975]).tolist(), "bootstrap_replicates": 10000,
        "corrected_by_finetune": int((~ca & cb).sum()), "spoiled_by_finetune": int((ca & ~cb).sum()),
        "both_correct": int((ca & cb).sum()), "both_wrong": int((~ca & ~cb).sum())}
    dump(out / "metrics.json", summary)
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["strategy", "accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted"])
        for s in STAGES:
            m=summary[s]; w.writerow([s,m["accuracy"],*[m[avg][key] for avg in ["macro","weighted"] for key in ["precision","recall","f1-score"]]])
    # Deterministic illustration: first alphabetically sorted test image of each class.
    selected = [next(i for i,r in enumerate(a[0]) if int(r["true"]) == c) for c in range(6)]
    make_examples(selected, a, b, figures / "predictions.png", "Первые тестовые изображения каждого класса")
    # Error illustrations are explicitly selected as errors, not a representative sample.
    errors = [i for i in range(len(ca)) if not ca[i] or not cb[i]][:6]
    make_examples(errors, a, b, figures / "errors.png", "Первые шесть изображений с ошибкой хотя бы одной стратегии")
    dump(out / "illustrations.json", {"predictions": [a[0][i]["image_id"] for i in selected], "errors": [a[0][i]["image_id"] for i in errors],
         "source": "Oxford-IIIT Pet, Parkhi et al., CVPR 2012; research use; original image rights apply"})
    print(json.dumps({s:{k:v for k,v in summary[s].items() if k in ['accuracy','correct','macro']} for s in STAGES},indent=2))


def make_examples(indices, a, b, target, title):
    fig, axes = plt.subplots(2,3,figsize=(10,7.3),layout="constrained")
    for ax in axes.flat: ax.axis("off")
    for ax,i in zip(axes.flat,indices):
        row=a[0][i]
        with Image.open(ROOT / "data/images" / (row["image_id"]+'.jpg')) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f'{row["image_id"]}\nИстина: {LABELS[a[1][i]]}\nFE: {LABELS[a[2][i]]}\nFT: {LABELS[b[2][i]]}',fontsize=14)
    fig.suptitle(title,fontsize=15); fig.savefig(target,dpi=200); plt.close(fig)


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,default=ROOT/"results"); args=p.parse_args(); main(args.output.resolve())

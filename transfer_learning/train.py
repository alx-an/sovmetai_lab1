"""Sequential feature extraction and partial fine-tuning; epoch-level resume."""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from prepare import ROOT, digest, dump


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp"); torch.save(obj, tmp); tmp.replace(path)


class Pets(Dataset):
    def __init__(self, rows, split, transform):
        self.rows = [r for r in rows if r["split"] == split]
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        with Image.open(ROOT / "data/images" / (row["image_id"] + ".jpg")) as image:
            x = self.transform(image.convert("RGB"))
        return x, int(row["label"]), row["image_id"]


def build_model(n_classes, pretrained=False):
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(576, n_classes))
    return model


def configure(model, config, stage):
    for p in model.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True
    if stage == "finetune":
        for block in model.features[config["unfreeze_from"]:]:
            for p in block.parameters(): p.requires_grad = True
    model.eval()
    model.classifier.train()
    if stage == "finetune":
        model.features[config["unfreeze_from"]:].train()
    # Fixed running statistics avoid drift on the small target training set.
    # Affine BN parameters in unfrozen blocks can still receive gradients.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm): module.eval()


def frozen_state(model, config, stage):
    last = len(model.features) if stage == "feature" else config["unfreeze_from"]
    return {k: v for k, v in model.features.state_dict().items() if int(k.split(".")[0]) < last}


def loader(rows, split, config, training=False, seed=0):
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    tf = weights.transforms()
    if training:
        tf = transforms.Compose([
            transforms.RandomResizedCrop(config["image_size"], scale=(0.7, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(Pets(rows, split, tf), batch_size=config["batch_size"], shuffle=training,
                      num_workers=config["workers"], generator=gen, pin_memory=torch.cuda.is_available())


def epoch_pass(model, batches, device, optimizer=None):
    total_loss = 0.; correct = 0; count = 0; predictions = []
    if optimizer is None: model.eval()
    with torch.set_grad_enabled(optimizer is not None):
        for x, y, ids in batches:
            x, y = x.to(device), y.to(device)
            if optimizer is not None: optimizer.zero_grad(set_to_none=True)
            logits = model(x); loss = nn.functional.cross_entropy(logits, y)
            if optimizer is not None: loss.backward(); optimizer.step()
            probs = logits.detach().softmax(1).cpu().numpy()
            pred = probs.argmax(1); truth = y.cpu().numpy()
            count += len(y); total_loss += loss.item() * len(y); correct += int((pred == truth).sum())
            if optimizer is None:
                predictions.extend({"image_id": image_id, "true": int(t), "pred": int(p),
                                    **{f"p_{j}": float(v) for j, v in enumerate(prob)}}
                                   for image_id, t, p, prob in zip(ids, truth, pred, probs))
    return {"loss": total_loss / count, "accuracy": correct / count}, predictions


def train_stage(config, rows, out, ckpt, stage, device, resume):
    last_path = ckpt / f"{stage}_last.pt"; best_path = ckpt / f"{stage}_best.pt"
    seed_all(config["seed"])
    model = build_model(len(config["classes"]), pretrained=(stage == "feature" and not (resume and last_path.exists()))).to(device)
    if stage == "finetune":
        initial = torch.load(ckpt / "feature_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(initial["model"])
    configure(model, config, stage)
    groups = [{"params": model.classifier.parameters(), "lr": config["feature_lr"] if stage == "feature" else config["finetune_head_lr"]}]
    if stage == "finetune":
        groups.append({"params": [p for p in model.features.parameters() if p.requires_grad], "lr": config["finetune_backbone_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    epochs = config["feature_epochs"] if stage == "feature" else config["finetune_epochs"]
    history = []; best_key = (-1., -float("inf")); start = 1
    metadata = {"initial_model_hash": state_hash(model.state_dict()), "initial_backbone_hash": state_hash(model.features.state_dict()),
                "frozen_before": state_hash(frozen_state(model, config, stage)),
                "head_before": state_hash(model.classifier.state_dict()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_names": [n for n,p in model.named_parameters() if p.requires_grad],
                "initialization": "ImageNet" if stage == "feature" else "feature_best.pt",
                "source_checkpoint_sha256": None if stage == "feature" else digest(ckpt / "feature_best.pt")}
    if resume and last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        assert saved["config"] == config and saved["split_sha256"] == digest(out / "split.csv")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]; metadata = saved["metadata"]; best_key = tuple(saved["best_key"]); start = saved["epoch"] + 1
    for epoch in range(start, epochs + 1):
        # Reseeding every epoch makes resume independent of the interrupted process.
        epoch_seed = config["seed"] + epoch + (10000 if stage == "finetune" else 0)
        seed_all(epoch_seed); configure(model, config, stage); begin = time.perf_counter()
        tr, _ = epoch_pass(model, loader(rows, "train", config, True, epoch_seed), device, optimizer)
        va, _ = epoch_pass(model, loader(rows, "val", config), device)
        frozen_after = state_hash(frozen_state(model, config, stage))
        assert frozen_after == metadata["frozen_before"], "Frozen tensors changed"
        record = {"epoch": epoch, "train_loss": tr["loss"], "train_accuracy": tr["accuracy"],
                  "val_loss": va["loss"], "val_accuracy": va["accuracy"], "seconds": time.perf_counter() - begin}
        history.append(record); key = (va["accuracy"], -va["loss"]); improved = key > best_key
        if improved: best_key = key
        saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                 "config": config, "history": history, "metadata": metadata, "best_key": best_key,
                 "split_sha256": digest(out / "split.csv")}
        if improved: atomic_save(saved, best_path)
        atomic_save(saved, last_path); dump(out / f"{stage}_history.json", history)
        print(stage, json.dumps(record), "best" if improved else "", flush=True)
    best = torch.load(best_path, map_location=device, weights_only=False); model.load_state_dict(best["model"])
    metadata.update({"best_epoch": best["epoch"], "best_validation_accuracy": best["best_key"][0],
        "best_validation_loss": -best["best_key"][1], "frozen_after": state_hash(frozen_state(model, config, stage)),
        "best_model_hash": state_hash(model.state_dict()), "best_backbone_hash": state_hash(model.features.state_dict()),
        "head_after": state_hash(model.classifier.state_dict()), "checkpoint_sha256": digest(best_path),
        "training_seconds": sum(h["seconds"] for h in history), "completed_epochs": len(history)})
    assert metadata["head_before"] != metadata["head_after"]
    assert metadata["frozen_before"] == metadata["frozen_after"]
    if stage == "finetune": assert metadata["initial_backbone_hash"] != metadata["best_backbone_hash"]
    dump(out / f"{stage}_training.json", metadata)


def evaluate(config, rows, out, ckpt, device):
    for stage in ["feature", "finetune"]:
        model = build_model(len(config["classes"])).to(device)
        saved = torch.load(ckpt / f"{stage}_best.pt", map_location=device, weights_only=False)
        assert saved["config"] == config and saved["split_sha256"] == digest(out / "split.csv")
        model.load_state_dict(saved["model"])
        scores = {}
        for split in ["train", "val", "test"]:
            scores[split], pred = epoch_pass(model, loader(rows, split, config), device)
            if split == "test":
                with (out / f"{stage}_predictions.csv").open("w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(pred[0])); w.writeheader(); w.writerows(pred)
        dump(out / f"{stage}_evaluation.json", scores)
        print(stage, "evaluation", scores, flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", type=Path, default=ROOT / "config.json")
    p.add_argument("--output", type=Path, default=ROOT / "results"); p.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--resume", action="store_true"); p.add_argument("--evaluate-only", action="store_true")
    args = p.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8-sig")); out = args.output.resolve(); ckpt = args.checkpoints.resolve()
    os.environ["TORCH_HOME"] = str(ROOT / ".cache/torch")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config["threads"]); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = list(csv.DictReader((out / "split.csv").open(encoding="utf-8")))
    assert config["image_size"] == 224, "Evaluation uses the original weights' 224px preset"
    if not args.evaluate_only:
        if not args.resume and any(ckpt.glob("*.pt")):
            raise RuntimeError("Checkpoints exist: use --resume or a new --checkpoints directory")
        dump(out / "config.json", config)
        dump(out / "environment.json", {"python": platform.python_version(), "torch": torch.__version__,
             "torchvision": torchvision.__version__, "numpy": np.__version__, "platform": platform.system(),
             "device": str(device), "cpu": platform.processor(), "threads": torch.get_num_threads(),
             "cuda_available": torch.cuda.is_available(), "weights_url": models.MobileNet_V3_Small_Weights.IMAGENET1K_V1.url})
        for stage in ["feature", "finetune"]: train_stage(config, rows, out, ckpt, stage, device, args.resume)
    # Test predictions are produced only after both training stages are complete.
    evaluate(config, rows, out, ckpt, device)


if __name__ == "__main__": main()

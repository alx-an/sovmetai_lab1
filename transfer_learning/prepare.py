"""Download official Oxford-IIIT Pet archives and make an auditable split."""
import argparse
import csv
import hashlib
import json
import random
import shutil
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"
ARCHIVES = {
    "images.tar.gz": "5c4f3ee8e5d25df40f4fd59a7f44e54c",
    "annotations.tar.gz": "95a8c909bbe2e81eed6a22bccdf3f68f",
}


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def prepare(config, out):
    from PIL import Image
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    sources = {}
    for name, expected in ARCHIVES.items():
        dest = data / name
        if not dest.exists() or digest(dest, "md5") != expected:
            print("Downloading", name, flush=True)
            part = dest.with_suffix(".part")
            with urllib.request.urlopen(URL + name, timeout=120) as r, part.open("wb") as f:
                shutil.copyfileobj(r, f, 1024 * 1024)
            if digest(part, "md5") != expected:
                raise ValueError(f"Checksum mismatch: {name}")
            part.replace(dest)
        sources[name] = {"url": URL + name, "md5": expected, "sha256": digest(dest)}
        with tarfile.open(dest) as tar:
            for member in tar:
                p = Path(member.name)
                wanted = (p.parent.as_posix() == "annotations" and p.name in ["trainval.txt", "test.txt", "README"])
                wanted |= (p.parent.as_posix() == "images" and p.stem.rsplit("_", 1)[0] in config["classes"] and p.suffix == ".jpg")
                if wanted and member.isfile():
                    target = data / p
                    target.parent.mkdir(exist_ok=True)
                    with tar.extractfile(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    rng = random.Random(config["seed"])
    rows = []
    for source in ["trainval", "test"]:
        lines = (data / "annotations" / f"{source}.txt").read_text().splitlines()
        for label, cls in enumerate(config["classes"]):
            ids = sorted(l.split()[0] for l in lines if l and not l.startswith("#") and l.split()[0].rsplit("_", 1)[0] == cls)
            rng.shuffle(ids)
            for i, image_id in enumerate(ids):
                split = "test" if source == "test" else ("val" if i < config["validation_per_class"] else "train")
                path = data / "images" / (image_id + ".jpg")
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                rows.append({"image_id": image_id, "class": cls, "label": label, "split": split,
                             "source_split": source, "sha256": digest(path), "pixel_sha256": pixels})
    rows.sort(key=lambda r: (r["split"], r["label"], r["image_id"]))
    for key in ["image_id", "sha256", "pixel_sha256"]:
        assert len({r[key] for r in rows}) == len(rows), f"Duplicate {key}; review before training"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "split.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    dump(out / "dataset.json", {"source": "https://www.robots.ox.ac.uk/~vgg/data/pets/", "archives": sources,
         "classes": config["classes"], "counts": dict(Counter(r["split"] for r in rows)),
         "per_class": {c: dict(Counter(r["split"] for r in rows if r["class"] == c)) for c in config["classes"]},
         "split_sha256": digest(out / "split.csv"), "exact_duplicates": 0,
         "note": "Official test preserved; stratified trainval split. Exact file and decoded-pixel duplicates checked; near-duplicates and ImageNet overlap not exhaustively audited."})
    print(json.dumps(dict(Counter(r["split"] for r in rows))), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", type=Path, default=ROOT / "config.json"); p.add_argument("--output", type=Path, default=ROOT / "results")
    args = p.parse_args()
    prepare(json.loads(args.config.read_text(encoding="utf-8-sig")), args.output.resolve())

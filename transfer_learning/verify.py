"""Audit splits, independently recalculate metrics, optionally rerun checkpoints."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from prepare import ROOT, digest, dump
from analyze import metrics, read_predictions


def main(out, checkpoints, full):
    config=json.loads((out/"config.json").read_text()); ds=json.loads((out/"dataset.json").read_text())
    rows=list(csv.DictReader((out/"split.csv").open(encoding="utf-8")))
    assert digest(out/"split.csv")==ds["split_sha256"]
    assert len(rows)==sum(ds["counts"].values())
    for key in ["image_id","sha256","pixel_sha256"]: assert len({r[key] for r in rows})==len(rows)
    test=[r for r in rows if r["split"]=="test"]
    assert all(r["source_split"] == ("test" if r["split"]=="test" else "trainval") for r in rows)
    expected=json.loads((out/"metrics.json").read_text())
    training={s:json.loads((out/f"{s}_training.json").read_text()) for s in ["feature","finetune"]}
    assert training["feature"]["best_model_hash"]==training["finetune"]["initial_model_hash"]
    assert training["feature"]["checkpoint_sha256"]==training["finetune"]["source_checkpoint_sha256"]
    for stage in training:
        t=training[stage]
        assert t["frozen_before"]==t["frozen_after"] and t["head_before"]!=t["head_after"]
        history=json.loads((out/f"{stage}_history.json").read_text())
        assert len(history)==config["feature_epochs" if stage=="feature" else "finetune_epochs"]
        best=max(history,key=lambda h:(h["val_accuracy"],-h["val_loss"]))
        assert best["epoch"]==t["best_epoch"]
        pred,y,p=read_predictions(out,stage)
        assert [r["image_id"] for r in pred]==[r["image_id"] for r in test]
        assert np.array_equal(y,[int(r["label"]) for r in test])
        probs=np.array([[float(r[f"p_{j}"]) for j in range(len(config["classes"]))] for r in pred])
        assert np.isfinite(probs).all() and (probs>=0).all() and np.allclose(probs.sum(1),1,atol=1e-6)
        assert np.array_equal(probs.argmax(1),p)
        assert metrics(y,p,config["classes"])==expected[stage]
        cm=np.array(expected[stage]["confusion_matrix"])
        precision=np.diag(cm)/np.maximum(cm.sum(0),1); recall=np.diag(cm)/cm.sum(1)
        f1=2*precision*recall/np.maximum(precision+recall,1e-15)
        assert np.isclose(np.trace(cm)/cm.sum(),expected[stage]["accuracy"])
        for k,v in [("precision",precision),("recall",recall),("f1-score",f1)]:
            assert np.isclose(v.mean(),expected[stage]["macro"][k])
        evaluation=json.loads((out/f"{stage}_evaluation.json").read_text())
        assert np.isclose(-np.log(probs[np.arange(len(y)),y]).mean(),evaluation["test"]["loss"],atol=1e-6)
        assert evaluation["test"]["accuracy"]==expected[stage]["accuracy"]
    if full:
        import hashlib
        import torch
        from train import build_model, loader, epoch_pass, state_hash
        os.environ["TORCH_HOME"]=str(ROOT/".cache/torch")
        torch.set_num_threads(config["threads"]); torch.use_deterministic_algorithms(True)
        model=build_model(len(config["classes"]),True)
        assert state_hash(model.features.state_dict())==training["feature"]["initial_backbone_hash"]
        for r in rows:
            path=ROOT/"data/images"/(r["image_id"]+'.jpg'); assert digest(path)==r["sha256"]
            with Image.open(path) as im:
                im=im.convert('RGB'); assert hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()==r["pixel_sha256"]
        for stage in training:
            path=checkpoints/f"{stage}_best.pt"; assert digest(path)==training[stage]["checkpoint_sha256"]
            ck=torch.load(path,map_location="cpu",weights_only=False); model.load_state_dict(ck["model"])
            assert state_hash(model.state_dict())==training[stage]["best_model_hash"]
            score,pred=epoch_pass(model,loader(rows,"test",config),torch.device("cpu"))
            stored,_,_=read_predictions(out,stage)
            assert [p["pred"] for p in pred]==[int(p["pred"]) for p in stored]
            assert all(np.allclose([r[f"p_{j}"] for j in range(6)],[float(s[f"p_{j}"]) for j in range(6)],atol=1e-6) for r,s in zip(pred,stored))
    audit={"split_and_exact_duplicate_checks":"passed", "metrics_from_csv":"passed", "independent_confusion_arithmetic":"passed",
           "sequential_initialization":"passed", "frozen_tensor_hashes":"passed", "completed_epochs":"passed",
           "full_image_and_checkpoint_reinference":"passed" if full else "not_requested"}
    dump(out/("verification_full.json" if full else "verification.json"),audit); print(json.dumps(audit,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,default=ROOT/"results")
    p.add_argument("--checkpoints",type=Path,default=ROOT/"checkpoints"); p.add_argument("--full",action="store_true")
    a=p.parse_args(); main(a.output.resolve(),a.checkpoints.resolve(),a.full)

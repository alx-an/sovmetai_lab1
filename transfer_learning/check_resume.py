"""Replay epochs after best checkpoints and compare to completed training tensors."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
from prepare import ROOT, dump
from train import state_hash


def main():
    cache=ROOT/".cache"; cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="resume-check-",dir=cache) as temp:
        work=Path(temp); out=work/"results"; ckpt=work/"checkpoints"; ckpt.mkdir()
        shutil.copytree(ROOT/"results",out)
        for stage in ["feature","finetune"]:
            src=ROOT/"checkpoints"/f"{stage}_best.pt"
            shutil.copy2(src,ckpt/src.name)
            shutil.copy2(src,ckpt/f"{stage}_last.pt")
        subprocess.run([sys.executable,str(ROOT/"train.py"),"--resume","--output",str(out),"--checkpoints",str(ckpt)],check=True)
        checks={}
        for stage in ["feature","finetune"]:
            original=torch.load(ROOT/"checkpoints"/f"{stage}_last.pt",map_location="cpu",weights_only=False)
            replay=torch.load(ckpt/f"{stage}_last.pt",map_location="cpu",weights_only=False)
            assert state_hash(original["model"])==state_hash(replay["model"])
            for a,b in zip(original["history"],replay["history"]):
                for key in ["epoch","train_loss","train_accuracy","val_loss","val_accuracy"]: assert a[key]==b[key]
            assert (ROOT/"results"/f"{stage}_predictions.csv").read_bytes()==(out/f"{stage}_predictions.csv").read_bytes()
            checks[stage]="passed: resumed epoch metrics, final tensors and test predictions match exactly"
        dump(ROOT/"results/resume_verification.json",checks)
        print(json.dumps(checks,indent=2))


if __name__=="__main__": main()

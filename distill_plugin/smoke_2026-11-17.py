#!/usr/bin/env python3
"""Real train.py smoke: teacher, baseline, FGD, checkpoint and deployment load.

Synthetic images verify execution only, never model accuracy. Uses the original
custom data loader, preprocessing and MPDIoU/NWD loss. Writes only under output.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    import cv2
    import numpy as np
    import torch
    import yaml
    from models.yolo import Model
    from distill_plugin.runtime import load_checkpoint
    from models.experimental import attempt_load
    torch.set_num_threads(2)
    names = yaml.safe_load((ROOT / "data/gesture_dataset.yaml").read_text())["names"]
    rng = np.random.RandomState(42)
    for split, count in (("train", 4), ("val", 2)):
        images, labels = output / "data/images" / split, output / "data/labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for i in range(count):
            image = rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            assert cv2.imwrite(str(images / ("%s.jpg" % i)), image)
            (labels / ("%s.txt" % i)).write_text("%s 0.5 0.5 0.25 0.25\n" % (i % len(names)))
    data = output / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(output / "data"), "train": "images/train", "val": "images/val",
                                    "names": names}), encoding="utf-8")
    hyp = yaml.safe_load((ROOT / "data/hyps/hyps.full_parameters_final20260622_focaler_scale_0.7_0.7_MPDIoU.yaml").read_text())
    # Also exercise original Mosaic/MixUp settings; do not replace the detection loss.
    hyp_path = output / "hyp.yaml"
    hyp_path.write_text(yaml.safe_dump(hyp))
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", YOLOv5_AUTOINSTALL="false",
               YOLO_AUTOINSTALL="false", YOLOv5_SKIP_UPDATE_CHECK="1", WANDB_MODE="disabled", COMET_MODE="DISABLED")
    student_cfg = ROOT / "models/yolov5s-cls13-2.2m-Mul8Channel025_eNPU_ReLU.yaml"
    teacher_cfg = ROOT / "distill_plugin/configs/yolov5m-gesture-eNPU-ReLU.yaml"
    initial = Model(str(student_cfg), ch=1, nc=len(names))
    initial.names = names
    torch.save({"model": initial, "epoch": -1}, output / "initial.pt")
    stages = {}
    for name, cfg, weights in (("teacher", teacher_cfg, ""), ("baseline", student_cfg, str(output / "initial.pt")),
                                ("fgd", student_cfg, str(output / "initial.pt"))):
        cmd = [sys.executable, "-u", str(ROOT / "train.py"), "--data", str(data), "--cfg", str(cfg),
               "--weights", weights, "--hyp", str(hyp_path), "--epochs", "2", "--batch-size", "2",
               "--imgsz", "320", "--device", args.device, "--workers", "0", "--noautoanchor",
               "--noplots", "--project", str(output / "runs"), "--name", name, "--save-period", "1"]
        if name == "fgd":
            cmd += ["--teacher_weight", str(output / "runs/teacher/weights/best.pt"), "--distill-method", "fgd"]
        with (output / (name + ".log")).open("w") as stream:
            result = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("%s failed: inspect %s" % (name, output / (name + ".log")))
        stages[name] = {"status": "complete", "command": cmd}
    # Resume a real mid-training checkpoint into an isolated output directory.
    resume_dir = output / "runs/fgd_resume"
    (resume_dir / "weights").mkdir(parents=True)
    shutil.copy2(str(output / "runs/fgd/weights/epoch0.pt"), str(resume_dir / "weights/last.pt"))
    options = yaml.safe_load((output / "runs/fgd/opt.yaml").read_text())
    options["save_dir"] = str(resume_dir)
    options["name"] = "fgd_resume"
    (resume_dir / "opt.yaml").write_text(yaml.safe_dump(options))
    resume_cmd = [sys.executable, "-u", str(ROOT / "train.py"), "--resume", str(resume_dir / "weights/last.pt")]
    with (output / "fgd_resume.log").open("w") as stream:
        result = subprocess.run(resume_cmd, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("FGD resume failed: inspect " + str(output / "fgd_resume.log"))
    assert load_checkpoint(resume_dir / "weights/last.pt")["epoch"] == 1
    stages["fgd_resume"] = {"status": "complete", "command": resume_cmd}
    best = load_checkpoint(output / "runs/fgd/weights/best.pt")
    last = load_checkpoint(output / "runs/fgd/weights/last.pt")
    assert "fgd" not in best and "scaler" not in best
    assert last["fgd"] and last["optimizer"] and last["scaler"] is not None
    assert not best["model"].model[-1]._forward_pre_hooks
    deployment = attempt_load(output / "runs/fgd/weights/best.pt", device=torch.device("cpu"))
    with torch.no_grad():
        assert torch.isfinite(deployment(torch.ones(1, 1, 256, 320))[0]).all()
    report = {"status": "complete", "device": args.device, "torch": torch.__version__, "synthetic_data": True,
              "accuracy_claim": False, "stages": stages, "student_parameters": sum(p.numel() for p in initial.parameters()),
              "teacher_parameters": sum(p.numel() for p in load_checkpoint(output / "runs/teacher/weights/best.pt")["model"].parameters()),
              "student_only_best_checkpoint": True, "resumable_fgd_last_checkpoint": True}
    (output / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

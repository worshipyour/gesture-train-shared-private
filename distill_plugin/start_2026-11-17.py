#!/usr/bin/env python3
"""Launch baseline+M teacher on GPUs 0/1, then the paired FGD student.

Uses the current Python environment. No downloads, process termination or batch
fallbacks. Per-GPU locks are shared across clones for the same Linux user.
"""
import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDENT = "models/yolov5s-cls13-2.2m-Mul8Channel025_eNPU_ReLU.yaml"
TEACHER = "distill_plugin/configs/yolov5m-gesture-eNPU-ReLU.yaml"
HYP = "data/hyps/hyps.full_parameters_final20260622_focaler_scale_0.7_0.7_MPDIoU.yaml"
ORIGINAL = Path("/home/sunyu/project/gesture_asc/gesture_train_shared")


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(temp), str(path))


def tail_log(path, lines=40):
    if not path.is_file():
        return "(log not created)"
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 32768))
        return "\n".join(stream.read().decode("utf-8", errors="replace").splitlines()[-lines:])


def inspect_run(run_dir):
    """Read diagnostics only; never reserve a GPU or launch training."""
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print("Run directory does not exist: " + str(run_dir), file=sys.stderr)
        return 1
    print("Python: %s\nRun directory: %s" % (sys.executable, run_dir))
    for name in ("status.json", "queue.log", "baseline.log", "teacher.log", "fgd.log"):
        path = run_dir / name
        if path.is_file():
            print("\n--- %s ---\n%s" % (name, tail_log(path, 60)))
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, timeout=15)
        print("\n--- nvidia-smi (exit %s) ---\n%s" % (result.returncode, result.stdout))
    except (OSError, subprocess.TimeoutExpired) as exc:
        print("GPU diagnostic unavailable: " + str(exc))
    return 0


def reserve_gpu(index):
    path = Path(tempfile.gettempdir()) / ("gesture_fgd_%s_gpu%s.lock" % (os.getuid(), index))
    lock = path.open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            ["nvidia-smi", "-i", str(index), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
        if result.stdout.strip():
            raise RuntimeError("GPU %s already has compute processes; inspect nvidia-smi." % index)
        return lock
    except Exception:
        lock.close()
        raise


def command(args, name, teacher_path=None):
    is_teacher = name == "teacher"
    cmd = [sys.executable, "-u", str(ROOT / "train.py"),
           "--name", name, "--project", str(args.run_dir),
           "--weights", args.teacher_init if is_teacher else str(args.weights),
           "--data", str(args.data), "--cfg", str(ROOT / (TEACHER if is_teacher else STUDENT)),
           "--hyp", str(args.hyp), "--epochs", str(args.epochs),
           "--batch-size", str(args.teacher_batch if is_teacher else args.student_batch),
           "--save-period", "1", "--imgsz", "320", "--device", "0",
           "--workers", str(args.workers), "--seed", str(args.seed)]
    if name == "fgd":
        cmd += ["--teacher_weight", str(teacher_path), "--distill-method", "fgd",
                "--fgd-weight", str(args.fgd_weight), "--fgd-warmup-epochs", "3"]
    return cmd


def best_metrics(folder):
    # Read metrics saved with the actual selected checkpoint, not rounded CSV rows.
    checkpoint = folder / "weights/best.pt"
    if checkpoint.is_file():
        sys.path.insert(0, str(ROOT))
        from distill_plugin.runtime import load_checkpoint
        saved = load_checkpoint(checkpoint)
        metrics = saved.get("validation_metrics")
        if metrics is not None:
            metrics = dict(metrics, epoch=saved["selection_epoch"])
            p, r = metrics["P"], metrics["R"]
            metrics["F1_from_mean_PR"] = 2*p*r/(p+r) if p+r else 0
            return metrics
    path = folder / "results.csv"
    if not path.is_file():
        return None
    with path.open() as stream:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    if not rows:
        return None
    row = max(rows, key=lambda r: (.1 * r["metrics/mAP_0.5"] + .9 * r["metrics/mAP_0.5:0.95"], r["epoch"]))
    p, r = row["metrics/precision"], row["metrics/recall"]
    return {"epoch": int(row["epoch"]), "P": p, "R": r, "F1_from_mean_PR": 2*p*r/(p+r) if p+r else 0,
            "mAP50": row["metrics/mAP_0.5"], "mAP50_95": row["metrics/mAP_0.5:0.95"]}


def write_summary(args, state):
    summary = {"status": state["status"], "error": state.get("error"),
               "split": "validation", "selection": "0.1*mAP50 + 0.9*mAP50_95",
               "test_evaluated": False, "seed": args.seed, "student_batch": args.student_batch,
               "jobs": state["jobs"], "metrics": {}}
    for name, job in state["jobs"].items():
        if job["status"] == "complete":
            summary["metrics"][name] = best_metrics(args.run_dir / name)
    b, k = summary["metrics"].get("baseline"), summary["metrics"].get("fgd")
    if b and k:
        summary["fgd_minus_baseline_pp"] = {key: 100*(k[key]-b[key]) for key in
                                           ("P", "R", "F1_from_mean_PR", "mAP50", "mAP50_95")}
    atomic_json(args.run_dir / "summary.json", summary)
    lines = ["# %s 训练验证结果" % args.run_dir.name, "",
             "队列状态：" + state["status"], "错误：" + state.get("error", "无"), "",
             "验证集按原 fitness 选择权重；未使用测试集选配置。F1 为平均 P/R 推导值。", "",
             "| 训练 | P | R | F1* | mAP50 | mAP50–95 |", "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in summary["metrics"].items():
        if metrics:
            lines.append("| %s | %s |" % (name, " | ".join("%.3f%%" % (100*metrics[key]) for key in
                         ("P", "R", "F1_from_mean_PR", "mAP50", "mAP50_95"))))
    if summary.get("fgd_minus_baseline_pp"):
        lines += ["", "FGD 减对照（百分点）：" + json.dumps(summary["fgd_minus_baseline_pp"], ensure_ascii=False)]
    lines += ["", "这是本次训练验证结果；GPU/端侧延时、AP95和正式测试尚未测量，不能从此表推断。"]
    (args.run_dir / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def execute(args):
    state = {"status": "running", "pid": os.getpid(), "jobs": {}}
    active = {}
    def save():
        atomic_json(args.run_dir / "status.json", state)
    def launch(name, gpu, teacher_path=None):
        if (args.run_dir / name).exists():
            raise RuntimeError("Run directory already exists: " + str(args.run_dir / name))
        lock = reserve_gpu(gpu)
        log = (args.run_dir / (name + ".log")).open("w")
        cmd = command(args, name, teacher_path)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), YOLOv5_AUTOINSTALL="false",
                   YOLO_AUTOINSTALL="false", YOLOv5_SKIP_UPDATE_CHECK="1",
                   WANDB_MODE="disabled", COMET_MODE="DISABLED")
        try:
            process = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
        except Exception:
            log.close()
            lock.close()
            raise
        active[name] = (process, lock, log)
        state["jobs"][name] = {"status": "running", "pid": process.pid, "gpu": gpu, "command": cmd}
        save()
        print("Started %s, physical GPU %s, PID %s" % (name, gpu, process.pid), flush=True)
    def wait():
        while active:
            for name in list(active):
                process, lock, log = active[name]
                code = process.poll()
                if code is None:
                    continue
                lock.close()
                log.close()
                checkpoint = args.run_dir / name / "weights/best.pt"
                state["jobs"][name].update(returncode=code,
                    status="complete" if code == 0 and checkpoint.is_file() else "failed")
                if state["jobs"][name]["status"] == "failed":
                    reason = "train.py exited with code %s" % code if code else "best.pt was not created"
                    state["jobs"][name]["error"] = reason
                    print("%s failed: %s\n%s" % (name, reason, tail_log(args.run_dir / (name + ".log"))),
                          file=sys.stderr, flush=True)
                del active[name]
                save()
                print("Finished %s: %s" % (name, state["jobs"][name]["status"]), flush=True)
            if active:
                time.sleep(5)
    try:
        if args.mode in ("all", "parallel", "baseline"):
            launch("baseline", 0)
        if args.mode in ("all", "parallel", "teacher"):
            launch("teacher", 1)
        wait()
        if any(j["status"] == "failed" for j in state["jobs"].values()):
            raise RuntimeError("A training job failed; inspect its log. No automatic batch changes or FGD launch.")
        if args.mode in ("all", "distill"):
            teacher_path = args.teacher_checkpoint or args.run_dir / "teacher/weights/best.pt"
            if not teacher_path.is_file():
                raise FileNotFoundError(str(teacher_path))
            launch("fgd", 0, teacher_path)
            wait()
            if state["jobs"]["fgd"]["status"] != "complete":
                raise RuntimeError("FGD failed; original checkpoints and logs are retained.")
        state["status"] = "complete"
    except Exception as exc:
        # Allow already-started training to finish and retain its GPU lock.
        state["error"] = str(exc)
        print("Queue error: " + str(exc), file=sys.stderr, flush=True)
        state["status"] = "waiting_for_started_jobs" if active else "failed"
        save()
        wait()
        state["status"] = "failed"
    finally:
        save()
        write_summary(args, state)
    print("Status: %s. Logs and report: %s" % (state["status"], args.run_dir), flush=True)
    return 0 if state["status"] == "complete" else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "parallel", "baseline", "teacher", "distill"], default="all")
    parser.add_argument("--weights", type=Path, default=ORIGINAL / "epoch76.pt")
    parser.add_argument("--teacher-init", default="", help="optional same-family M initial checkpoint; default from scratch")
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data/gesture_dataset.yaml")
    parser.add_argument("--hyp", type=Path, default=ROOT / HYP)
    parser.add_argument("--student-batch", type=int, default=128, help="same batch for baseline and FGD")
    parser.add_argument("--teacher-batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fgd-weight", type=float, default=0.005)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--inspect-run", type=Path, help="read a previous run's status/logs/GPU state; starts no training")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.inspect_run is not None:
        return inspect_run(args.inspect_run)
    for attr in ("weights", "data", "hyp", "teacher_checkpoint"):
        value = getattr(args, attr)
        if value is not None:
            setattr(args, attr, value.expanduser().resolve())
    needs_student = args.mode in ("all", "parallel", "baseline", "distill")
    required = [args.data, args.hyp, ROOT / STUDENT, ROOT / TEACHER]
    if needs_student:
        required.append(args.weights)
    if args.teacher_checkpoint:
        required.append(args.teacher_checkpoint)
    if args.teacher_init:
        args.teacher_init = str(Path(args.teacher_init).expanduser().resolve())
        required.append(Path(args.teacher_init))
    for path in required:
        if not path.is_file():
            raise FileNotFoundError("Required local file missing: " + str(path))
    if min(args.epochs, args.student_batch, args.teacher_batch) <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if args.mode == "distill" and not args.teacher_checkpoint:
        raise ValueError("--mode distill requires --teacher-checkpoint")
    if args.run_dir is None:
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d_%H%M%S")
        args.run_dir = ROOT / "runs" / (stamp + "_fgd_deployment")
    args.run_dir = args.run_dir.expanduser().resolve()
    if args.foreground:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        with (args.run_dir / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return execute(args)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    plan = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    atomic_json(args.run_dir / "plan.json", plan)
    with (args.run_dir / "queue.log").open("w") as stream:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())] + sys.argv[1:] +
                    ["--foreground", "--run-dir", str(args.run_dir)], cwd=str(ROOT),
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    (args.run_dir / "queue.pid").write_text(str(process.pid)+"\n")
    time.sleep(2)
    if process.poll() not in (None, 0):
        print((args.run_dir / "queue.log").read_text(), file=sys.stderr)
        return 1
    print("Queue submitted, PID %s\nDirectory: %s\nInspect: tail -f %s/queue.log" %
          (process.pid, args.run_dir, args.run_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())

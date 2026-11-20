#!/usr/bin/env python3
"""Read results and list potential no-KD controls; never start training or load GPU models."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path("/home/sunyu/project/gesture_asc/gesture_train_shared")
DEFAULT_FGD = ROOT / "runs/2026-11-19_191837_fgd_deployment"
FIELDS = ("epochs", "batch_size", "imgsz", "seed", "optimizer", "rect", "multi_scale", "single_cls",
          "label_smoothing", "autopad", "partialRemove", "autoCanvasPadd", "quad", "image_weights",
          "cos_lr", "freeze", "noval", "noautoanchor")


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def options(folder):
    path = folder / "opt.yaml"
    return yaml.safe_load(path.read_text()) if path.is_file() else None


def resolve_file(value, project):
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    path = path if path.is_absolute() else project / path
    return path.resolve() if path.is_file() else None


def identity(value, project, as_yaml=False):
    if isinstance(value, dict):
        data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    else:
        path = resolve_file(value, project)
        if path is None:
            return None
        if as_yaml:
            data = json.dumps(yaml.safe_load(path.read_text()), sort_keys=True,
                              ensure_ascii=False, allow_nan=False).encode("utf-8")
        else:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
    return hashlib.sha256(data).hexdigest()


def differences(reference, candidate, candidate_project):
    if not reference or not candidate:
        return ["missing opt.yaml; configuration not verified"]
    result = []
    for key in FIELDS:
        if reference.get(key) != candidate.get(key):
            result.append(key)
    if candidate.get("teacher_weight") or candidate.get("teacher_weights"):
        result.append("teacher enabled")
    if candidate.get("resume"):
        result.append("resumed run; randomness not verified")
    for key in ("cfg", "data", "hyp", "weights"):
        ref_id = identity(reference.get(key), ROOT, as_yaml=key != "weights")
        candidate_id = identity(candidate.get(key), candidate_project, as_yaml=key != "weights")
        if ref_id is None or candidate_id is None:
            result.append(key + " unavailable; not verified")
        elif ref_id != candidate_id:
            result.append(key + " content differs")
    return result


def csv_metrics(folder):
    path = folder / "results.csv"
    if not path.is_file():
        return None, []
    with path.open() as stream:
        rows = [{key.strip(): float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    required = ("epoch", "metrics/precision", "metrics/recall", "metrics/mAP_0.5", "metrics/mAP_0.5:0.95")
    if not rows:
        return None, []
    if any(not all(key in row and math.isfinite(row[key]) for key in required) for row in rows):
        raise ValueError("Missing or non-finite validation metrics: " + str(path))
    best = max(rows, key=lambda row: (.1*row["metrics/mAP_0.5"] + .9*row["metrics/mAP_0.5:0.95"], row["epoch"]))
    p, r = best["metrics/precision"], best["metrics/recall"]
    return dict(epoch=int(best["epoch"]), P=p, R=r, F1_from_mean_PR=2*p*r/(p+r) if p+r else 0,
                mAP50=best["metrics/mAP_0.5"], mAP50_95=best["metrics/mAP_0.5:0.95"],
                source="rounded CSV validation-fitness selection; best.pt not re-evaluated"), rows


def summarize_run(folder):
    metrics, rows = csv_metrics(folder)
    summary = read_json(folder.parent / "summary.json")
    if summary and summary.get("metrics", {}).get(folder.name):
        metrics = dict(summary["metrics"][folder.name], source="saved best checkpoint validation metrics from summary.json")
    return {"run": str(folder), "best_pt_exists": (folder / "weights/best.pt").is_file(),
            "epoch_rows": len(rows), "last_epoch": max((int(row["epoch"]) for row in rows), default=None),
            "metrics": metrics}


def collect(fgd_run, baseline_run=None, original=ORIGINAL):
    fgd_folder = fgd_run / "fgd" if (fgd_run / "fgd").is_dir() else fgd_run
    if not (fgd_folder / "results.csv").is_file():
        raise FileNotFoundError("FGD results.csv not found. Pass the queue directory or its fgd subdirectory: " + str(fgd_folder))
    reference = options(fgd_folder)
    status = read_json(fgd_folder.parent / "status.json")
    short_status = ({"status": status.get("status"), "error": status.get("error"),
                     "jobs": {name: {key: job.get(key) for key in ("status", "returncode")}
                              for name, job in status.get("jobs", {}).items()}} if status else None)
    report = {"read_only": True, "fgd": summarize_run(fgd_folder),
              "fgd_queue_status": short_status,
              "baseline_candidates": [], "fgd_minus_baseline_pp": None,
              "limitations": ["configuration matching uses saved opt.yaml and currently accessible config/weight files",
                              "dataset image/label history and source-code parity are not retrospectively verified",
                              "CSV fallback is rounded and is not an independent best.pt evaluation",
                              "validation only; AP95, held-out test and inference latency are not measured"]}
    paths = []
    if baseline_run:
        project = original if original.resolve() in baseline_run.resolve().parents else ROOT
        paths.append((baseline_run, project))
    else:
        for project in (original, ROOT):
            if (project / "runs").is_dir():
                paths.extend((csv_file.parent, project) for csv_file in sorted((project / "runs").rglob("results.csv")))
    visited = set()
    for folder, project in paths:
        if folder.resolve() == fgd_folder.resolve() or str(folder.resolve()) in visited:
            continue
        visited.add(str(folder.resolve()))
        saved = options(folder)
        if saved and (saved.get("teacher_weight") or saved.get("teacher_weights")):
            continue
        candidate = {"run": str(folder)}
        try:
            candidate.update(summarize_run(folder))
            mismatches = differences(reference, saved, project)
            expected_epochs = int(reference.get("epochs", 0)) if reference else 0
            if not candidate["best_pt_exists"]:
                mismatches.append("best.pt missing")
            if expected_epochs <= 0 or candidate["epoch_rows"] != expected_epochs or candidate["last_epoch"] != expected_epochs-1:
                mismatches.append("completed training budget differs or cannot be verified")
            candidate["configuration_differences_or_unknowns"] = mismatches
            candidate["configuration_match"] = not mismatches
            candidate["training_options"] = {key: saved.get(key) for key in ("epochs", "batch_size", "imgsz", "seed", "weights", "cfg")} if saved else None
        except Exception as exc:
            candidate.update(configuration_match=False, error=str(exc))
        report["baseline_candidates"].append(candidate)
    matches = [item for item in report["baseline_candidates"] if item.get("configuration_match")]
    report["configuration_matched_baselines"] = len(matches)
    if len(matches) == 1 and matches[0].get("metrics") and report["fgd"].get("metrics"):
        report["comparison_baseline"] = matches[0]["run"]
        report["fgd_minus_baseline_pp"] = {key: 100*(report["fgd"]["metrics"][key]-matches[0]["metrics"][key])
                                          for key in ("P", "R", "F1_from_mean_PR", "mAP50", "mAP50_95")}
    elif len(matches) > 1:
        report["next_action"] = "Multiple matching controls found; specify --baseline-run, do not select by the highest score."
    else:
        report["next_action"] = "No verified matched control found. Review candidate differences before deciding whether to train a new baseline."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fgd-run", type=Path, default=DEFAULT_FGD)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--original-project", type=Path, default=ORIGINAL)
    args = parser.parse_args()
    print(json.dumps(collect(args.fgd_run.resolve(), args.baseline_run.resolve() if args.baseline_run else None,
                             args.original_project.resolve()), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

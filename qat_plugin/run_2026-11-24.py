#!/usr/bin/env python3
"""Standalone generic W8A8 AIMET PTQ/QAT. Original train.py is not patched."""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL = "models/yolov5s-cls13-2.2m-Mul8Channel025_eNPU_ReLU.yaml"
HYP = "data/hyps/hyps.full_parameters_final20260622_focaler_scale_0.7_0.7_MPDIoU.yaml"
WEIGHTS = "/home/sunyu/project/gesture_asc/gesture_train_shared/runs/train/train_0915_2026-11-17_223808_baseline/weights/best.pt"


def environment():
    result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "aimet_required": "2.44.0 (Python >=3.10); existing training environment is not modified"}
    try:
        if sys.version_info < (3, 10):
            raise RuntimeError("AIMET 2.44 requires Python >=3.10; keep the original yolov5 environment unchanged")
        from importlib.metadata import version
        result["packages"] = {key: version(key) for key in ("torch", "torchvision", "aimet-torch", "onnx", "setuptools")}
        if result["packages"]["aimet-torch"] != "2.44.0":
            raise RuntimeError("This integration targets aimet-torch==2.44.0; use a separate compatible environment")
        import torch
        from aimet_torch import QuantizationSimModel
        from aimet_torch.nn.modules import custom
        import aimet_torch.onnx
        result["torch_cuda"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        result["aimet_import_ok"] = bool(QuantizationSimModel and custom.Add and aimet_torch.onnx.export)
        try:
            import pkg_resources  # legacy project's utils.general imports this package
        except ImportError as exc:
            raise RuntimeError("Legacy YOLO needs pkg_resources. In the isolated server environment run: python -m pip install setuptools==80.9.0") from exc
        result["legacy_pkg_resources_import_ok"] = bool(pkg_resources.parse_version)
        result["compatible"] = True
    except Exception as exc:
        result.update(compatible=False, error=str(exc))
    return result


def json_save(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def idle_gpu(device):
    """No bypass/kill; graphics-only activity does not block experiments."""
    if device == "cpu":
        return
    if not device.isdigit():
        raise ValueError("Use one physical GPU index, e.g. --device 0 or --device 1")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible != device:
        raise ValueError("Unset CUDA_VISIBLE_DEVICES or make it match the physical --device index")
    def query(args):
        return subprocess.check_output(["nvidia-smi"] + args, universal_newlines=True).strip()
    mapping = dict(line.split(", ", 1) for line in query(["--query-gpu=index,uuid", "--format=csv,noheader"]).splitlines())
    if device not in mapping:
        raise ValueError("GPU index not found: " + device)
    apps = query(["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"])
    if any(line.startswith(mapping[device] + ",") for line in apps.splitlines()):
        raise RuntimeError("GPU " + device + " already has compute processes; choose an idle authorized GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = device


class LimitedLoader:
    """Snapshot a fixed smoke subset; clone batches to protect labels from val's in-place crop."""
    def __init__(self, loader, limit):
        from itertools import islice
        self.dataset = loader.dataset
        self.batches = [deepcopy(batch) for batch in islice(loader, limit)]
        if not self.batches:
            raise ValueError("Validation subset is empty")

    def __iter__(self):
        return (deepcopy(batch) for batch in self.batches)

    def __len__(self):
        return len(self.batches)


@contextmanager
def record_ap_metrics(evaluator, names, classes):
    """Capture original evaluator's AP matrix without enabling its legacy plot threads."""
    original = evaluator.ap_per_class
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        tp, fp, p, r, f1, ap, accuracy, false_alarm, class_ids = result
        target_classes = args[3] if len(args) > 3 else kwargs["target_cls"]
        counts = Counter(int(x) for x in target_classes)
        for i, cls in enumerate(class_ids):
            cls = int(cls)
            row = [float(x) for x in ap[i]]
            if len(row) != 10:
                raise ValueError("Expected original evaluator AP at IoU 0.50:0.05:0.95")
            classes.append({"class_id": cls, "name": names[cls], "instances": counts[cls],
                            "P": float(p[i]), "R": float(r[i]), "F1": float(f1[i]),
                            "AP50": row[0], "AP50_95": sum(row)/len(row), "AP95": row[-1]})
        return result
    evaluator.ap_per_class = capture
    try:
        yield
    finally:
        evaluator.ap_per_class = original


def metrics_dict(values):
    p, r, ap50, ap = [float(x) for x in values[:4]]
    return {"P": p, "R": r, "F1_from_mean_PR": 2*p*r/(p+r) if p+r else 0., "mAP50": ap50, "mAP50_95": ap}


def delta(lhs, rhs):
    keys = ["P", "R", "F1_from_mean_PR", "mAP50", "mAP50_95"]
    if lhs.get("AP95") is not None and rhs.get("AP95") is not None:
        keys.append("AP95")
    return {key: 100*(lhs[key]-rhs[key]) for key in keys}


def evaluate(raw, detector, data, loader, loss_fn, output, batch_size, image_size):
    from qat_plugin.model import DecodedDetector
    import val
    output.mkdir(parents=True, exist_ok=True)
    classes = []
    adapter = DecodedDetector(raw, detector).eval()
    # compute_loss makes val use (decoded, raw); labels/cropping/NMS stay in the original evaluator.
    with record_ap_metrics(val, detector.names, classes):
        results, maps, _ = val.run(data, model=adapter, dataloader=loader, compute_loss=loss_fn,
                                  imgsz=image_size, batch_size=batch_size, half=False, plots=False,
                                  verbose=True, save_dir=output)
    metric = metrics_dict(results)
    metric.update(per_class=classes, split="validation", latency_measured=False)
    metric["AP95"] = sum(row["AP95"] for row in classes)/len(classes) if classes else None
    # val.py's speed is deliberately not promoted to an INT8/edge latency measurement.
    json_save(output / "metrics.json", metric)
    return metric


def make_loaders(data, hyp, detector, args):
    from utils.dataloaders_nopad_nopar import create_dataloader
    from utils.general import check_img_size
    stride = max(int(detector.stride.max()), 32)
    image_size = check_img_size([120, 160], stride)  # identical to original train.py --imgsz 320
    common = dict(imgsz=image_size, stride=stride, hyp=hyp, workers=args.workers, seed=args.seed, cache=False)
    train, dataset = create_dataloader(data["train"], batch_size=args.batch_size, augment=True,
                                        shuffle=True, prefix="QAT train: ", **common)
    calibration, _ = create_dataloader(data["train"], batch_size=args.batch_size, augment=False,
                                       shuffle=True, prefix="calibration TRAIN only: ", **common)
    val, _ = create_dataloader(data["val"], batch_size=args.val_batch, augment=False,
                              rect=True, pad=.5, prefix="validation: ", **common)
    return train, calibration, val, image_size


def inventory(raw):
    import torch
    convs = []
    activations = []
    for name, layer in raw.named_modules():
        for attr in ("input_quantizers", "output_quantizers"):
            for index, q in enumerate(getattr(layer, attr, [])):
                if q is not None:
                    if q.bitwidth != 8:
                        raise RuntimeError("Non-A8 activation quantizer: " + name)
                    activations.append({"name": name + "." + attr + "." + str(index), "bits": int(q.bitwidth)})
        if isinstance(layer, torch.nn.Conv2d):
            quantizers = getattr(layer, "param_quantizers", {})
            quantizer = quantizers["weight"] if "weight" in quantizers else None
            if quantizer is None or quantizer.bitwidth != 8:
                raise RuntimeError("Missing W8 weight quantizer for conv: " + name)
            convs.append({"name": name, "weight_bits": int(quantizer.bitwidth), "weight_shape": list(layer.weight.shape)})
    if not convs or not activations:
        raise RuntimeError("No quantized convolutions found; refusing to label this run W8A8")
    return {"convs": convs, "activations": activations}


def run(args, output):
    from copy import deepcopy
    import math
    import torch
    import yaml
    from aimet_torch import QuantizationSimModel
    from aimet_torch.common.defs import QuantScheme
    from aimet_torch.common.quantsim_config.utils import get_path_for_per_channel_config
    from aimet_torch.nn.modules import custom
    from qat_plugin.model import prepare_detector, adjust_targets
    from utils.general import check_dataset, init_seeds
    from utils.loss import ComputeLoss
    from tqdm import tqdm
    device = torch.device("cpu" if args.device == "cpu" else "cuda:0")
    init_seeds(args.seed, deterministic=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    weights = Path(args.weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError("Trained FP student checkpoint required: " + str(weights))
    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
    detector = deepcopy(checkpoint.get("ema") or checkpoint["model"]).float().to(device).eval()
    for param in detector.parameters():
        param.requires_grad_(True)  # stripped best.pt checkpoints normally have requires_grad=False
    data = check_dataset(args.data, autodownload=False)
    if int(data["nc"]) != detector.model[-1].nc:
        raise ValueError("Dataset/head class count mismatch")
    expected_names = list(data["names"].values()) if isinstance(data["names"], dict) else data["names"]
    actual_names = list(detector.names.values()) if isinstance(detector.names, dict) else list(detector.names)
    if expected_names != actual_names:
        raise ValueError("Dataset/checkpoint class order mismatch")
    cfg = yaml.safe_load(Path(args.cfg).read_text())
    cfg["nc"], cfg["ch"] = int(data["nc"]), 1
    if cfg != detector.yaml:
        raise ValueError("Checkpoint YAML differs from requested student architecture; no partial weight loading is allowed")
    hyp = yaml.safe_load(Path(args.hyp).read_text())
    for key in ("mosaic", "mixup", "copy_paste"):
        hyp[key] = 0.0
    train, calibration, val_loader, image_size = make_loaders(data, hyp, detector, args)
    train_paths = {str(Path(x).resolve()) for x in train.dataset.im_files}
    val_paths = {str(Path(x).resolve()) for x in val_loader.dataset.im_files}
    if train_paths & val_paths:
        raise ValueError("Train/validation image paths overlap; calibration must use the training split only")
    first = next(iter(calibration))
    dummy = first[1][:1].to(device).float() / 255.
    if dummy.shape[1] != 1:
        raise ValueError("Expected original single-channel Y preprocessing")
    input_shape = list(dummy.shape)
    # Identical original loss scaling; the input tensor shape is inferred, never replaced by RGB.
    hyp["box"] *= 3 / detector.model[-1].nl
    hyp["cls"] *= data["nc"] / 80 * 3 / detector.model[-1].nl
    hyp["obj"] *= (max(image_size) / 640)**2 * 3 / detector.model[-1].nl
    hyp["label_smoothing"] = 0.
    detector.hyp, detector.nc, detector.names = hyp, data["nc"], data["names"]
    loss_fn = ComputeLoss(detector)
    raw = prepare_detector(detector, custom, dummy).to(device)
    for parameter in raw.parameters():
        parameter.requires_grad_(True)  # project's inference BN-fuser creates frozen Conv parameters
    manifest = {"mode": args.mode, "smoke_only": args.smoke, "input_shape": input_shape,
                "weights": str(weights), "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "source_parameters": sum(p.numel() for p in detector.parameters()),
                "prepared_parameters": sum(p.numel() for p in raw.parameters()), "options": vars(args),
                "data": data, "scaled_loss_hyp": hyp, "calibration_split": "train",
                "train_images": len(train_paths), "val_images": len(val_paths),
                "train_list_sha256": hashlib.sha256("\n".join(sorted(train_paths)).encode()).hexdigest(),
                "val_list_sha256": hashlib.sha256("\n".join(sorted(val_paths)).encode()).hexdigest(),
                "calibration_files": [], "calibration_images": 0, "test_evaluated": False,
                "decode_and_nms": "floating point, outside quantized raw-head graph", "environment": environment()}
    # Keep filenames as strings for JSON while recording resolved data and exact hypothesis.
    manifest = json.loads(json.dumps(manifest, default=str))
    if args.smoke:
        val_loader = LimitedLoader(val_loader, 2)
        manifest["smoke_validation_files"] = [str(path) for batch in val_loader.batches for path in batch[3]]
    summary = {"status": "running", "smoke_only": args.smoke, "metrics": {}, "changes_pp": {},
               "test_evaluated": False, "edge_latency_measured": False, "selection": "0.1*mAP50+0.9*mAP50_95",
               "accuracy_comparison_enabled": not args.smoke,
               "metrics_note": "Smoke metrics are diagnostic only; no accuracy-gain claims." if args.smoke else "Full fixed validation split."}
    summary["metrics"]["FP32"] = evaluate(raw, detector, data, val_loader, loss_fn, output / "fp32", args.val_batch, image_size)
    sim = None
    if args.mode != "float-control":
        config = args.quant_config or get_path_for_per_channel_config()
        manifest["quant_config"] = str(config)
        manifest["quant_config_sha256"] = hashlib.sha256(Path(config).read_bytes()).hexdigest()
        sim = QuantizationSimModel(raw.eval(), dummy_input=dummy,
                                   quant_scheme=QuantScheme.post_training_tf,
                                   default_param_bw=8, default_output_bw=8, config_file=config)
        raw = sim.model
        def calibrate(model):
            model.eval()
            with torch.no_grad():
                for batch in calibration:
                    remaining = args.calibration_samples - manifest["calibration_images"]
                    if remaining <= 0:
                        break
                    inputs = batch[1][:remaining].to(device).float() / 255.
                    if list(inputs.shape[1:]) != input_shape[1:]:
                        raise ValueError("Calibration input geometry changed")
                    model(inputs)
                    manifest["calibration_files"].extend(str(x) for x in batch[3][:len(inputs)])
                    manifest["calibration_images"] += len(inputs)
        sim.compute_encodings(calibrate)
        manifest["quantizers"] = inventory(raw)
        summary["metrics"]["PTQ_W8A8"] = evaluate(raw, detector, data, val_loader, loss_fn, output / "ptq", args.val_batch, image_size)
        if not args.smoke:
            summary["changes_pp"]["PTQ_minus_FP32"] = delta(summary["metrics"]["PTQ_W8A8"], summary["metrics"]["FP32"])
    json_save(output / "manifest.json", manifest)
    json_save(output / "summary.json", summary)
    initial_key = "FP32" if sim is None else "PTQ_W8A8"
    best = summary["metrics"][initial_key]
    best_score = .1*best["mAP50"] + .9*best["mAP50_95"]
    best_epoch = -1
    def save_state(name, epoch, optimizer=None):
        state = {"model_state": raw.state_dict(), "epoch": epoch, "best_epoch": best_epoch,
                 "best_metrics": best, "optimizer": optimizer.state_dict() if optimizer else None,
                 "manifest": manifest, "format": "AIMET simulation state; not a deployable INT8 checkpoint"}
        temp = output / (name + ".tmp")
        torch.save(state, temp)
        temp.replace(output / name)
    save_state("best_sim_state.pt", -1)
    history = []
    if args.mode != "ptq":
        init_seeds(args.seed, deterministic=True)  # same training seed for QAT and FP control
        optimizer = torch.optim.SGD(raw.parameters(), lr=args.lr, momentum=.9, weight_decay=0.)
        for epoch in range(args.epochs):
            raw.train()
            # BN was folded; no AMP, EMA, distillation, AutoAnchor, or recalibration during fine-tuning.
            for group in optimizer.param_groups:
                group["lr"] = args.lr * (.1 + .9*(1+math.cos(math.pi*epoch / args.epochs))/2)
            losses = []
            for i, batch in enumerate(tqdm(train, desc="%s epoch %d/%d" % (args.mode, epoch+1, args.epochs))):
                if args.smoke and i >= 2:
                    break
                inputs = batch[1].to(device).float() / 255.
                if list(inputs.shape[1:]) != input_shape[1:]:
                    raise ValueError("Train input geometry changed; fixed-shape run aborted")
                targets = adjust_targets(batch[2].to(device), inputs.shape[2])
                optimizer.zero_grad(set_to_none=True)
                loss, _ = loss_fn(list(raw(inputs)), targets)
                # ComputeLoss already multiplies by batch size; average for stable fine-tuning LR.
                loss = loss / len(inputs)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite detection loss; no fallback to FP training")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(raw.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            metric = evaluate(raw, detector, data, val_loader, loss_fn, output / ("epoch_%03d" % epoch), args.val_batch, image_size)
            score = .1*metric["mAP50"] + .9*metric["mAP50_95"]
            if score > best_score:
                best, best_score, best_epoch = metric, score, epoch
                save_state("best_sim_state.pt", epoch, optimizer)
            history.append({"epoch": epoch, "mean_loss": sum(losses)/max(len(losses), 1), "metrics": metric})
            save_state("last_sim_state.pt", epoch, optimizer)
            json_save(output / "history.json", history)
            summary.update(best_epoch=best_epoch, completed_epochs=epoch+1)
            json_save(output / "summary.json", summary)
    best_state = torch.load(output / "best_sim_state.pt", map_location=device, weights_only=False)
    raw.load_state_dict(best_state["model_state"], strict=True)
    key = "QAT_W8A8_selected" if sim else "FP32_control_selected"
    if args.mode != "ptq":
        summary["metrics"][key] = best
        if not args.smoke:
            summary["changes_pp"][key + "_minus_FP32"] = delta(best, summary["metrics"]["FP32"])
            if sim:
                summary["changes_pp"]["QAT_minus_PTQ"] = delta(best, summary["metrics"]["PTQ_W8A8"])
    summary["best_epoch"] = best_epoch
    summary["selection_includes_initial_model"] = True
    summary["qat_selected_initial_ptq"] = args.mode == "qat" and best_epoch == -1
    summary["best_state"] = str(output / "best_sim_state.pt")
    summary["training_status"] = "complete"
    summary["status"] = "exporting" if sim is not None and not args.skip_export else "complete"
    json_save(output / "summary.json", summary)
    # Export is a separate step with its own failure marker; a training result is retained if export fails.
    if sim is not None and not args.skip_export:
        import aimet_torch.onnx
        raw.eval()
        export_dir = output / "export"
        export_dir.mkdir()
        try:
            aimet_torch.onnx.export(raw, dummy, str(export_dir / "gesture_w8a8.onnx"),
                                    input_names=["y_plane"], output_names=["p3", "p4", "p5"],
                                    opset_version=21, dynamo=False, prequantize_constants=True)
            import onnx
            exported = onnx.load(str(export_dir / "gesture_w8a8.onnx"))
            onnx.checker.check_model(exported)
            q_nodes = sum(n.op_type == "QuantizeLinear" for n in exported.graph.node)
            dq_nodes = sum(n.op_type == "DequantizeLinear" for n in exported.graph.node)
            if not q_nodes or not dq_nodes:
                raise RuntimeError("Export is missing ONNX Q/DQ nodes; refusing to label it quantized")
            manifest_export = {"status": "onnx_checked", "compiled_for_hardware": False,
                               "format": "ONNX QDQ; scales/zero points embedded in graph", "q_nodes": q_nodes, "dq_nodes": dq_nodes,
                               "files": [{"name": p.name, "bytes": p.stat().st_size} for p in export_dir.iterdir() if p.is_file()],
                               "decode": {"anchors_grid_units": detector.model[-1].anchors.detach().cpu().tolist(),
                                          "stride": detector.stride.detach().cpu().tolist(), "names": detector.names},
                               "note": "Raw head outputs; use original decode+NMS. ONNX/encodings are not proof of INT8 kernel execution."}
            json_save(export_dir / "deployment.json", manifest_export)
            summary["export"] = manifest_export
        except Exception as exc:
            summary["export"] = {"status": "failed", "error": str(exc)}
            summary["status"] = "export_failed"
            json_save(output / "summary.json", summary)
            raise
    summary["status"] = "complete"
    json_save(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--mode", choices=["ptq", "qat", "float-control"], default="qat")
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--cfg", default=MODEL)
    parser.add_argument("--data", default="data/gesture_dataset.yaml")
    parser.add_argument("--hyp", default=HYP)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-samples", type=int, default=1024)
    parser.add_argument("--quant-config", help="AIMET config path; default is generic per-channel W8A8")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="1 epoch/2 train/2 val batches, 8 calibration images; not accuracy evidence")
    args = parser.parse_args()
    if args.preflight:
        info = environment()
        print(json.dumps(info, indent=2))
        return 0 if info["compatible"] else 2
    if args.epochs <= 0 or min(args.batch_size, args.val_batch, args.calibration_samples) <= 0 or args.lr <= 0:
        parser.error("Epochs, batches, calibration samples and LR must be positive")
    if args.smoke:
        args.epochs, args.calibration_samples = 1, 8
    # Acquire before importing torch/AIMET; lock is shared across clone names.
    lock = Path("/tmp") / ("gesture_aimet_gpu_" + args.device + ".lock")
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError("GPU experiment lock exists; inspect its PID before any manual removal: " + str(lock))
    output = None
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        idle_gpu(args.device)
        info = environment()
        if not info["compatible"]:
            print(json.dumps(info, indent=2))
            return 2
        os.chdir(str(ROOT))
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d_%H%M%S_%f")
        output = ROOT / "runs" / (stamp + "_aimet_" + args.mode + ("_smoke" if args.smoke else ""))
        output.mkdir(parents=True, exist_ok=False)
        json_save(output / "status.json", {"status": "running", "pid": os.getpid(), "options": vars(args)})
        print("Run directory:", output, flush=True)
        run(args, output)
        json_save(output / "status.json", {"status": "complete", "pid": os.getpid()})
        return 0
    except BaseException as exc:
        if output:
            json_save(output / "status.json", {"status": "failed", "pid": os.getpid(), "error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    sys.exit(main())

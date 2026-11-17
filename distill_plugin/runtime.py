"""Training-only FGD support for this repository's single-channel YOLOv5.

No modules or persistent hooks are attached to the deployed student.
"""
import contextlib
import hashlib
import json
import math
from pathlib import Path

import torch

from .fgd import FGD


def load_checkpoint(path, map_location="cpu"):
    """Load our trusted, full-model YOLO checkpoints on old and new PyTorch."""
    import inspect
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_names(names):
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def validate_teacher(teacher, student, names):
    sh, th = student.model[-1], teacher.model[-1]
    if th.nc != sh.nc or ordered_names(teacher.names) != ordered_names(names):
        raise ValueError("Teacher classes/order differ from the dataset. Train the teacher with the same data YAML.")
    if th.nl != sh.nl or not torch.equal(th.stride.cpu(), sh.stride.cpu()):
        raise ValueError("FGD requires matching student/teacher feature strides and levels.")
    if next(student.parameters()).shape[1] != next(teacher.parameters()).shape[1]:
        raise ValueError("Teacher and student input channels differ.")


class FGDTrainer:
    def __init__(self, student, teacher, teacher_path, weight=0.005, warmup_epochs=3.0, seed=0):
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("--fgd-weight must be finite and positive")
        if not math.isfinite(warmup_epochs) or warmup_epochs < 0:
            raise ValueError("--fgd-warmup-epochs must be finite and nonnegative")
        self.student = student
        self.teacher = teacher.eval().requires_grad_(False)
        self.teacher_sha256 = file_sha256(teacher_path)
        self.config = {"weight": float(weight), "warmup_epochs": float(warmup_epochs), "seed": int(seed)}
        sc = [m.in_channels for m in student.model[-1].m]
        tc = [m.in_channels for m in teacher.model[-1].m]
        # Keep the student's data/augmentation RNG unchanged by adapter initialization.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(seed + 917).get_state())
            self.criterion = FGD(sc, tc).to(next(student.parameters()).device)
        self._student_features = None
        self._sums = None
        self._samples = 0
        self.input_shape = None

    @staticmethod
    @contextlib.contextmanager
    def capture(model):
        captured = []
        def hook(module, inputs):
            # Detect mutates its input list in place; retain the original feature references.
            captured.append(tuple(inputs[0]))
        handle = model.model[-1].register_forward_pre_hook(hook)
        try:
            yield captured
        finally:
            handle.remove()

    @contextlib.contextmanager
    def student_forward(self):
        self._student_features = None
        with self.capture(self.student) as captured:
            yield
        if len(captured) != 1:
            raise RuntimeError("Expected exactly one student Detect forward for FGD")
        self._student_features = captured[0]

    def loss(self, images, targets, epoch_position):
        if self._student_features is None:
            raise RuntimeError("Capture student features before calculating FGD")
        try:
            self.teacher.eval()
            with torch.no_grad(), self.capture(self.teacher) as captured:
                self.teacher(images)
            if len(captured) != 1:
                raise RuntimeError("Expected exactly one teacher Detect forward for FGD")
            with torch.cuda.amp.autocast(enabled=False):
                raw, components = self.criterion(self._student_features, captured[0], targets)
            warmup = self.config["warmup_epochs"]
            gain = self.config["weight"] * (min(1.0, max(0.0, epoch_position) / warmup) if warmup else 1.0)
            weighted = raw * gain
            if not torch.isfinite(weighted):
                raise FloatingPointError("Non-finite FGD loss; stopping instead of saving invalid weights")
            batch = images.shape[0]
            values = torch.stack([raw.detach(), weighted.detach()] + [v.detach() for v in components.values()])
            self._sums = values * batch if self._sums is None else self._sums + values * batch
            self._samples += batch
            self.input_shape = list(images.shape[1:])
            return weighted
        finally:
            self._student_features = None

    def log_epoch(self, path, epoch):
        if self._samples:
            keys = ("raw", "weighted", "foreground", "background", "attention", "relation")
            record = dict(zip(keys, (self._sums / self._samples).cpu().tolist()))
            record.update(epoch=int(epoch), samples=self._samples, input_shape=self.input_shape,
                          teacher_sha256=self.teacher_sha256)
            with Path(path).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        self._sums, self._samples = None, 0

    def state_dict(self):
        return {"version": 1, "config": self.config, "teacher_sha256": self.teacher_sha256,
                "criterion": self.criterion.state_dict()}

    def load_state_dict(self, state):
        if state.get("version") != 1 or state["config"] != self.config:
            raise ValueError("FGD resume configuration mismatch")
        if state["teacher_sha256"] != self.teacher_sha256:
            raise ValueError("Teacher checkpoint changed since the saved FGD run")
        self.criterion.load_state_dict(state["criterion"], strict=True)

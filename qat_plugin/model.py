"""Equivalent operator preparation; no edits to the project's model definitions."""
from copy import deepcopy

import torch
from torch import nn


class PreparedBottleneck(nn.Module):
    def __init__(self, layer, ops):
        super().__init__()
        self.cv1, self.cv2, self.add = layer.cv1, layer.cv2, layer.add
        self.sum = ops.Add() if self.add else None

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return self.sum(x, y) if self.add else y


class PreparedC3(nn.Module):
    def __init__(self, layer, ops):
        super().__init__()
        self.cv1, self.cv2, self.cv3, self.m = layer.cv1, layer.cv2, layer.cv3, layer.m
        self.cat = ops.Concat(axis=1)

    def forward(self, x):
        return self.cv3(self.cat(self.m(self.cv1(x)), self.cv2(x)))


class PreparedSPPF(nn.Module):
    def __init__(self, layer, ops):
        super().__init__()
        self.cv1, self.cv2 = layer.cv1, layer.cv2
        self.pools = nn.ModuleList([deepcopy(layer.m) for _ in range(3)])
        self.cat = ops.Concat(axis=1)

    def forward(self, x):
        x = self.cv1(x)
        a = self.pools[0](x)
        b = self.pools[1](a)
        return self.cv2(self.cat(x, a, b, self.pools[2](b)))


class PreparedConcat(nn.Module):
    def __init__(self, layer, ops):
        super().__init__()
        self.cat = ops.Concat(axis=layer.d)

    def forward(self, inputs):
        return self.cat(*inputs)


def prepare_ops(module, ops):
    from models.common import Bottleneck, C3, SPPF, Concat
    types = {Bottleneck: PreparedBottleneck, C3: PreparedC3, SPPF: PreparedSPPF, Concat: PreparedConcat}
    for name, child in list(module.named_children()):
        prepare_ops(child, ops)
        cls = types.get(type(child))
        if cls:
            replacement = cls(child, ops)
            for key in ("i", "f", "type", "np"):
                if hasattr(child, key):
                    setattr(replacement, key, getattr(child, key))
            setattr(module, name, replacement)


class RawDetector(nn.Module):
    """Quantize backbone/neck and all three head convs; decode/NMS stay outside."""
    def __init__(self, detector):
        super().__init__()
        self.layers = nn.ModuleList(list(detector.model.children())[:-1])
        head = detector.model[-1]
        self.head_convs = head.m
        self.head_sources = tuple(head.f)
        self.saved = set(detector.save)
        self.na, self.no = head.na, head.no

    def forward(self, x):
        saved = []
        for layer in self.layers:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [x if j == -1 else saved[j] for j in layer.f]
            x = layer(x)
            saved.append(x if layer.i in self.saved else None)
        outputs = []
        for conv, source in zip(self.head_convs, self.head_sources):
            y = conv(saved[source])
            bs, _, h, w = y.shape
            outputs.append(y.view(bs, self.na, self.no, h, w).permute(0, 1, 3, 4, 2).contiguous())
        return tuple(outputs)


class DecodedDetector(nn.Module):
    """Original YOLO decode in FP32, compatible with val.run(model=..., compute_loss=...)."""
    def __init__(self, raw, detector):
        super().__init__()
        self.raw = raw
        head = detector.model[-1]
        self.register_buffer("anchors", head.anchors.detach().clone())
        self.register_buffer("stride", head.stride.detach().clone())
        self.names = detector.names

    def forward(self, x, augment=False):
        if augment:
            raise ValueError("Augmented inference is not supported for this fixed-shape quantization run")
        raw = self.raw(x)
        decoded = []
        for i, p in enumerate(raw):
            bs, na, h, w, no = p.shape
            yy, xx = torch.meshgrid(torch.arange(h, device=p.device), torch.arange(w, device=p.device), indexing="ij")
            grid = torch.stack((xx, yy), -1).to(p.dtype).view(1, 1, h, w, 2) - .5
            anchor_grid = (self.anchors[i] * self.stride[i]).view(1, na, 1, 1, 2)
            xy, wh, conf = p.sigmoid().split((2, 2, no - 4), -1)
            xy = (xy * 2 + grid) * self.stride[i]
            wh = (wh * 2).square() * anchor_grid
            decoded.append(torch.cat((xy, wh, conf), -1).view(bs, -1, no))
        return torch.cat(decoded, 1), list(raw)


def prepare_detector(detector, ops, dummy):
    """Check original -> BN-folded/operator-prepared equivalence before calibration."""
    source = deepcopy(detector).float().eval()
    with torch.no_grad():
        expected, expected_raw = source(dummy)
    prepared = deepcopy(source).fuse().eval()
    prepare_ops(prepared, ops)
    raw = RawDetector(prepared).eval()
    adapter = DecodedDetector(raw, source).eval()
    with torch.no_grad():
        actual, actual_raw = adapter(dummy)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-4)
        for lhs, rhs in zip(actual_raw, expected_raw):
            torch.testing.assert_close(lhs, rhs, atol=2e-5, rtol=2e-4)
    return raw


def adjust_targets(targets, height):
    """Preserve train.py's existing Y-crop target transform, including its 256-row rule."""
    targets = targets.clone()
    source, offset = {224: (240, 8), 352: (360, 4), 256: (272, 7)}.get(height, (height, 0))
    if height in (224, 352, 256):
        targets[:, 3] = (targets[:, 3] * source - offset).clamp(0, height) / height
        targets[:, 5] = (targets[:, 5] * source).clamp(0, height) / height
    return targets

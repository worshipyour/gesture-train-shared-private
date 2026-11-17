# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Minimal YOLO model code for the provided Conv/C3/SPPF/Concat/Detect configuration."""

import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from models.common import C3, Concat, Conv, SPPF
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, make_divisible
from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, scale_img


class Detect(nn.Module):
    """YOLOv5 detection head."""

    stride = None
    dynamic = False
    export = False

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.empty(0) for _ in range(self.nl)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = inplace

    def calc_feat(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
        return x

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]
                wh = (wh * 2) ** 2 * self.anchor_grid[i]
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        if self.training:
            return x
        pred = torch.cat(z, 1)
        return (pred,) if self.export else (pred, x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2
        y = torch.arange(ny, device=d, dtype=t)
        x = torch.arange(nx, device=d, dtype=t)
        yv, xv = (
            torch.meshgrid(y, x, indexing="ij")
            if torch_1_10
            else torch.meshgrid(y, x)
        )
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (
            (self.anchors[i] * self.stride[i])
            .view((1, self.na, 1, 1, 2))
            .expand(shape)
        )
        return grid, anchor_grid


class BaseModel(nn.Module):
    """Base class for the detection model."""

    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)

    def _forward_once(self, x, profile=False, visualize=False):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f
                ]

            x = m(x)
            y.append(x if m.i in self.save else None)

        return x

    def fuse(self):
        """Fuse Conv2d + BatchNorm2d for inference."""
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, Conv) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_fuse
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        self = super()._apply(fn)
        m = self.model[-1]
        if isinstance(m, Detect):
            if m.stride is not None:
                m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    """YOLOv5 detection model for the provided YAML structure."""

    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        super().__init__()

        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)

        ch = self.yaml["ch"] = self.yaml.get("ch", ch)

        if nc is not None and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc

        if anchors is not None:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = anchors

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml["nc"])]
        self.inplace = self.yaml.get("inplace", True)

        m = self.model[-1]
        if not isinstance(m, Detect):
            raise TypeError(f"This minimal yolo.py expects Detect as the last layer, got {type(m).__name__}")

        # Build strides and normalize anchors.
        # Use the configured input-channel count rather than a hard-coded 1.
        s = 320
        m.inplace = self.inplace
        m.stride = torch.tensor(
            [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))]
        )
        check_anchor_order(m)
        m.anchors /= m.stride.view(-1, 1, 1)
        self.stride = m.stride
        self._initialize_biases()

        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)
        return self._forward_once(x, profile, visualize)

    def _forward_augment(self, x):
        img_size = x.shape[-2:]
        scales = [1, 0.83, 0.67]
        flips = [None, 3, None]
        y = []

        for scale, flip in zip(scales, flips):
            xi = scale_img(
                x.flip(flip) if flip else x,
                scale,
                gs=int(self.stride.max()),
            )
            yi = self._forward_once(xi)[0]
            yi = self._descale_pred(yi, flip, scale, img_size)
            y.append(yi)

        y = self._clip_augmented(y)
        return torch.cat(y, 1), None

    def _descale_pred(self, p, flips, scale, img_size):
        if self.inplace:
            p[..., :4] /= scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]
        else:
            x = p[..., 0:1] / scale
            y = p[..., 1:2] / scale
            wh = p[..., 2:4] / scale
            if flips == 2:
                y = img_size[0] - y
            elif flips == 3:
                x = img_size[1] - x
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        nl = self.model[-1].nl
        g = sum(4**x for x in range(nl))
        e = 1
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))
        y[0] = y[0][:, :-i]
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))
        y[-1] = y[-1][:, i:]
        return y

    def _initialize_biases(self, cf=None):
        m = self.model[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999))
                if cf is None
                else torch.log(cf / cf.sum())
            )
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


# Backwards-compatible name used by YOLOv5 train/val scripts.
Model = DetectionModel


def _resolve_arg(a, nc, anchors):
    """Resolve only the symbolic values used by this YAML; leave activation names as strings."""
    if not isinstance(a, str):
        return a
    if a == "nc":
        return nc
    if a == "anchors":
        return anchors
    if a == "None":
        return None
    if a == "True":
        return True
    if a == "False":
        return False
    return a


def parse_model(d, ch):
    """Parse only Conv, C3, SPPF, Upsample, Concat and Detect layers."""
    LOGGER.info(
        f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  "
        f"{'module':<40}{'arguments':<30}"
    )

    anchors = d["anchors"]
    nc = d["nc"]
    gd = d["depth_multiple"]
    gw = d["width_multiple"]

    na = len(anchors[0]) // 2 if isinstance(anchors, list) else anchors
    no = na * (nc + 5)

    module_map = {
        "Conv": Conv,
        "C3": C3,
        "SPPF": SPPF,
        "nn.Upsample": nn.Upsample,
        "Concat": Concat,
        "Detect": Detect,
    }

    layers = []
    save = []
    c2 = ch[-1]

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        if isinstance(m, str):
            if m not in module_map:
                raise ValueError(
                    f"Unsupported module {m!r} in minimal yolo.py. "
                    f"Allowed: {', '.join(module_map)}"
                )
            m = module_map[m]

        args = [_resolve_arg(a, nc, anchors) for a in args]
        n_ = n = max(round(n * gd), 1) if n > 1 else n

        if m in {Conv, C3, SPPF}:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]

            # Custom C3 signature in the paired minimal common.py is:
            # C3(c1, c2, activation='SiLU', n=1, ...)
            if m is C3:
                args.insert(3, n)
                n = 1

        elif m is nn.Upsample:
            c2 = ch[f]

        elif m is Concat:
            c2 = sum(ch[x] for x in f)

        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
            c2 = ch[f[0]]

        else:
            raise ValueError(f"Unsupported module: {m}")

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = m.__name__ if hasattr(m, "__name__") else type(m).__name__
        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np

        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")

        for x in ([f] if isinstance(f, int) else f):
            if x != -1:
                save.append(x if x >= 0 else x % i)

        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)

import torch


def _squeeze_last(x):
    """Squeeze a trailing singleton dimension produced by box.chunk()."""
    return x.squeeze(-1) if isinstance(x, torch.Tensor) and x.ndim > 1 and x.shape[-1] == 1 else x


def _to_xyxy_and_wh(box, xywh=False, eps=1e-7):
    """Convert boxes to xyxy geometry and return x1, y1, x2, y2, w, h."""
    if xywh:
        x, y, w, h = box.chunk(4, dim=-1)
        w = w.clamp(min=eps)
        h = h.clamp(min=eps)
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
    else:
        x1, y1, x2, y2 = box.chunk(4, dim=-1)
        w = (x2 - x1).clamp(min=eps)
        h = (y2 - y1).clamp(min=eps)

    return x1, y1, x2, y2, w, h


def bbox_mpdiou(
    box1,
    box2,
    xywh=False,
    eps=1e-7,
    hw=1.0,
    mpdiou_norm="enclose_diag2",
    return_raw_iou=False,
):
    """
    Minimum Point Distance IoU (MPDIoU).

    `mpdiou_norm`:
      - enclose_diag2: squared diagonal of the smallest enclosing box
      - box1_diag2: squared diagonal of box1
      - box2_diag2: squared diagonal of box2
      - legacy_hw: supplied `hw`
    """
    b1_x1, b1_y1, b1_x2, b1_y2, w1, h1 = _to_xyxy_and_wh(box1, xywh, eps)
    b2_x1, b2_y1, b2_x2, b2_y2, w2, h2 = _to_xyxy_and_wh(box2, xywh, eps)

    inter_w = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(min=0)
    inter_h = (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(min=0)
    inter = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - inter + eps
    raw_iou = inter / union

    d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
    d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2

    if mpdiou_norm == "enclose_diag2":
        cw = (torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)).clamp(min=eps)
        ch = (torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)).clamp(min=eps)
        norm = cw ** 2 + ch ** 2 + eps
    elif mpdiou_norm == "box1_diag2":
        norm = w1 ** 2 + h1 ** 2 + eps
    elif mpdiou_norm == "box2_diag2":
        norm = w2 ** 2 + h2 ** 2 + eps
    elif mpdiou_norm == "legacy_hw":
        norm = torch.as_tensor(hw, device=box1.device, dtype=box1.dtype) + eps
    else:
        raise ValueError(f"Unsupported mpdiou_norm: {mpdiou_norm}")

    score = _squeeze_last(raw_iou - d1 / norm - d2 / norm)
    raw_iou = _squeeze_last(raw_iou)

    return (score, raw_iou) if return_raw_iou else score


def bbox_iou_special(
    box1,
    box2,
    xywh=False,
    mpdiou=True,
    hw=1.0,
    eps=1e-7,
    mpdiou_norm="enclose_diag2",
    return_raw_iou=False,
    **kwargs,
):
    """
    Backward-compatible wrapper.

    This cleaned file intentionally retains only MPDIoU as an extra IoU loss.
    Standard IoU/CIoU should use utils.metrics.bbox_iou().
    """
    if not mpdiou:
        raise ValueError(
            "The cleaned bbox_iou_special() only supports MPDIoU. "
            "Use utils.metrics.bbox_iou() for standard YOLOv5 IoU/CIoU."
        )

    return bbox_mpdiou(
        box1,
        box2,
        xywh=xywh,
        eps=eps,
        hw=hw,
        mpdiou_norm=mpdiou_norm,
        return_raw_iou=return_raw_iou,
    )


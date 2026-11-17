# YOLOv5 loss functions
#
# Retained:
#   - standard BCE / Focal / Quality-Focal helpers
#   - standard YOLOv5 CIoU box loss
#   - NWD
#   - MPDIoU
#   - prediction-level distillation
#
# Removed:
#   - WIoU / SIoU / ShapeIoU / PIoU / Inner-IoU / Focaler-IoU / Alpha-IoU
#   - GWD / KLD and other experimental bbox-loss branches

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.IoUs import bbox_mpdiou
from utils.metrics import bbox_iou, nwd
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps


class QFL(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulation_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulation_factor
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BCEBlurWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=2.0, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class VFLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "mean"

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        focal_weight = (
            true * (true > 0.0).float()
            + self.alpha * (pred_prob - true).abs().pow(self.gamma) * (true <= 0.0).float()
        )
        loss *= focal_weight
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class QFocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ComputeLoss:
    """
    Standard YOLOv5 detection loss with two optional additions:
      - MPDIoU instead of the default CIoU box score
      - NWD blended with the active IoU score
    """

    sort_obj_iou = False

    def __init__(
        self,
        model,
        autobalance=False,
        aspect_penalty=False,
        aspect_penalty_weight=1.0,
    ):
        # The two aspect_penalty args are accepted only for old call-site compatibility.
        # The custom aspect-ratio penalty itself is removed.
        device = next(model.parameters()).device
        h = model.hyp

        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h["cls_pw"]], device=device)
        )
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h["obj_pw"]], device=device)
        )

        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))

        g = h["fl_gamma"]
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]

        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            det.nl,
            [4.0, 1.0, 0.25, 0.06, 0.02],
        )
        self.ssi = list(det.stride).index(16) if autobalance else 0

        self.BCEcls = BCEcls
        self.BCEobj = BCEobj
        self.gr = 1.0
        self.hyp = h
        self.autobalance = autobalance

        for k in ("na", "nc", "nl", "anchors"):
            setattr(self, k, getattr(det, k))

        self.stride = det.stride

    def _use_mpdiou(self):
        return bool(self.hyp.get("MPDIoU", self.hyp.get("mpdiou", False)))

    def _get_mpdiou_hw(self, pi=None):
        hw = self.hyp.get("mpdiou_hw", self.hyp.get("hw", None))
        if hw is not None:
            return hw

        mode = self.hyp.get("mpdiou_hw_mode", "feature_diag2")
        if pi is not None and mode == "feature_diag2":
            ny, nx = pi.shape[2], pi.shape[3]
            return float(nx ** 2 + ny ** 2)
        if pi is not None and mode == "feature_area":
            ny, nx = pi.shape[2], pi.shape[3]
            return float(nx * ny)
        return 1.0

    def _get_iou_for_loss_and_obj(self, pbox, tbox, pi=None):
        if self._use_mpdiou():
            mpdiou, raw_iou = bbox_mpdiou(
                pbox,
                tbox,
                xywh=True,
                eps=float(self.hyp.get("iou_eps", 1e-7)),
                hw=self._get_mpdiou_hw(pi),
                mpdiou_norm=self.hyp.get("mpdiou_norm", "enclose_diag2"),
                return_raw_iou=True,
            )
            # Regression uses MPDIoU; objectness target stays ordinary IoU.
            return mpdiou, raw_iou

        # Original YOLOv5 CIoU path.
        iou = bbox_iou(pbox.T, tbox, x1y1x2y2=False, CIoU=True)
        return iou, iou

    def _compute_nwd(self, pbox, tbox_i, layer_index):
        """
        NWD is computed in pixel coordinates.
        Grid-space boxes are multiplied by the corresponding detection stride.
        """
        nwd_constant = float(self.hyp.get("nwd_constant", 12.8))
        nwd_eps = float(self.hyp.get("nwd_eps", 1e-7))

        stride = self.stride[layer_index].to(pbox.device).type_as(pbox)
        pbox_pix = pbox * stride
        tbox_pix = tbox_i * stride

        return nwd(
            pbox_pix,
            tbox_pix,
            scale1=nwd_constant,
            eps=nwd_eps,
        )

    @torch.no_grad()
    def _maybe_sort_by_iou(self, b, a, gj, gi, score_iou):
        if not self.sort_obj_iou:
            return b, a, gj, gi, score_iou
        sort_id = torch.argsort(score_iou)
        return b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]

    def __call__(self, p, targets):
        device = targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)

        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)

            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]

                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2.0) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)

                iou_for_loss, iou_for_obj = self._get_iou_for_loss_and_obj(
                    pbox,
                    tbox[i],
                    pi=pi,
                )

                if self.hyp.get("nwd", False):
                    nwd_alpha = float(self.hyp.get("nwd_alpha", 0.5))
                    nwd_score = self._compute_nwd(pbox, tbox[i], layer_index=i)
                    lbox += (
                        (1.0 - nwd_alpha) * (1.0 - iou_for_loss).mean()
                        + nwd_alpha * (1.0 - nwd_score).mean()
                    )
                else:
                    lbox += (1.0 - iou_for_loss).mean()

                score_iou = iou_for_obj.detach().clamp(0.0, 1.0).type(tobj.dtype)

                b, a, gj, gi, score_iou = self._maybe_sort_by_iou(
                    b,
                    a,
                    gj,
                    gi,
                    score_iou,
                )

                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou

                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]

            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]

        bs = p[0].shape[0]
        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []

        gain = torch.ones(7, device=targets.device)

        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5
        off = torch.tensor(
            [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
            device=targets.device,
        ).float() * g

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(p[i].shape, device=targets.device)[[3, 2, 3, 2]]
            t = targets * gain

            if nt:
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1.0 / r).max(2)[0] < self.hyp["anchor_t"]
                t = t[j]

                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy

                j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                l, m = ((gxi % 1.0 < g) & (gxi > 1.0)).T

                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]

            gij = (gxy - offsets).long()
            gi, gj = gij.T
            a = t[:, 6].long()

            indices.append(
                (
                    b,
                    a,
                    gj.clamp_(0, shape[2] - 1),
                    gi.clamp_(0, shape[3] - 1),
                )
            )
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)

        return tcls, tbox, indices, anch


class ComputeLoss_dis(ComputeLoss):
    """
    YOLOv5 supervised detection loss + prediction-level distillation.

    Distillation is intentionally retained:
      - box: weighted MSE on decoded xywh
      - obj: BCE(student objectness logit, teacher objectness probability)
      - cls: temperature-scaled KL
    """

    def __init__(
        self,
        model,
        autobalance=False,
        aspect_penalty=False,
        aspect_penalty_weight=1.0,
    ):
        super().__init__(
            model,
            autobalance=autobalance,
            aspect_penalty=aspect_penalty,
            aspect_penalty_weight=aspect_penalty_weight,
        )

        self.distill_weight = float(
            self.hyp.get("distill", self.hyp.get("dis", self.hyp.get("kd", 0.5)))
        )
        self.distill_box_weight = float(self.hyp.get("distill_box", 1.0))
        self.distill_obj_weight = float(self.hyp.get("distill_obj", 1.0))
        self.distill_cls_weight = float(self.hyp.get("distill_cls", 0.5))
        self.distill_temp = float(
            self.hyp.get("distill_temp", self.hyp.get("kd_T", 4.0))
        )
        self.distill_obj_thr = float(self.hyp.get("distill_obj_thr", 0.0))
        self.distill_obj_power = float(self.hyp.get("distill_obj_power", 1.0))

        self.kd_bce = nn.BCEWithLogitsLoss(reduction="none")
        self.eps = 1e-9

    def _resize_teacher_pred(self, pt, ps):
        if pt.shape == ps.shape:
            return pt

        bs, na, nyt, nxt, no = pt.shape
        bs_s, na_s, nys, nxs, no_s = ps.shape

        assert bs == bs_s, f"teacher/student batch mismatch: {bs} vs {bs_s}"
        assert na == na_s, f"teacher/student anchor num mismatch: {na} vs {na_s}"
        assert no == no_s, f"teacher/student output dim mismatch: {no} vs {no_s}"

        pt = pt.permute(0, 1, 4, 2, 3).contiguous()
        pt = pt.view(bs, na * no, nyt, nxt)

        pt = F.interpolate(
            pt.float(),
            size=(nys, nxs),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=ps.dtype)

        pt = pt.view(bs, na, no, nys, nxs)
        return pt.permute(0, 1, 3, 4, 2).contiguous()

    def _decode_box(self, pi, anchors_i):
        anchors_i = anchors_i.to(pi.device).type_as(pi).view(1, self.na, 1, 1, 2)
        pxy = pi[..., 0:2].sigmoid() * 2.0 - 0.5
        pwh = (pi[..., 2:4].sigmoid() * 2.0) ** 2 * anchors_i
        return torch.cat((pxy, pwh), dim=-1)

    def _weighted_mean(self, loss, weight=None):
        if weight is None:
            return loss.mean()

        weight = weight.to(loss.device).type_as(loss)
        base_numel = weight.numel()

        while weight.ndim < loss.ndim:
            weight = weight.unsqueeze(-1)

        expand_factor = loss.numel() / max(base_numel, 1)
        denom = weight.sum().clamp_min(self.eps) * expand_factor
        return (loss * weight).sum() / denom

    def _distill_loss(self, p, p_teacher):
        device = p[0].device

        ldis_box = torch.zeros(1, device=device)
        ldis_obj = torch.zeros(1, device=device)
        ldis_cls = torch.zeros(1, device=device)

        for i, (pi, pti) in enumerate(zip(p, p_teacher)):
            pti = self._resize_teacher_pred(pti.detach(), pi)

            t_obj = pti[..., 4].sigmoid()
            obj_weight = t_obj.pow(self.distill_obj_power)

            if self.distill_obj_thr > 0:
                obj_weight = obj_weight * (t_obj > self.distill_obj_thr).float()

            if self.distill_box_weight > 0:
                s_box = self._decode_box(pi, self.anchors[i])
                t_box = self._decode_box(pti, self.anchors[i])
                box_loss = (s_box - t_box).pow(2)
                ldis_box += self._weighted_mean(box_loss, obj_weight)

            if self.distill_obj_weight > 0:
                obj_loss = self.kd_bce(pi[..., 4], t_obj)
                ldis_obj += obj_loss.mean() * self.balance[i]

            if self.nc > 1 and self.distill_cls_weight > 0:
                T = self.distill_temp

                s_log_prob = F.log_softmax(pi[..., 5:] / T, dim=-1)
                t_prob = F.softmax(pti[..., 5:] / T, dim=-1)

                cls_loss = F.kl_div(
                    s_log_prob,
                    t_prob,
                    reduction="none",
                ).sum(dim=-1) * (T * T)

                ldis_cls += self._weighted_mean(cls_loss, obj_weight)

        ldis = (
            self.distill_box_weight * ldis_box
            + self.distill_obj_weight * ldis_obj
            + self.distill_cls_weight * ldis_cls
        )

        return ldis * self.distill_weight

    def __call__(self, p, p_teacher, targets):
        loss_det, loss_items = super().__call__(p, targets)

        if p_teacher is None or self.distill_weight <= 0:
            ldis = torch.zeros(1, device=targets.device)
            return loss_det, torch.cat((loss_items, ldis.detach()))

        ldis = self._distill_loss(p, p_teacher)
        bs = p[0].shape[0]

        loss = loss_det + ldis * bs
        loss_items = torch.cat((loss_items, ldis.detach().reshape(1)))

        return loss, loss_items

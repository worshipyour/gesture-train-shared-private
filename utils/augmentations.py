# a Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Image augmentation functions."""

import math
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box, xywhn2xyxy
from utils.metrics import bbox_ioa

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation


class Albumentations:
    """Provides optional data augmentation for YOLOv5 using Albumentations library if installed."""

    def __init__(self, size=[640,640]):
        """Initializes Albumentations class for optional data augmentation in YOLOv5 with specified input size."""
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            T = [
                A.RandomResizedCrop(height=size[0], width=size[1], scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, im, labels, p=1.0):
        """Applies transformations to an image and labels with probability `p`, returning updated image and labels."""
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new["image"], np.array([[c, *b] for c, b in zip(new["class_labels"], new["bboxes"])])
        return im, labels


def normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
    """
    Applies ImageNet normalization to RGB images in BCHW format, modifying them in-place if specified.

    Example: y = (x - mean) / std
    """
    return TF.normalize(x, mean, std, inplace=inplace)


def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Reverses ImageNet normalization for BCHW format RGB images by applying `x = x * std + mean`."""
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    """Applies HSV color-space augmentation to an image with random gains for hue, saturation, and value."""
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def hist_equalize(im, clahe=True, bgr=False):
    """Equalizes image histogram, with optional CLAHE, for BGR or RGB image with shape (n,m,3) and range 0-255."""
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    """
    Replicates half of the smallest object labels in an image for data augmentation.

    Returns augmented image and labels.
    """
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[: round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def random_perspective_normal(
    im, targets=(), segments=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0)
):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    """Applies random perspective transformation to an image, modifying the image and corresponding labels."""
    height = im.shape[0] + border[0] * 2 * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2 * 2
    
    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    if n := len(targets):
        use_segments = any(x.any() for x in segments) and len(segments) == n
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets

def random_perspective(
    im,
    targets=(),
    segments=(),
    degrees=10,
    translate=0.1,
    scale_down=0.5,
    scale_up=0.5,
    shear=10,
    perspective=0.0,
    scale_bias=1.0,
    border=(0, 0),
    reject_partial=False,
    return_reject=False,
):
    """
    Applies random perspective transformation to an image, modifying the image and corresponding labels.

    改动版：
        1. scale:
           如果随机缩放导致 bbox 无法完整保留在输出图像中，
           则计算当前 rotation / shear / perspective 下的最大可行缩放比例，
           并在可行范围内重新随机采样 scale。

        2. translate:
           如果随机平移导致 bbox 部分移出屏幕，
           则计算 x/y 四个方向不出界的极限平移区间，
           并在该区间内重新随机采样 translation。

        3. reject_partial:
           现在主要作为兜底。
           如果在最小 scale 下仍然无法让 bbox 完整保留，
           则根据 reject_partial / return_reject 返回原图或继续原 YOLO clip 行为。
    """

    # ------------------------------------------------------------
    # Backup
    # ------------------------------------------------------------
    im0 = im.copy()
    targets0 = targets.copy() if targets is not None and len(targets) else targets

    # YOLOv5 原逻辑：border[0] 对 height，border[1] 对 width
    height = im.shape[0] + border[0] * 4
    width = im.shape[1] + border[1] * 4

    n = len(targets) if targets is not None else 0

    # ------------------------------------------------------------
    # Center
    # ------------------------------------------------------------
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2
    C[1, 2] = -im.shape[0] / 2

    # ------------------------------------------------------------
    # Perspective
    # ------------------------------------------------------------
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    # ------------------------------------------------------------
    # Shear
    # ------------------------------------------------------------
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    # ------------------------------------------------------------
    # Helpers for adaptive scale / translation
    # ------------------------------------------------------------
    def _get_limit_points():
        """
        用于判断 bbox 是否出界的源点。
        优先使用 segment 点；否则使用 bbox 四角点。
        """
        if n == 0:
            return None

        use_segments_for_limit = (
            segments is not None
            and len(segments) == n
            and any(np.asarray(x).size and np.asarray(x).any() for x in segments)
        )

        if use_segments_for_limit:
            pts = []
            for seg in segments:
                seg = np.asarray(seg)
                if seg.size:
                    pts.append(seg[:, :2])
            if len(pts):
                return np.concatenate(pts, axis=0).astype(np.float32)

        xy = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        return xy.astype(np.float32)

    limit_points = _get_limit_points()

    def _transform_points_no_translation(s_value):
        """
        使用给定 scale，但不加最终 translation。
        返回经过 C -> P -> R -> S 之后的点。
        注意：最终 T 是加性平移，因此可以后面单独求 tx/ty 范围。
        """
        R_tmp = np.eye(3)
        R_tmp[:2] = cv2.getRotationMatrix2D(
            angle=a,
            center=(0, 0),
            scale=s_value,
        )

        M_no_T = S @ R_tmp @ P @ C

        xy = np.ones((len(limit_points), 3), dtype=np.float32)
        xy[:, :2] = limit_points

        xy = xy @ M_no_T.T

        if perspective:
            z = xy[:, 2:3]
            if np.any(np.abs(z) < 1e-9):
                return None
            xy = xy[:, :2] / z
        else:
            xy = xy[:, :2]

        if not np.isfinite(xy).all():
            return None

        return xy

    def _valid_translation_interval(s_value):
        """
        对给定 scale，计算是否存在某个 tx, ty，
        能让所有 bbox/segment 点完整落在 [0,width] x [0,height] 内。

        返回：
            valid,
            tx_min, tx_max,
            ty_min, ty_max
        """
        if limit_points is None or len(limit_points) == 0:
            return True, 0.0, width, 0.0, height

        xy = _transform_points_no_translation(s_value)
        if xy is None:
            return False, None, None, None, None

        x_min = float(xy[:, 0].min())
        y_min = float(xy[:, 1].min())
        x_max = float(xy[:, 0].max())
        y_max = float(xy[:, 1].max())

        # 需要满足：
        #   0 <= x + tx <= width
        #   0 <= y + ty <= height
        tx_min = -x_min
        tx_max = width - x_max
        ty_min = -y_min
        ty_max = height - y_max

        valid = (tx_min <= tx_max) and (ty_min <= ty_max)
        return valid, tx_min, tx_max, ty_min, ty_max

    # ------------------------------------------------------------
    # Rotation and adaptive Scale
    # ------------------------------------------------------------
    a = random.uniform(-degrees, degrees)

    scale_min = max(1e-6, 1.0 - scale_down)
    scale_max = 1.0 + scale_up

    if scale_bias != 1.0:
        r = random.random()
        s = (1 - scale_down) + (2 * scale_up) * math.pow(r, scale_bias)
    else:
        s = random.uniform(scale_min, scale_max)

    if n:
        valid_s, _, _, _, _ = _valid_translation_interval(s)

        if not valid_s:
            # 如果当前随机 scale 不可行，则搜索当前几何条件下的最大可行 scale
            low = scale_min
            high = s

            valid_low, _, _, _, _ = _valid_translation_interval(low)

            if valid_low:
                # 二分找 [scale_min, s] 内最大的可行 scale
                for _ in range(32):
                    mid = (low + high) / 2.0
                    valid_mid, _, _, _, _ = _valid_translation_interval(mid)
                    if valid_mid:
                        low = mid
                    else:
                        high = mid

                max_valid_s = low

                # 如果原来是放大导致越界，则在 [1, max_valid_s] 内重新采样
                # 如果 max_valid_s < 1，则只能在 [scale_min, max_valid_s] 内采样
                if max_valid_s >= 1.0:
                    s = random.uniform(1.0, max_valid_s)
                else:
                    s = random.uniform(scale_min, max_valid_s)
            else:
                # 连最小 scale 都无法保证不出界，说明 rotation/shear/perspective 太极端
                # 这里作为兜底：如果用户要求 reject，则返回原图
                if reject_partial:
                    if return_reject:
                        return im0, targets0, True
                    return im0, targets0
                # 否则保持原随机 scale，后面走原 YOLO clip 行为

    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # ------------------------------------------------------------
    # Adaptive Translation
    # ------------------------------------------------------------
    T = np.eye(3)

    if n:
        valid_t, tx_min_valid, tx_max_valid, ty_min_valid, ty_max_valid = _valid_translation_interval(s)

        if valid_t:
            # YOLO 原始随机平移范围
            tx_min_yolo = (0.5 - translate) * width
            tx_max_yolo = (0.5 + translate) * width
            ty_min_yolo = (0.5 - translate) * height
            ty_max_yolo = (0.5 + translate) * height

            # 先按原 YOLO 方式随机采样
            tx = random.uniform(tx_min_yolo, tx_max_yolo)
            ty = random.uniform(ty_min_yolo, ty_max_yolo)

            # 如果这次随机平移会导致 bbox 出界，则重新限制采样范围
            out_by_translation = (
                tx < tx_min_valid
                or tx > tx_max_valid
                or ty < ty_min_valid
                or ty > ty_max_valid
            )

            if out_by_translation:
                # 优先取 YOLO 平移范围 和 bbox 不出界范围 的交集
                tx_low = max(tx_min_yolo, tx_min_valid)
                tx_high = min(tx_max_yolo, tx_max_valid)
                ty_low = max(ty_min_yolo, ty_min_valid)
                ty_high = min(ty_max_yolo, ty_max_valid)

                # 如果交集为空，说明原 translate 范围太小，
                # 但为了保证 bbox 不出界，直接使用可行区间。
                if tx_low > tx_high:
                    tx_low, tx_high = tx_min_valid, tx_max_valid
                if ty_low > ty_high:
                    ty_low, ty_high = ty_min_valid, ty_max_valid

                tx = random.uniform(tx_low, tx_high)
                ty = random.uniform(ty_low, ty_high)

            T[0, 2] = tx
            T[1, 2] = ty

        else:
            # 给定当前 scale 后，根本不存在任何 tx/ty 能保证不出界
            if reject_partial:
                if return_reject:
                    return im0, targets0, True
                return im0, targets0

            # fallback: 原 YOLO translation
            T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
            T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    else:
        # 没有 bbox 时保持原 YOLO 行为
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    # Combined rotation matrix
    M = T @ S @ R @ P @ C

    # ------------------------------------------------------------
    # Warp image
    # ------------------------------------------------------------
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            im = cv2.warpPerspective(
                im,
                M,
                dsize=(width, height),
                borderValue=(114, 114, 114),
            )
        else:
            im = cv2.warpAffine(
                im,
                M[:2],
                dsize=(width, height),
                borderValue=(114, 114, 114),
            )

    # ------------------------------------------------------------
    # Warp labels
    # ------------------------------------------------------------
    n = len(targets) if targets is not None else 0

    if n:
        use_segments = any(x.any() for x in segments) and len(segments) == n
        new = np.zeros((n, 4))

        if use_segments:
            # ----------------------------------------------------
            # Segment mode
            # ----------------------------------------------------
            segments = resample_segments(segments)

            raw_new = np.zeros((n, 4))

            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment

                xy = xy @ M.T
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]

                x = xy[:, 0]
                y = xy[:, 1]

                raw_new[i] = np.array(
                    [
                        x.min(),
                        y.min(),
                        x.max(),
                        y.max(),
                    ]
                )

            new = raw_new.copy()

        else:
            # ----------------------------------------------------
            # Box mode
            # ----------------------------------------------------
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)

            xy = xy @ M.T
            xy = (
                xy[:, :2] / xy[:, 2:3]
                if perspective
                else xy[:, :2]
            ).reshape(n, 8)

            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]

            raw_new = np.concatenate(
                (
                    x.min(1),
                    y.min(1),
                    x.max(1),
                    y.max(1),
                )
            ).reshape(4, n).T

            new = raw_new.copy()

        # --------------------------------------------------------
        # Strict reject before clip
        # --------------------------------------------------------
        # 正常情况下，前面的 adaptive scale / translation 已经会避免出界。
        # 这里保留 reject_partial 作为兜底，防止 perspective 数值异常等情况。
        if reject_partial:
            eps = 1e-6

            invalid_or_outside = (
                (raw_new[:, 0] < -eps)
                | (raw_new[:, 1] < -eps)
                | (raw_new[:, 2] > width + eps)
                | (raw_new[:, 3] > height + eps)
                | (raw_new[:, 2] <= raw_new[:, 0] + eps)
                | (raw_new[:, 3] <= raw_new[:, 1] + eps)
            )

            if invalid_or_outside.any():
                if return_reject:
                    return im0, targets0, True
                return im0, targets0

        # --------------------------------------------------------
        # Original YOLOv5 clip behavior
        # --------------------------------------------------------
        # 这里仍然保留 clip，主要是为了处理极小数值误差。
        # 正常 adaptive 之后，clip 不应该再大幅改变 bbox。
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # --------------------------------------------------------
        # Filter candidates
        # --------------------------------------------------------
        i = box_candidates(
            box1=targets[:, 1:5].T * s,
            box2=new.T,
            area_thr=0.01 if use_segments else 0.10,
        )

        targets = targets[i]
        targets[:, 1:5] = new[i]

    # ------------------------------------------------------------
    # Return
    # ------------------------------------------------------------
    if return_reject:
        return im, targets, False

    return im, targets


def copy_paste(im, labels, segments, p=0.5):
    """
    Applies Copy-Paste augmentation by flipping and merging segments and labels on an image.

    Details at https://arxiv.org/abs/2012.07177.
    """
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (1, 1, 1), cv2.FILLED)

        result = cv2.flip(im, 1)  # augment segments (flip left-right)
        i = cv2.flip(im_new, 1).astype(bool)
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments


def cutout(im, labels, p=0.5):
    """
    Applies cutout augmentation to an image with optional label adjustment, using random masks of varying sizes.

    Details at https://arxiv.org/abs/1708.04552.
    """
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, xywhn2xyxy(labels[:, 1:5], w, h))  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(im, labels, im2, labels2):
    """
    Applies MixUp augmentation by blending images and labels.

    See https://arxiv.org/pdf/1710.09412.pdf for details.
    """
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
    """
    Filters bounding box candidates by minimum width-height threshold `wh_thr` (pixels), aspect ratio threshold
    `ar_thr`, and area ratio threshold `area_thr`.

    box1(4,n) is before augmentation, box2(4,n) is after augmentation.
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def classify_albumentations(
    augment=True,
    size=224,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.0 / 0.75),  # 0.75, 1.33
    hflip=0.5,
    vflip=0.0,
    jitter=0.4,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    auto_aug=False,
):
    # YOLOv5 classification Albumentations (optional, only used if package is installed)
    """Sets up and returns Albumentations transforms for YOLOv5 classification tasks depending on augmentation
    settings.
    """
    prefix = colorstr("albumentations: ")
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        check_version(A.__version__, "1.0.3", hard=True)  # version requirement
        if augment:  # Resize and crop
            T = [A.RandomResizedCrop(height=size, width=size, scale=scale, ratio=ratio)]
            if auto_aug:
                # TODO: implement AugMix, AutoAug & RandAug in albumentation
                LOGGER.info(f"{prefix}auto augmentations are currently not supported")
            else:
                if hflip > 0:
                    T += [A.HorizontalFlip(p=hflip)]
                if vflip > 0:
                    T += [A.VerticalFlip(p=vflip)]
                if jitter > 0:
                    color_jitter = (float(jitter),) * 3  # repeat value for brightness, contrast, saturation, 0 hue
                    T += [A.ColorJitter(*color_jitter, 0)]
        else:  # Use fixed crop for eval set (reproducibility)
            T = [A.SmallestMaxSize(max_size=size), A.CenterCrop(height=size, width=size)]
        T += [A.Normalize(mean=mean, std=std), ToTensorV2()]  # Normalize and convert to Tensor
        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        return A.Compose(T)

    except ImportError:  # package not installed, skip
        LOGGER.warning(f"{prefix}⚠️ not found, install with `pip install albumentations` (recommended)")
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")


def classify_transforms(size=224):
    """Applies a series of transformations including center crop, ToTensor, and normalization for classification."""
    assert isinstance(size, int), f"ERROR: classify_transforms size {size} must be integer, not (list, tuple)"
    # T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose([CenterCrop(size), ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class LetterBox:
    """Resizes and pads images to specified dimensions while maintaining aspect ratio for YOLOv5 preprocessing."""

    def __init__(self, size=(640, 640), auto=False, stride=32):
        """Initializes a LetterBox object for YOLOv5 image preprocessing with optional auto sizing and stride
        adjustment.
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):
        """
        Resizes and pads input image `im` (HWC format) to specified dimensions, maintaining aspect ratio.

        im = np.array HWC
        """
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old
        h, w = round(imh * r), round(imw * r)  # resized image
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else self.h, self.w
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
        im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


class CenterCrop:
    """Applies center crop to an image, resizing it to the specified size while maintaining aspect ratio."""

    def __init__(self, size=640):
        """Initializes CenterCrop for image preprocessing, accepting single int or tuple for size, defaults to 640."""
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):
        """
        Applies center crop to the input image and resizes it to a specified size, maintaining aspect ratio.

        im = np.array HWC
        """
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top : top + m, left : left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


class ToTensor:
    """Converts BGR np.array image from HWC to RGB CHW format, normalizes to [0, 1], and supports FP16 if half=True."""

    def __init__(self, half=False):
        """Initializes ToTensor for YOLOv5 image preprocessing, with optional half precision (half=True for FP16)."""
        super().__init__()
        self.half = half

    def __call__(self, im):
        """
        Converts BGR np.array image from HWC to RGB CHW format, and normalizes to [0, 1], with support for FP16 if
        `half=True`.

        im = np.array HWC in BGR order
        """
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im

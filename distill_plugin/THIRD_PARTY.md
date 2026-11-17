# FGD attribution

`fgd.py` adapts the foreground/background attention and global relation losses from
Yang et al., FGD, CVPR 2022: https://github.com/yzd-v/FGD . The original loss is
Apache-2.0 licensed; the full license is in `licenses/FGD-Apache-2.0.txt`.

This adaptation was carried over from the existing local distillation tool. It
removes MMDetection/MMCV dependencies, converts normalized YOLO boxes into masks,
handles empty targets, clips mask bounds, and constructs several feature levels.
The runtime adds temporary feature hooks, frozen-teacher execution, checkpoints
and logging for this custom YOLOv5 repository. It is not a claim to reproduce the
paper's COCO results or to guarantee gains on the user's new gesture dataset.

Original model/training source retains its existing Ultralytics AGPL-3.0 notices.

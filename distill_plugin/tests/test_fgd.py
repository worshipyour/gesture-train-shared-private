import copy
import importlib.util
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch
import yaml

from distill_plugin.fgd import FGD, foreground_masks
from distill_plugin.runtime import FGDTrainer, validate_teacher
from models.yolo import Model
from utils.loss import ComputeLoss

ROOT = Path(__file__).resolve().parents[2]
STUDENT = ROOT / "models/yolov5s-cls13-2.2m-Mul8Channel025_eNPU_ReLU.yaml"
TEACHER = ROOT / "distill_plugin/configs/yolov5m-gesture-eNPU-ReLU.yaml"
HYP = ROOT / "data/hyps/hyps.full_parameters_final20260622_focaler_scale_0.7_0.7_MPDIoU.yaml"


class FGDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.student = Model(str(STUDENT), ch=1, nc=17)
        cls.teacher = Model(str(TEACHER), ch=1, nc=17)
        cls.teacher.names = [str(i) for i in range(17)]
        cls.temp = tempfile.TemporaryDirectory()
        cls.teacher_path = Path(cls.temp.name) / "teacher.pt"
        cls.teacher_path.write_bytes(b"test teacher identity")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_only_teacher_scaling_changes(self):
        student, teacher = yaml.safe_load(STUDENT.read_text()), yaml.safe_load(TEACHER.read_text())
        self.assertEqual((teacher.pop("depth_multiple"), teacher.pop("width_multiple")), (0.67, 0.75))
        student.pop("depth_multiple")
        student.pop("width_multiple")
        self.assertEqual(student, teacher)
        self.assertGreater(sum(p.numel() for p in self.teacher.parameters()),
                           sum(p.numel() for p in self.student.parameters()) * 5)

    def test_masks_empty_overlap_and_border(self):
        fg, bg = foreground_masks(torch.empty(0, 6), 2, 4, 4, torch.device("cpu"))
        self.assertEqual(float(fg.sum()), 0)
        torch.testing.assert_close(bg.sum((1, 2)), torch.ones(2))
        boxes = torch.tensor([[0, 0, .5, .5, .5, .5], [0, 0, .5, .5, .5, .5],
                              [1, 0, 0, 0, .2, .2]])
        fg, bg = foreground_masks(boxes, 2, 4, 4, torch.device("cpu"))
        self.assertAlmostEqual(float(fg[0, 1, 1]), 1/9, places=6)
        self.assertTrue(torch.isfinite(fg).all())
        self.assertTrue(torch.all(bg[fg > 0] == 0))

    def test_fgd_backward_and_no_persistent_hooks(self):
        student, teacher = copy.deepcopy(self.student), copy.deepcopy(self.teacher)
        student.train()
        student.hyp = yaml.safe_load(HYP.read_text())
        student.hyp["label_smoothing"] = 0.0
        keys = tuple(student.state_dict())
        validate_teacher(teacher, student, student.names)
        trainer = FGDTrainer(student, teacher, self.teacher_path, warmup_epochs=0)
        before = trainer.criterion.levels[0].align.weight.detach().clone()
        optimizer = torch.optim.SGD(list(student.parameters()) + list(trainer.criterion.parameters()), lr=.001)
        images = torch.rand(2, 1, 256, 320)
        targets = torch.tensor([[0, 0, .5, .5, .25, .25], [1, 1, .3, .4, .2, .3]])
        with trainer.student_forward():
            pred = student(images)
        det, _ = ComputeLoss(student)(pred, targets)
        kd = trainer.loss(images, targets, 1)
        (det + kd * images.shape[0]).backward()
        self.assertTrue(torch.isfinite(kd))
        self.assertGreater(float(kd), 0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertGreater(float(student.model[0].conv.weight.grad.abs().sum()), 0)
        self.assertGreater(float(trainer.criterion.levels[0].align.weight.grad.abs().sum()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, trainer.criterion.levels[0].align.weight))
        self.assertEqual(keys, tuple(student.state_dict()))
        self.assertFalse(student.model[-1]._forward_pre_hooks)
        self.assertFalse(teacher.model[-1]._forward_pre_hooks)
        self.assertIsNone(trainer._student_features)
        # Ordinary student serialization is independent of teacher and adapters.
        clone = copy.deepcopy(student).eval()
        self.assertFalse(clone.model[-1]._forward_pre_hooks)
        with torch.no_grad():
            self.assertTrue(torch.isfinite(clone(images)[0]).all())

    def test_adapter_checkpoint_and_teacher_identity(self):
        rng_before = torch.get_rng_state().clone()
        first = FGDTrainer(self.student, copy.deepcopy(self.teacher), self.teacher_path)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        second = FGDTrainer(self.student, copy.deepcopy(self.teacher), self.teacher_path)
        second.load_state_dict(first.state_dict())
        for key, value in first.criterion.state_dict().items():
            torch.testing.assert_close(second.criterion.state_dict()[key], value)
        bad = dict(first.state_dict(), teacher_sha256="wrong")
        with self.assertRaises(ValueError):
            second.load_state_dict(bad)

    def test_hook_removed_when_forward_fails(self):
        trainer = FGDTrainer(self.student, copy.deepcopy(self.teacher), self.teacher_path)
        with self.assertRaises(RuntimeError):
            with trainer.student_forward():
                raise RuntimeError("injected forward error")
        self.assertFalse(self.student.model[-1]._forward_pre_hooks)

    def test_wrong_class_order_rejected(self):
        teacher = copy.deepcopy(self.teacher)
        teacher.names = list(reversed(teacher.names))
        with self.assertRaises(ValueError):
            validate_teacher(teacher, self.student, self.student.names)

    def test_launcher_uses_equal_student_conditions(self):
        spec = importlib.util.spec_from_file_location("launcher", ROOT / "distill_plugin/start_2026-11-17.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from argparse import Namespace
        args = Namespace(run_dir=Path("/tmp/example"), weights=Path("/tmp/epoch76.pt"), data=Path("data.yaml"),
                         hyp=HYP, teacher_init="", epochs=200, student_batch=128, teacher_batch=16,
                         workers=8, seed=42, fgd_weight=.005)
        baseline = module.command(args, "baseline")
        kd = module.command(args, "fgd", Path("/tmp/teacher.pt"))
        for flag in ("--weights", "--data", "--cfg", "--hyp", "--epochs", "--batch-size", "--imgsz", "--seed"):
            self.assertEqual(baseline[baseline.index(flag)+1], kd[kd.index(flag)+1])

    def test_gpu_lock_and_busy_device_rejected(self):
        spec = importlib.util.spec_from_file_location("launcher_locks", ROOT / "distill_plugin/start_2026-11-17.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with mock.patch.object(module.subprocess, "run") as query:
            query.return_value.stdout = "12345\n"
            with self.assertRaises(RuntimeError):
                module.reserve_gpu(9876)  # mock query, no hardware access
            query.return_value.stdout = ""
            lock = module.reserve_gpu(9876)
            try:
                with self.assertRaises(BlockingIOError):
                    module.reserve_gpu(9876)
            finally:
                lock.close()


if __name__ == "__main__":
    unittest.main()

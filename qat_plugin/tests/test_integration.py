"""CPU structural tests. TestOps do NOT simulate or validate AIMET quantization."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import torch
from torch import nn
import yaml

from models.yolo import Model
from qat_plugin.model import prepare_detector, DecodedDetector, adjust_targets
from utils.loss import ComputeLoss

ROOT = Path(__file__).resolve().parents[2]


class Add(nn.Module):
    def forward(self, x, y):
        return x + y


class Concat(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, *inputs):
        return torch.cat(inputs, self.axis)


TEST_OPS = types.SimpleNamespace(Add=Add, Concat=Concat)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        cls.detector = Model(str(ROOT / 'models/yolov5s-cls13-2.2m-Mul8Channel025_eNPU_ReLU.yaml'), ch=1, nc=17).eval()
        cls.detector.hyp = yaml.safe_load((ROOT / 'data/hyps/hyps.full_parameters_final20260622_focaler_scale_0.7_0.7_MPDIoU.yaml').read_text())
        cls.detector.hyp['label_smoothing'] = 0.

    def test_prepared_outputs_and_decode_match_original_at_two_input_shapes(self):
        for h in (224, 256):
            x = torch.rand(1, 1, h, 320)
            raw = prepare_detector(self.detector, TEST_OPS, x)
            with torch.no_grad():
                a, _ = self.detector(x)
                b, _ = DecodedDetector(raw, self.detector)(x)
            torch.testing.assert_close(a, b, atol=2e-3, rtol=2e-4)

    def test_original_weights_and_forward_remain_unchanged(self):
        state = {k: v.clone() for k, v in self.detector.state_dict().items()}
        prepare_detector(self.detector, TEST_OPS, torch.rand(1, 1, 224, 320))
        self.assertTrue(hasattr(self.detector.model[0], 'bn'))
        for key, value in self.detector.state_dict().items():
            torch.testing.assert_close(value, state[key], atol=0, rtol=0)

    def test_fused_network_full_backward_updates_backbone_not_only_head(self):
        x = torch.rand(2, 1, 224, 320)
        raw = prepare_detector(self.detector, TEST_OPS, x[:1])
        for p in raw.parameters():
            p.requires_grad_(True)
        self.assertTrue(all(p.is_leaf for p in raw.parameters()), "BN folding must leave optimizer-ready leaf parameters")
        raw.train()
        first = raw.layers[0].conv.weight
        before = first.detach().clone()
        opt = torch.optim.SGD(raw.parameters(), lr=.001)
        loss, _ = ComputeLoss(self.detector)(list(raw(x)), torch.tensor([[0., 0., .5, .5, .3, .3]]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(first.grad)
        self.assertGreater(float(first.grad.abs().sum()), 0)
        opt.step()
        self.assertFalse(torch.equal(first, before))

    def test_target_transform_matches_original_training_and_does_not_mutate(self):
        targets = torch.tensor([[0., 0., .5, .5, .2, .3]])
        original = targets.clone()
        actual = adjust_targets(targets, 224)
        self.assertAlmostEqual(float(actual[0, 3]), 112/224, places=6)
        self.assertAlmostEqual(float(actual[0, 5]), 72/224, places=6)
        torch.testing.assert_close(original, targets)
        actual = adjust_targets(targets, 256)
        self.assertAlmostEqual(float(actual[0, 3]), (136-7)/256, places=6)

    def test_onnx_raw_head_export_and_checker(self):
        import onnx
        raw = prepare_detector(self.detector, TEST_OPS, torch.rand(1, 1, 224, 320))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'raw.onnx'
            torch.onnx.export(raw.eval(), torch.rand(1, 1, 224, 320), str(path),
                              input_names=['y_plane'], output_names=['p3', 'p4', 'p5'], opset_version=17)
            model = onnx.load(str(path))
            onnx.checker.check_model(model)
            self.assertEqual([x.name for x in model.graph.output], ['p3', 'p4', 'p5'])

    def test_evaluation_records_per_class_without_legacy_plots(self):
        import numpy as np
        import val
        spec = importlib.util.spec_from_file_location('qat_eval_integration', ROOT/'qat_plugin/run_2026-11-24.py')
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        ap_result = ([1], [0], [.8], [.6], [.685714], np.array([[.9]*9+[.2]]), [.8], [.1], [0])
        def fake_run(*args, **kwargs):
            self.assertFalse(kwargs['plots'])
            val.ap_per_class([],[],[],[0,0])
            return (.8,.6,.9,.83,0.,0.,0.), None, None
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(val,'ap_per_class', return_value=ap_result) as original:
                with mock.patch.object(val,'run',side_effect=fake_run):
                    metric = runtime.evaluate(nn.Identity(),self.detector,{},[],None,Path(folder),2,[128,160])
                self.assertIs(val.ap_per_class,original)
            self.assertAlmostEqual(metric['AP95'],.2)
            self.assertEqual(metric['per_class'][0]['instances'],2)

    def test_installer_preserves_source_skips_weights_and_refuses_existing_destination(self):
        spec = importlib.util.spec_from_file_location('qat_install', ROOT / 'qat_plugin/install_2026-11-24.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            source, dest = Path(folder)/'source', Path(folder)/'source_QAT'
            (source/'models').mkdir(parents=True)
            (source/'train.py').write_text('original training')
            (source/'models/yolo.py').write_text('original model')
            (source/'epoch76.pt').write_bytes(b'weight')
            module.install(source, dest, ROOT/'qat_plugin')
            self.assertEqual((dest/'train.py').read_text(), 'original training')
            self.assertEqual((source/'train.py').read_text(), 'original training')
            self.assertFalse((dest/'epoch76.pt').exists())
            with self.assertRaises(FileExistsError):
                module.install(source, dest, ROOT/'qat_plugin')


if __name__ == '__main__':
    unittest.main()

"""Real AIMET CPU integration; synthetic inputs do not establish detection accuracy."""
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from models.yolo import Model
from qat_plugin.model import prepare_detector
from utils.loss import ComputeLoss

ROOT = Path(__file__).resolve().parents[2]


class AimetSmokeTests(unittest.TestCase):
    def test_real_w8a8_calibration_backward_state_and_qdq_export(self):
        import aimet_torch.onnx
        from aimet_torch import QuantizationSimModel
        from aimet_torch.common.defs import QuantScheme
        from aimet_torch.common.quantsim_config.utils import get_path_for_per_channel_config
        from aimet_torch.nn.modules import custom
        import onnx
        import onnxruntime as ort
        import numpy as np
        torch.set_num_threads(2)
        torch.manual_seed(42)
        detector = Model(str(ROOT/'models/yolov5s-cls13-2.2m-Mul8Channel025_eNPU_ReLU.yaml'), ch=1, nc=17).eval()
        detector.hyp = yaml.safe_load((ROOT/'data/hyps/hyps.full_parameters_final20260622_focaler_scale_0.7_0.7_MPDIoU.yaml').read_text())
        detector.hyp['label_smoothing'] = 0.
        x = torch.rand(2, 1, 224, 320)
        raw = prepare_detector(detector, custom, x[:1])
        for p in raw.parameters():
            p.requires_grad_(True)
        sim = QuantizationSimModel(raw, dummy_input=x[:1], quant_scheme=QuantScheme.post_training_tf,
                                   default_param_bw=8, default_output_bw=8,
                                   config_file=get_path_for_per_channel_config())
        def calibrate(model):
            with torch.no_grad():
                model(x)
        sim.compute_encodings(calibrate)
        sim.model.train()
        first = next(sim.model.parameters())
        before = first.detach().clone()
        optimizer = torch.optim.SGD(sim.model.parameters(), lr=1e-3)
        loss, _ = ComputeLoss(detector)(list(sim.model(x)), torch.tensor([[0., 0., .5, .5, .2, .3]]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(first, before))
        sim.model.eval()
        with torch.no_grad():
            expected = [p.detach().numpy() for p in sim.model(x[:1])]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'gesture.onnx'
            checkpoint = Path(folder)/'qat.pt'
            torch.save(sim.model.state_dict(), checkpoint)
            sim.model.load_state_dict(torch.load(checkpoint, weights_only=False))
            aimet_torch.onnx.export(sim.model, x[:1], str(path), dynamo=False, opset_version=21,
                                    input_names=['y_plane'], output_names=['p3','p4','p5'], prequantize_constants=True)
            model = onnx.load(str(path))
            onnx.checker.check_model(model)
            self.assertTrue(any(n.op_type == 'QuantizeLinear' for n in model.graph.node))
            self.assertTrue(any(n.op_type == 'DequantizeLinear' for n in model.graph.node))
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
            actual = session.run(None, {'y_plane': x[:1].numpy()})
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, rtol=.02, atol=.05)


if __name__ == '__main__':
    unittest.main()

"""Visualization compatibility and isolation from the model being trained."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from utils import plots
from utils.loggers import log_tensorboard_graph


class VisualizationTests(unittest.TestCase):
    def test_plot_saves_both_ndarray_and_pil_results_without_rgb_swap(self):
        for use_pil in (False, True):
            class Annotator:
                def __init__(self, image, **kwargs):
                    self.im = Image.fromarray(image) if use_pil else image
                def rectangle(self, *args, **kwargs):
                    pass
                def result(self):
                    return np.asarray(self.im)
            with self.subTest(pil=use_pil), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "mosaic.png"
                images = np.zeros((1, 3, 16, 24), dtype=np.uint8)
                images[:, 0] = 200
                images[:, 1] = 10
                images[:, 2] = 30
                with mock.patch.object(plots, "Annotator", Annotator):
                    thread = plots.plot_images(images, np.empty((0, 6)), fname=path)
                    thread.join(timeout=10)
                self.assertFalse(thread.is_alive())
                self.assertTrue(path.is_file())
                with Image.open(path) as saved:
                    self.assertEqual(saved.getpixel((0, 0)), (200, 10, 30))

    def test_tensorboard_one_and_three_channel_graphs_preserve_training_state(self):
        from torch.utils.tensorboard import SummaryWriter
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        for channels in (1, 3):
            model = torch.nn.Sequential(torch.nn.Conv2d(channels, 2, 3, padding=1),
                                        torch.nn.BatchNorm2d(2), torch.nn.ReLU()).train()
            before = {key: value.clone() for key, value in model.state_dict().items()}
            with self.subTest(channels=channels), tempfile.TemporaryDirectory() as folder:
                writer = SummaryWriter(folder)
                with mock.patch("utils.loggers.LOGGER.warning") as warning:
                    log_tensorboard_graph(writer, model, imgsz=(16, 24))
                writer.close()
                warning.assert_not_called()
                event = EventAccumulator(folder)
                event.Reload()
                self.assertTrue(event.Graph().node)
            self.assertTrue(model.training)
            self.assertTrue(model[1].training)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

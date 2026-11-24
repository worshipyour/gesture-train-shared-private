"""No torch/AIMET/network required: protect the original project during deployment."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('qat_install_offline', ROOT/'qat_plugin/install_2026-11-24.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class OfflineInstallTests(unittest.TestCase):
    def fixture(self, temp):
        source = Path(temp)/'original'
        (source/'models').mkdir(parents=True)
        (source/'models/yolo.py').write_text('original model')
        (source/'train.py').write_text('original training')
        return source

    def test_copy_and_hash_leave_original_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.fixture(temp)
            target = Path(temp)/'separate'/'gesture_train_shared_QAT'
            installer.install(source, target, ROOT/'qat_plugin')
            self.assertEqual((source/'train.py').read_text(), 'original training')
            self.assertEqual((target/'train.py').read_text(), 'original training')
            report = json.loads((target/'2026-11-24_QAT_install_manifest.json').read_text())
            self.assertFalse(report['source_modified'])
            self.assertIn('models/yolo.py', report['original_source_sha256'])

    def test_weights_runs_and_images_are_not_copied(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.fixture(temp)
            (source/'epoch76.pt').write_bytes(b'weight')
            (source/'runs').mkdir()
            (source/'runs/metrics.json').write_text('{}')
            (source/'images').mkdir()
            (source/'images/a.txt').write_text('excluded directory')
            target = Path(temp)/'QAT'
            installer.install(source, target, ROOT/'qat_plugin')
            for name in ('epoch76.pt','runs','images'):
                self.assertFalse((target/name).exists())

    def test_existing_or_nested_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.fixture(temp)
            with self.assertRaises(ValueError):
                installer.install(source, source/'QAT', ROOT/'qat_plugin')
            existing = Path(temp)/'existing'
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                installer.install(source, existing, ROOT/'qat_plugin')


if __name__ == '__main__':
    unittest.main()

"""No torch or network: fixed validation subset and lossless AP capture."""
import importlib.util
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('qat_eval_offline', ROOT/'qat_plugin/run_2026-11-24.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class AdvancingLoader:
    """Like the project's InfiniteDataLoader: subsequent iterations continue the stream."""
    dataset = object()

    def __init__(self):
        self.position = 0

    def __iter__(self):
        while True:
            index = self.position
            self.position += 1
            yield {'path': str(index), 'targets': [index]}


def ap_fixture(*args, **kwargs):
    return ([1], [0], [.8], [.6], [.6857142857142857],
            [[.9,.8,.7,.6,.5,.4,.3,.2,.1,0.]], [.8], [.1], [0])


class EvaluationOfflineTests(unittest.TestCase):
    def test_smoke_reuses_same_batches_and_clones_in_place_mutations(self):
        source = AdvancingLoader()
        fixed = runtime.LimitedLoader(source, 2)
        first = list(fixed)
        first[0]['targets'][0] = -999
        second = list(fixed)
        self.assertEqual([x['path'] for x in second], ['0','1'])
        self.assertEqual(second[0]['targets'], [0])
        self.assertEqual(source.position, 2)
        self.assertEqual(len(fixed), 2)

    def test_ap_capture_preserves_original_return_and_uses_iou95_endpoint(self):
        evaluator = types.SimpleNamespace(ap_per_class=ap_fixture)
        rows = []
        expected = ap_fixture()
        with runtime.record_ap_metrics(evaluator, {0:'open_palm'}, rows):
            actual = evaluator.ap_per_class([], [], [], [0,0,0,1])
            self.assertEqual(actual, expected)
        self.assertIs(evaluator.ap_per_class, ap_fixture)
        self.assertEqual(rows[0]['instances'], 3)
        self.assertAlmostEqual(rows[0]['AP50_95'], .45)
        self.assertEqual(rows[0]['AP95'], 0.)
        self.assertEqual(rows[0]['AP50'], .9)

    def test_original_metric_function_is_restored_after_failure(self):
        evaluator = types.SimpleNamespace(ap_per_class=ap_fixture)
        with self.assertRaises(RuntimeError):
            with runtime.record_ap_metrics(evaluator, {0:'open_palm'}, []):
                raise RuntimeError('simulated evaluation failure')
        self.assertIs(evaluator.ap_per_class, ap_fixture)

    def test_delta_keeps_negative_changes_and_ap95_when_available(self):
        baseline = runtime.metrics_dict([.8,.6,.9,.5])
        candidate = runtime.metrics_dict([.81,.58,.89,.49])
        baseline['AP95'], candidate['AP95'] = .2, .18
        actual = runtime.delta(candidate, baseline)
        self.assertAlmostEqual(actual['P'], 1.)
        self.assertAlmostEqual(actual['mAP50_95'], -1.)
        self.assertAlmostEqual(actual['AP95'], -2.)


if __name__ == '__main__':
    unittest.main()

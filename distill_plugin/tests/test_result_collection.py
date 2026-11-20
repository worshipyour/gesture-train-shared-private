import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import yaml


class ResultCollectionTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "collect_results_2026-11-20.py"
        spec = importlib.util.spec_from_file_location("collector", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.original = root / "original"
        self.module.ROOT = root / "new"
        self.original.mkdir()
        self.module.ROOT.mkdir()
        cfg, data, weights = root / "student.yaml", root / "data.yaml", root / "epoch76.pt"
        cfg.write_text("depth_multiple: 0.5\nwidth_multiple: 0.25\n")
        data.write_text("names: [a, b]\ntrain: /fixed/train\nval: /fixed/val\n")
        weights.write_bytes(b"shared student initializer")
        self.options = dict(epochs=2, batch_size=128, imgsz=320, seed=0, cfg=str(cfg), data=str(data),
                            weights=str(weights), hyp={"lr0": .01}, teacher_weight=None)
        self.fgd = self.module.ROOT / "runs/fgd_run/fgd"
        self.baseline = self.original / "runs/train/control"
        self.write_run(self.fgd, teacher=True, values=(.90,.80,.95,.75))
        self.write_run(self.baseline, values=(.91,.78,.94,.70))

    def tearDown(self):
        self.temp.cleanup()

    def write_run(self, folder, teacher=False, values=(.91,.78,.94,.70), overrides=None):
        (folder / "weights").mkdir(parents=True)
        (folder / "weights/best.pt").write_bytes(b"not loaded by read-only collector")
        opts = dict(self.options, teacher_weight="/teacher/best.pt" if teacher else None)
        opts.update(overrides or {})
        (folder / "opt.yaml").write_text(yaml.safe_dump(opts))
        with (folder / "results.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["epoch","metrics/precision","metrics/recall","metrics/mAP_0.5","metrics/mAP_0.5:0.95"])
            writer.writerow([0,.95,.70,.99,.60])
            writer.writerow([1,*values])

    def test_fitness_selection_does_not_choose_highest_ap50(self):
        metrics, rows = self.module.csv_metrics(self.baseline)
        self.assertEqual(metrics["epoch"], 1)
        self.assertEqual(metrics["mAP50"], .94)

    def test_matched_control_reports_both_positive_and_negative_changes(self):
        report = self.module.collect(self.fgd.parent, self.baseline, self.original)
        self.assertEqual(report["configuration_matched_baselines"], 1)
        self.assertAlmostEqual(report["fgd_minus_baseline_pp"]["P"], -1)
        self.assertAlmostEqual(report["fgd_minus_baseline_pp"]["mAP50_95"], 5)
        self.assertTrue(report["read_only"])

    def test_seed_mismatch_blocks_gain_claim(self):
        opts = yaml.safe_load((self.baseline / "opt.yaml").read_text())
        opts["seed"] = 43
        (self.baseline / "opt.yaml").write_text(yaml.safe_dump(opts))
        report = self.module.collect(self.fgd.parent, self.baseline, self.original)
        self.assertIsNone(report["fgd_minus_baseline_pp"])
        self.assertIn("seed", report["baseline_candidates"][0]["configuration_differences_or_unknowns"])

    def test_multiple_matching_controls_are_not_selected_by_score(self):
        self.write_run(self.original / "runs/train/control_2", values=(.95,.95,.99,.9))
        report = self.module.collect(self.fgd.parent, original=self.original)
        self.assertEqual(report["configuration_matched_baselines"], 2)
        self.assertIsNone(report["fgd_minus_baseline_pp"])
        self.assertIn("Multiple", report["next_action"])

    def test_saved_exact_summary_metrics_take_precedence_over_rounded_csv(self):
        metrics = dict(epoch=1,P=.90123,R=.80123,F1_from_mean_PR=.8483,mAP50=.95123,mAP50_95=.75123)
        (self.fgd.parent / "summary.json").write_text(json.dumps({"metrics":{"fgd":metrics}}))
        report = self.module.collect(self.fgd.parent, self.baseline, self.original)
        self.assertEqual(report["fgd"]["metrics"]["mAP50_95"], .75123)
        self.assertAlmostEqual(report["fgd_minus_baseline_pp"]["mAP50_95"], 5.123)


if __name__ == "__main__":
    unittest.main()

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from v3 import split_protocol as protocol


def fixture():
    return {split: [
        {"image": f"{split}/real.jpg", "group": split, "split": split,
         "type": "real", "twin": None, "source_id": f"source:{split}"},
        {"image": f"{split}/fake.jpg", "group": split, "split": split,
         "type": "edit", "twin": f"{split}/real.jpg", "source_id": f"source:{split}"},
    ] for split in protocol.FILES}


def write_manifests(root, splits):
    for split, filename in protocol.FILES.items():
        (root / filename).write_text("".join(json.dumps(row) + "\n" for row in splits[split]))


class SplitTests(unittest.TestCase):
    def test_valid_and_missing_identity_coverage(self):
        report = protocol.validate_splits(fixture())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["identity_coverage_rows"]["test"]["patient_id"], 0)

    def test_duplicate_image_and_lexical_alias(self):
        for name in ("train/real.jpg", "train/../train/real.jpg"):
            with self.subTest(name=name):
                data = fixture()
                data["test"][0]["image"] = name
                with self.assertRaisesRegex(ValueError, "Duplicate image"):
                    protocol.validate_splits(data)

    def test_duplicate_within_partition(self):
        data = fixture()
        data["train"].append(copy.deepcopy(data["train"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate image"):
            protocol.validate_splits(data)

    def test_all_identity_overlap_types(self):
        for field in protocol.IDENTITIES:
            with self.subTest(field=field):
                data = fixture()
                data["train"][0][field] = data["test"][0][field] = "same"
                with self.assertRaisesRegex(ValueError, "Identity overlap"):
                    protocol.validate_splits(data)

    def test_source_cannot_have_two_groups_within_train(self):
        data = fixture()
        data["train"][1]["group"] = "different"
        with self.assertRaisesRegex(ValueError, "Identity overlap"):
            protocol.validate_splits(data)

    def test_wrong_declared_split(self):
        data = fixture()
        data["test"][0]["split"] = "select"
        with self.assertRaisesRegex(ValueError, "split declaration"):
            protocol.validate_splits(data)

    def test_invalid_twins(self):
        for twin in ("train/real.jpg", "select/fake.jpg", "missing.jpg"):
            with self.subTest(twin=twin):
                data = fixture()
                data["select"][1]["twin"] = twin
                with self.assertRaisesRegex(ValueError, "[Tt]win"):
                    protocol.validate_splits(data)

    def test_absent_twin_is_reported(self):
        data = fixture()
        data["test"][1]["twin"] = None
        self.assertEqual(protocol.validate_splits(data)["fake_rows_without_twin"], 1)

    def test_empty_partition_and_missing_group(self):
        data = fixture()
        data["select"] = []
        with self.assertRaisesRegex(ValueError, "Empty"):
            protocol.validate_splits(data)
        data = fixture()
        del data["train"][0]["group"]
        with self.assertRaisesRegex(ValueError, "Missing"):
            protocol.validate_splits(data)

    def test_freeze_verify_then_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_manifests(root, fixture())
            lock = root / "lock.json"
            protocol.write_new_json(lock, protocol.audit_directory(root))
            protocol.verify_lock(root, lock)
            with self.assertRaises(FileExistsError):
                protocol.write_new_json(lock, {})
            # Even a harmless newline changes the frozen bytes.
            with (root / "test_core.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "lock mismatch"):
                protocol.verify_lock(root, lock)

    def test_cli_leak_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture()
            data["test"][0]["group"] = "train"
            write_manifests(Path(directory), data)
            run = subprocess.run([sys.executable, "-m", "v3.split_protocol", "--mfclean", directory],
                                 capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("Identity overlap", run.stderr)

    def test_sft_selftest_only_propagates_failure(self):
        from v3 import build_sft_v3 as builder
        with tempfile.TemporaryDirectory() as directory:
            argv = ["builder", "--mfclean", directory, "--selftest_only", "--report", directory + "/report.md"]
            with patch.object(sys, "argv", argv), patch.object(builder.os, "nice"), \
                    patch.object(builder, "selftest", return_value=[("leak", "FAIL", "")]):
                with self.assertRaisesRegex(ValueError, "Data self-test failed"):
                    builder.main()

    def test_sft_rejects_leak_before_pixel_processing(self):
        from v3 import build_sft_v3 as builder
        with tempfile.TemporaryDirectory() as directory:
            data = fixture()
            data["test"][0]["group"] = "train"
            write_manifests(Path(directory), data)
            argv = ["builder", "--mfclean", directory, "--report", directory + "/report.md"]
            with patch.object(sys, "argv", argv), patch.object(builder.os, "nice"):
                with self.assertRaisesRegex(ValueError, "Identity overlap"):
                    builder.main()


class SelectionTests(unittest.TestCase):
    def test_select_tie_and_reject_test_and_altered_truth(self):
        from v3.select_checkpoint import select
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture()
            write_manifests(root, data)
            lock = root / "lock.json"
            protocol.write_new_json(lock, protocol.audit_directory(root))
            (root / "weights").write_bytes(b"synthetic weights")
            (root / "config").write_text("{}")
            predictions = [dict(row, output="<verdict>" + ("real" if row["type"] == "real" else "deepfake")
                                + "</verdict>") for row in data["select"]]
            def save_predictions(rows):
                (root / "pred.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            save_predictions(predictions)
            plan = root / "candidates.json"
            plan.write_text(json.dumps([{"id": name, "predictions": "pred.jsonl", "artifact": "weights",
                                         "config": "config"} for name in ("b", "a")]))
            result = select(root, lock, plan)
            self.assertEqual(result["selected"], "a")
            self.assertEqual(result["candidates"][0]["balanced_accuracy"], 1)
            for bad in (data["test"], [dict(predictions[0], type="edit"), predictions[1]], predictions[:1]):
                with self.subTest(bad=bad):
                    save_predictions(bad)
                    with self.assertRaises(ValueError):
                        select(root, lock, plan)


class ResumeTests(unittest.TestCase):
    def test_resume_preserves_scaled_box(self):
        from PIL import Image
        import numpy as np
        from v3 import build_mfclean as builder
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (32, 32), "gray").save(root / "real.png")
            Image.new("RGB", (32, 32), "white").save(root / "fake.png")
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[5:10, 5:10] = 255
            Image.fromarray(mask).save(root / "mask.png")
            row = {"rel": "fake.png", "src_rel": "real.png", "mask_rel": "mask.png", "organ": "brain-mri"}
            with patch.object(builder, "MF", directory + "/"):
                first = builder.work(("fake", row, directory + "/out"))
                second = builder.work(("fake", row, directory + "/out"))
            self.assertTrue(first["ok"], first)
            self.assertEqual(first, second)
            self.assertEqual(first["box"], [320, 320, 576, 576])


if __name__ == "__main__":
    unittest.main()

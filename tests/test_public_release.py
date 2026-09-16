"""Data-free release integrity checks; no model weights or clinical rows required."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
import download_weights as weights
import evaluate_release as evaluate
import prepare_data as prepare


def sha(data):
    return hashlib.sha256(data).hexdigest()


class WeightArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.destination = self.root / 'repo'
        self.destination.mkdir()
        self.path = self.root / 'weights.zip'
        self.files = {'runs/demo/final.pt': b'synthetic weight fixture',
                      'runs/demo/run.json': b'{"synthetic": true}'}

    def archive(self, extra=(), checkpoint='runs/demo/final.pt', symlink=False):
        files = {checkpoint: self.files['runs/demo/final.pt'],
                 'runs/demo/run.json': self.files['runs/demo/run.json']}
        with warnings.catch_warnings(), zipfile.ZipFile(self.path, 'w') as z:
            warnings.simplefilter('ignore', UserWarning)
            for name, content in list(files.items()) + list(extra):
                if symlink and name == checkpoint:
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    z.writestr(info, content)
                else:
                    z.writestr(name, content)
        return {'sha256': weights.digest(self.path), 'models': [{
            'checkpoint_path': checkpoint, 'checkpoint_sha256': sha(files[checkpoint]),
            'run_path': 'runs/demo/run.json', 'run_sha256': sha(files['runs/demo/run.json']),
            'config_path': 'configs/demo.json', 'config_sha256': sha(b'{}')}]}

    def test_install_is_exact_and_idempotent(self):
        archive = self.archive()
        weights.install_archive(self.path, archive, self.destination)
        weights.install_archive(self.path, archive, self.destination)
        for name, content in self.files.items():
            self.assertEqual((self.destination / name).read_bytes(), content)

    def test_wrong_archive_digest_rejected_before_writes(self):
        archive = self.archive()
        archive['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Archive checksum'):
            weights.install_archive(self.path, archive, self.destination)
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_unlisted_or_duplicate_members_rejected(self):
        for extra in [[('data/raw/image.nii', b'not allowed')],
                      [('runs/demo/final.pt', b'duplicate')]]:
            with self.subTest(extra=extra):
                archive = self.archive(extra)
                with self.assertRaisesRegex(ValueError, 'unexpected or duplicated'):
                    weights.install_archive(self.path, archive, self.destination)
                self.assertEqual(list(self.destination.iterdir()), [])

    def test_member_digest_failure_does_not_install_corrupt_file(self):
        archive = self.archive()
        archive['models'][0]['checkpoint_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Extracted file checksum'):
            weights.install_archive(self.path, archive, self.destination)
        self.assertFalse((self.destination / 'runs/demo/final.pt').exists())
        self.assertEqual(list(self.destination.rglob('*.part')), [])

    def test_conflicting_local_file_is_preserved(self):
        archive = self.archive()
        target = self.destination / 'runs/demo/final.pt'
        target.parent.mkdir(parents=True)
        target.write_bytes(b'local model')
        with self.assertRaises(FileExistsError):
            weights.install_archive(self.path, archive, self.destination)
        self.assertEqual(target.read_bytes(), b'local model')

    def test_traversal_and_symlink_members_rejected(self):
        for checkpoint, symlink in [('../escape.pt', False),
                                    ('runs/demo/final.pt', True)]:
            with self.subTest(checkpoint=checkpoint, symlink=symlink):
                archive = self.archive(checkpoint=checkpoint, symlink=symlink)
                with self.assertRaisesRegex(ValueError, 'Unsafe archive member'):
                    weights.install_archive(self.path, archive, self.destination)
        self.assertFalse((self.root / 'escape.pt').exists())

    def test_modified_config_is_rejected(self):
        archive = self.archive()
        weights.install_archive(self.path, archive, self.destination)
        config = self.destination / 'configs/demo.json'
        config.parent.mkdir()
        config.write_bytes(b'{}')
        weights.verify_archive(archive, self.destination)
        config.write_bytes(b'{"changed": true}')
        with self.assertRaisesRegex(ValueError, 'Configuration was changed'):
            weights.verify_archive(archive, self.destination)


class PublishedSplitTests(unittest.TestCase):
    def test_id_only_split_is_complete_and_pinned(self):
        path = ROOT / 'data/splits/patients.csv'
        identity = json.loads((ROOT / 'data/splits/identity.json').read_text())
        self.assertEqual(weights.digest(path), identity['id_split_sha256'])
        with path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            self.assertEqual(set(reader.fieldnames), {'patient_id', 'split', 'development_split'})
            rows = list(reader)
        self.assertEqual(len(rows), 235)
        self.assertEqual(len({r['patient_id'] for r in rows}), 235)
        self.assertEqual(sum(r['split'] == 'train' for r in rows), 188)
        self.assertEqual(sum(r['split'] == 'test' for r in rows), 47)
        training = [r for r in rows if r['split'] == 'train']
        self.assertEqual(sum(r['development_split'] == 'train' for r in training), 150)
        self.assertEqual(sum(r['development_split'] == 'validation' for r in training), 38)

    def test_synthetic_partition_or_source_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            split = root / 'data/splits/patients.csv'
            split.parent.mkdir(parents=True)
            content = b'patient_id,split,development_split\nSYNTHETIC_1,train,train\n'
            split.write_bytes(content)
            actual = root / 'synthetic.csv'
            actual.write_bytes(content)
            identity = {'id_split_sha256': sha(content), 'clinical_patient_csv_sha256': sha(content)}
            (split.parent / 'identity.json').write_text(json.dumps(identity))
            self.assertEqual(len(prepare.verify(actual, root)), 1)
            actual.write_bytes(content.replace(b',train,train', b',test,test'))
            with self.assertRaisesRegex(ValueError, 'partition differs'):
                prepare.verify(actual, root)
            actual.write_bytes(content + b'\n')
            with self.assertRaisesRegex(ValueError, 'Clinical metadata differs'):
                prepare.verify(actual, root)
            split.write_bytes(content + b'\n')
            with self.assertRaisesRegex(ValueError, 'ID split was modified'):
                prepare.verify(actual, root)

    def test_manifest_has_six_unique_model_keys_and_expected_configs(self):
        manifest = json.loads((ROOT / 'release-manifest.json').read_text())
        models = [m for a in manifest['archives'] for m in a['models']]
        self.assertEqual(len(models), 6)
        self.assertEqual({m['model'] for m in models}, {
            'csfinet', 'unet', 'image', 'image_age', 'image_resection', 'image_age_resection'})
        for model in models:
            self.assertEqual(weights.digest(ROOT / model['config_path']), model['config_sha256'])


class EvaluationCliTests(unittest.TestCase):
    def test_invalid_partial_runs_fail_before_accessing_data(self):
        invalid = [('--limit', '0'), ('--limit', '47'), ('--limit', '1', '--stage', 'all'),
                   ('--limit', '1', '--stage', 'shap'), ('--slice-batch', '0'),
                   ('--device', 'unsupported')]
        for args in invalid:
            with self.subTest(args=args), patch.object(sys, 'argv', ['evaluate_release.py', *args]), \
                    patch.object(evaluate, 'verify') as verify, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    evaluate.main()
                self.assertEqual(caught.exception.code, 2)
                verify.assert_not_called()


if __name__ == '__main__':
    unittest.main()

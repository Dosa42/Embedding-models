import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location('fetch_models', Path(__file__).resolve().parents[1] / 'scripts/fetch_xenova_models.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class Downloads(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.body = b'expected bytes'
        self.entry = dict(repo='Xenova/e5-base-v2', revision='a' * 40, path='config.json', size=len(self.body), sha256=hashlib.sha256(self.body).hexdigest())

    def test_cached_valid_file_needs_no_network(self):
        path = self.root / self.entry['repo'] / self.entry['path']
        path.parent.mkdir(parents=True)
        path.write_bytes(self.body)
        with patch.object(m, 'urlopen', side_effect=AssertionError('Network used')):
            m.fetch(self.entry, self.root)

    def test_corrupt_download_never_replaces_existing_file(self):
        path = self.root / self.entry['repo'] / self.entry['path']
        path.parent.mkdir(parents=True)
        path.write_bytes(b'existing')
        with patch.object(m, 'urlopen', side_effect=lambda *a, **kw: io.BytesIO(b'wrong payload!')), patch.object(m.time, 'sleep'):
            with self.assertRaises(ValueError):
                m.fetch(self.entry, self.root)
        self.assertEqual(path.read_bytes(), b'existing')
        self.assertFalse(path.with_name('config.json.partial').exists())

    def test_successful_download_checks_pinned_url(self):
        with patch.object(m, 'urlopen', return_value=io.BytesIO(self.body)) as request:
            m.fetch(self.entry, self.root)
        self.assertIn('/resolve/' + 'a' * 40 + '/config.json', request.call_args.args[0].full_url)
        self.assertTrue(m.valid(self.root / self.entry['repo'] / self.entry['path'], self.entry))

    def test_verify_only_fails_on_same_size_corruption(self):
        path = self.root / self.entry['repo'] / self.entry['path']
        path.parent.mkdir(parents=True)
        path.write_bytes(b'x' * len(self.body))
        with patch.object(m, 'urlopen', side_effect=AssertionError('Network used')):
            with self.assertRaises(ValueError):
                m.fetch(self.entry, self.root, verify_only=True)

    def test_archive_contains_only_selected_precision_and_checksums(self):
        entries = []
        for name in ('config.json', 'tokenizer.json', 'onnx/model_quantized.onnx', 'onnx/model_fp16.onnx'):
            e = dict(self.entry, path=name)
            p = self.root / e['repo'] / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(self.body)
            entries.append(e)
        output = self.root / 'dist'
        output.mkdir()
        archive = m.package('e5-base-v2', 'q8', entries, self.root, output)
        with zipfile.ZipFile(archive) as z:
            self.assertIn('Xenova/e5-base-v2/onnx/model_quantized.onnx', z.namelist())
            self.assertNotIn('Xenova/e5-base-v2/onnx/model_fp16.onnx', z.namelist())
            for line in z.read('SHA256SUMS.txt').decode().splitlines():
                digest, name = line.split('  ', 1)
                self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(), digest)
            self.assertEqual(len(json.loads(z.read('DOWNLOAD_MANIFEST.json'))), 3)

    def test_lock_has_complete_pinned_snapshots_and_rejects_traversal(self):
        lock = m.load_lock(m.ROOT / 'xenova-models.lock.json')
        self.assertEqual(len({e['repo'] for e in lock['files']}), 3)
        lock['files'][0]['path'] = '../escape'
        path = self.root / 'lock.json'
        path.write_text(json.dumps(lock))
        with self.assertRaises(ValueError):
            m.load_lock(path)


if __name__ == '__main__':
    unittest.main()

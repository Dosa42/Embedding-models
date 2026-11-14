import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

SPEC = importlib.util.spec_from_file_location('repo_models', Path(__file__).resolve().parents[1] / 'scripts/repo_model_packages.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class RepositoryPackages(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.name = 'Xenova-e5-base-v2-q8-standalone.zip'
        self.body = b'{"local":"model"}'
        self.entry = {'repo': 'Xenova/e5-base-v2', 'path': 'config.json', 'size': len(self.body), 'sha256': hashlib.sha256(self.body).hexdigest()}
        archive = self.root / self.name
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('Xenova/e5-base-v2/config.json', self.body)
            z.writestr('DOWNLOAD_MANIFEST.json', json.dumps([self.entry]))
        self.expected = m.digest(archive)
        self.parts = self.root / 'parts'
        self.folder = self.parts / self.name.removesuffix('.zip')

    def stage(self):
        m.stage(self.root, self.parts, self.name, self.expected, chunk_bytes=100)

    def test_full_roundtrip_and_extraction(self):
        self.stage()
        manifest = json.loads((self.folder / 'manifest.json').read_text())
        self.assertGreater(len(manifest['parts']), 1)
        self.assertTrue(all(p['size'] <= 100 for p in manifest['parts']))
        output = m.restore(self.folder, self.root / 'restored')
        self.assertEqual(output.read_bytes(), (self.root / self.name).read_bytes())
        m.extract(output, self.root / 'models')
        self.assertEqual((self.root / 'models/Xenova/e5-base-v2/config.json').read_bytes(), self.body)
        self.stage()  # Re-importing a verified package is safe and idempotent.

    def test_corruption_is_rejected_before_reconstruction(self):
        self.stage()
        part = next(self.folder.glob('*.part001'))
        part.write_bytes(b'x' * part.stat().st_size)
        with self.assertRaises(ValueError):
            m.restore(self.folder, self.root / 'restored')
        self.assertFalse((self.root / 'restored' / self.name).exists())

    def test_missing_part_is_rejected(self):
        self.stage()
        next(self.folder.glob('*.part002')).unlink()
        with self.assertRaises(ValueError):
            m.restore(self.folder, self.root / 'restored')

    def test_incorrect_release_checksum_is_rejected(self):
        with self.assertRaises(ValueError):
            m.stage(self.root, self.parts, self.name, '0' * 64)
        self.assertFalse(self.folder.exists())

    def test_changed_local_model_is_not_overwritten(self):
        destination = self.root / 'models'
        target = destination / 'Xenova/e5-base-v2/config.json'
        target.parent.mkdir(parents=True)
        target.write_bytes(b'user modification')
        with self.assertRaises(ValueError):
            m.extract(self.root / self.name, destination)
        self.assertEqual(target.read_bytes(), b'user modification')

    def test_manifest_cannot_reference_parent_path(self):
        self.stage()
        path = self.folder / 'manifest.json'
        data = json.loads(path.read_text())
        data['parts'][0]['name'] = '../escape'
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            m.restore(self.folder, self.root / 'restored')


if __name__ == '__main__':
    unittest.main()

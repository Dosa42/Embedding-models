import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

SPEC = importlib.util.spec_from_file_location('verify_repo_models', Path(__file__).resolve().parents[1] / 'scripts/verify_repo_models.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class CompleteModels(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'model.zip'
        with zipfile.ZipFile(self.path, 'w') as archive:
            archive.writestr('config.json', '{}')
        self.asset = {'name': self.path.name, 'size': self.path.stat().st_size, 'sha256': hashlib.sha256(self.path.read_bytes()).hexdigest()}

    def test_complete_zip_passes(self):
        m.verify_archive(self.path, self.asset)

    def test_same_size_corrupt_bytes_fail(self):
        self.path.write_bytes(b'x' * self.asset['size'])
        with self.assertRaises(ValueError):
            m.verify_archive(self.path, self.asset)

    def test_pointer_must_match_the_complete_archive(self):
        pointer = 'version https://git-lfs.github.com/spec/v1\noid sha256:' + self.asset['sha256'] + '\nsize ' + str(self.asset['size']) + '\n'
        m.verify_pointer(pointer, self.asset)
        with self.assertRaises(ValueError):
            m.verify_pointer(pointer.replace(self.asset['sha256'], '0' * 64), self.asset)


if __name__ == '__main__':
    unittest.main()

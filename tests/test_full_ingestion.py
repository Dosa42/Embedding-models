import base64
import json
from unittest.mock import patch
import numpy as np
from test_ingestion import FakeEmbedder
import tempfile
import unittest
import zipfile
from pathlib import Path
from ingestion.core import extract_zip
from ingestion.full import allocation, read_all


class FullIngestionTests(unittest.TestCase):
    def test_hidden_unknown_and_html_content_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'input.zip'
            samples = {'.hidden/.settings': 'secret setting', '.git/config': '[core]',
                       'src/code.mts': 'export const value = 42;',
                       'page.html': '<script>important source()</script><style>.a{}</style>',
                       'script.S': 'MOV X0, X1'}
            with zipfile.ZipFile(archive, 'w') as z:
                for name, text in samples.items():
                    z.writestr(name, text)
            paths = extract_zip(archive, root / 'out')
            self.assertEqual(len(paths), len(samples))
            for p in paths:
                self.assertEqual(''.join(x[2] for x in read_all(p)), samples[p.relative_to(root / 'out').as_posix()])

    def test_binary_byte_representation_is_lossless(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'image.png'
            raw = bytes(range(256)) * 6000
            p.write_bytes(raw)
            segments = list(read_all(p))
            self.assertTrue(all(x[0].startswith('binary_base64:') for x in segments))
            self.assertEqual(base64.b64decode(''.join(x[2] for x in segments)), raw)

    def test_empty_and_utf16_and_large_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / '.file'
            p.write_bytes(b'')
            self.assertEqual(list(read_all(p))[0][0], 'empty_or_whitespace')
            p.write_bytes('hello unicode'.encode('utf-16'))
            self.assertEqual(list(read_all(p))[0][2], 'hello unicode')
            text = 'abc' * 800000
            p.write_text(text)
            self.assertEqual(''.join(x[2] for x in read_all(p)), text)

    def test_full_shards_merge_and_reopen(self):
        from ingestion.full import shard, merge
        from ingestion.build import verify_persisted
        class FullFake(FakeEmbedder):
            def __init__(self, *args):
                super().__init__(*args)
                self.dimension = 1024
                self.provenance.update(model='Xenova/bge-m3', dtype='fp16', dimension=1024)
            def encode(self, texts):
                return np.tile(np.array([1 / 32] * 1024, dtype='<f4'), (len(texts), 1))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.zip'
            samples = {'.hidden/config.mts': 'word1 word2', 'image.png': bytes(range(256)), 'empty': ''}
            with zipfile.ZipFile(source, 'w') as z:
                for name, content in samples.items():
                    z.writestr(name, content)
            with patch('ingestion.build.Embedder', FullFake), patch('ingestion.build.verify_model', return_value='test'):
                for i in range(2):
                    shard(source, root, root / 'shards' / f'shard-{i}', i, 2)
            merge(source, root / 'shards', root / 'final', 2)
            verify_persisted(root / 'final')
            report = json.loads((root / 'final/performance-quality.json').read_text())
            self.assertEqual(report['files_embedded'], 3)
            self.assertEqual(report['excluded_files'], 0)
            self.assertEqual(report['hidden_files_embedded'], 1)
            self.assertEqual(report['representations_files']['binary_base64'], 1)
            self.assertEqual((root / 'final/source-original.zip').read_bytes(), source.read_bytes())
            (root / 'shards/shard-0/manifest.json').unlink()
            with self.assertRaises(FileNotFoundError):
                merge(source, root / 'shards', root / 'incomplete', 2)

    def test_shards_have_exactly_once_coverage(self):
        entries = [(f'.hidden/file{i}', i * 7) for i in range(100)]
        groups = allocation(entries, 16)
        flat = [p for g in groups for p in g]
        self.assertEqual(len(flat), len(entries))
        self.assertEqual(set(flat), {p for p, _ in entries})
        self.assertEqual(groups, allocation(list(reversed(entries)), 16))

if __name__ == '__main__':
    unittest.main()

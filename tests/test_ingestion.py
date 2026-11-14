import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from ingestion.core import chunk_text, extract_zip, read_segments
from ingestion.prepare import select_input
from ingestion.build import ingest_worker, verify_persisted, package_database, chroma_client


def tokenizer():
    vocab = {'[UNK]': 0, '[CLS]': 1, '[SEP]': 2, 'passage': 3, ':': 4}
    vocab.update({f'word{i}': i + 5 for i in range(100)})
    t = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.post_processor = processors.TemplateProcessing(single='[CLS] $A [SEP]', special_tokens=[('[CLS]', 1), ('[SEP]', 2)])
    return t


class FakeEmbedder:
    def __init__(self, *args):
        self.chunk_tokenizer = tokenizer()
        self.max_sequence, self.dimension = 512, 4
        self.provenance = {'model': 'Xenova/e5-base-v2', 'dtype': 'q8', 'dimension': 4, 'pooling': 'mean', 'normalize': True, 'weights_sha256': 'test', 'tokenizer_sha256': 'test'}

    def prefix(self, query=False):
        return 'query: ' if query else 'passage: '

    def encode(self, texts):
        return np.tile(np.array([0.5] * 4, dtype='<f4'), (len(texts), 1))


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_zip(self, entries):
        path = self.root / 'input.zip'
        with zipfile.ZipFile(path, 'w') as z:
            for name, content in entries.items():
                z.writestr(name, content)
        return path

    def test_chunk_overlap_offsets_and_no_truncation(self):
        text = ' '.join(f'word{i}' for i in range(50))
        chunks = list(chunk_text(text, tokenizer(), 16, 4, 'passage: ', 24))
        self.assertGreater(len(chunks), 2)
        for c in chunks:
            self.assertEqual(c.text, text[c.start:c.end])
            self.assertLessEqual(len(tokenizer().encode('passage: ' + c.text).ids), 24)
        self.assertEqual(chunks[0].text.split()[-4:], chunks[1].text.split()[:4])
        self.assertEqual(chunks[-1].end, len(text))

    def test_invalid_overlap_and_context_fail(self):
        for params in [(16, 16, 512), (20, 2, 20)]:
            with self.assertRaises(ValueError):
                list(chunk_text('word1', tokenizer(), params[0], params[1], 'passage: ', params[2]))

    def test_zip_traversal_symlinks_and_limits_fail(self):
        with self.assertRaises(ValueError):
            extract_zip(self.make_zip({'../escape.txt': 'bad'}), self.root / 'extract')
        path = self.root / 'link.zip'
        with zipfile.ZipFile(path, 'w') as z:
            info = zipfile.ZipInfo('link')
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, '/etc/passwd')
        with self.assertRaises(ValueError):
            extract_zip(path, self.root / 'links')
        with self.assertRaises(ValueError):
            extract_zip(self.make_zip({'big.txt': 'x' * 20}), self.root / 'big', max_file=10)

    def test_input_selection_is_explicit_when_ambiguous(self):
        self.make_zip({'a.txt': 'a'})
        self.assertEqual(select_input(self.root).name, 'input.zip')
        (self.root / 'second.zip').write_bytes(b'fake')
        with self.assertRaises(ValueError):
            select_input(self.root)
        with self.assertRaises(ValueError):
            select_input(self.root, '../input.zip')

    def test_html_and_docx_extraction(self):
        html = self.root / 'a.html'
        html.write_text('<p>Readable &amp; text</p><script>hidden()</script>')
        self.assertIn('Readable & text', list(read_segments(html))[0][2])
        self.assertNotIn('hidden', list(read_segments(html))[0][2])
        from docx import Document
        doc = Document()
        doc.add_paragraph('Document paragraph')
        doc.add_table(rows=1, cols=1).cell(0, 0).text = 'Table content'
        doc.save(self.root / 'a.docx')
        text = '\n'.join(s[2] for s in read_segments(self.root / 'a.docx'))
        self.assertIn('Document paragraph', text)
        self.assertIn('Table content', text)

    def test_database_contains_vectors_sources_and_skipped_report(self):
        source = self.make_zip({'notes.txt': ' '.join(f'word{i}' for i in range(50)), 'image.png': b'not text'})
        output = self.root / 'result'
        with patch('ingestion.build.Embedder', FakeEmbedder), patch('ingestion.build.verify_model', return_value='test-revision'):
            result = ingest_worker(source, self.root, 'e5-base-v2', 'q8', output, 16, 4, 2)
        self.assertGreater(result['metadata']['chunk_count'], 2)
        self.assertEqual(result['metadata']['skipped_documents'], 1)
        verify_persisted(output)
        client = chroma_client(output / 'chroma_db')
        try:
            collection = client.get_collection('chunks', embedding_function=None)
            self.assertEqual(collection.count(), result['metadata']['chunk_count'])
            stored = collection.get(include=['embeddings', 'metadatas'])
            self.assertEqual(stored['embeddings'].shape[1], 4)
            self.assertTrue(np.allclose(np.linalg.norm(stored['embeddings'], axis=1), 1))
            self.assertTrue(all(row['source'] == 'notes.txt' for row in stored['metadatas']))
            self.assertGreater(len(collection.get(where_document={'$contains': 'word1'})['ids']), 0)
        finally:
            client.close()
        package_database(output, 'e5-base-v2', 'q8')
        archive = output / 'chunked-chromadb-e5-base-v2-q8.zip'
        with zipfile.ZipFile(archive) as z:
            self.assertIsNone(z.testzip())
            self.assertTrue(any(name.startswith('chroma_db/') for name in z.namelist()))
            self.assertIn('ingestion/search.py', z.namelist())

    def test_hidden_files_and_directories_are_indexed(self):
        names = ['.notes.md', '.gitignore', '.env', '.git/config', '.github/workflows/test.yml',
                 '__MACOSX/notes.txt', 'node_modules/package/readme.md', 'LICENSE']
        source = self.make_zip({**{name: 'word1 word2' for name in names}, '.git/objects/binary': b'\x00\xff'})
        output = self.root / 'hidden-result'
        with patch('ingestion.build.Embedder', FakeEmbedder), patch('ingestion.build.verify_model', return_value='test'):
            result = ingest_worker(source, self.root, 'e5-base-v2', 'q8', output, 16, 4)
        self.assertEqual({d['path'] for d in result['documents'] if d['status'] == 'indexed'}, set(names))
        self.assertEqual(result['metadata']['skipped_documents'], 1)
        client = chroma_client(output / 'chroma_db')
        try:
            stored = client.get_collection('chunks', embedding_function=None).get(include=['metadatas'])
            self.assertEqual({m['source'] for m in stored['metadatas']}, set(names))
        finally:
            client.close()

    def test_prepare_selects_one_model_and_verifies_its_members(self):
        from ingestion.prepare import prepare
        target = self.root / 'standalone-models' / 'ingestion target'
        target.mkdir(parents=True)
        with zipfile.ZipFile(target / 'source.zip', 'w') as z:
            z.writestr('notes.txt', 'Text')
        content = b'{"model_type":"bert"}'
        asset_name = 'Xenova-e5-base-v2-q8-standalone.zip'
        archive = self.root / 'standalone-models' / asset_name
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('Xenova/e5-base-v2/config.json', content)
        package = {'name': asset_name, 'size': archive.stat().st_size, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
        (self.root / 'repo-model-release.lock.json').write_text(json.dumps({'assets': [package]}))
        (self.root / 'xenova-models.lock.json').write_text(json.dumps({'files': [{'repo': 'Xenova/e5-base-v2', 'path': 'config.json', 'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}]}))
        with patch('ingestion.prepare.ROOT', self.root), patch('ingestion.prepare.TARGET', target), patch('ingestion.prepare.subprocess.run') as command:
            result = prepare('e5-base-v2', 'q8', '', self.root / 'work')
        self.assertEqual((Path(result['model_dir']) / 'config.json').read_bytes(), content)
        self.assertIn(asset_name, command.call_args.args[0][3])
        self.assertIn('source.zip', command.call_args.args[0][3])

    def test_empty_corpus_does_not_publish_database(self):
        source = self.make_zip({'empty.txt': '   '})
        output = self.root / 'empty-result'
        with patch('ingestion.build.Embedder', FakeEmbedder), patch('ingestion.build.verify_model', return_value='test'):
            with self.assertRaises(ValueError):
                ingest_worker(source, self.root, 'e5-base-v2', 'q8', output, 16, 4)
        self.assertFalse(list(output.glob("*.zip")))


if __name__ == '__main__':
    unittest.main()

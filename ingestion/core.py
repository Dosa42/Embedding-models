"""Text extraction, tokenizer-aware chunking and local ONNX embeddings."""
from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

import numpy as np
from tokenizers import Tokenizer

MODELS = {'bge-m3': (1024, 8192, 'cls'), 'e5-base-v2': (768, 512, 'mean'), 'e5-large-v2': (1024, 512, 'mean')}
WEIGHTS = {'q8': 'model_quantized.onnx', 'fp16': 'model_fp16.onnx'}
TEXT_EXTENSIONS = set('.txt .md .rst .csv .tsv .json .jsonl .yaml .yml .xml .log .py .js .ts .jsx .tsx .java .kt .kts .c .h .cpp .hpp .cs .go .rs .sql .sh .toml .ini .cfg .css .scss .vue .svelte .ipynb'.split())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def extract_zip(source, destination, max_files=20000, max_total=2 * 1024**3, max_file=100 * 1024**2):
    """Extract only regular files, rejecting traversal, links and excessive sizes."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths, seen, total = [], set(), 0
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        if len(entries) > max_files:
            raise ValueError(f'ZIP has more than {max_files} entries')
        for info in entries:
            name = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if name.is_absolute() or '..' in name.parts or '\\' in info.filename or ':' in info.filename or '\x00' in info.filename:
                raise ValueError('Unsafe ZIP path: ' + info.filename)
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError('ZIP links and special files are not supported')
            if info.flag_bits & 1:
                raise ValueError('Password-protected ZIPs are not supported')
            if info.is_dir():
                continue
            if str(name) in seen:
                raise ValueError('Duplicate ZIP path: ' + str(name))
            seen.add(str(name))
            if info.file_size > max_file:
                raise ValueError('ZIP member exceeds 100 MiB: ' + str(name))
            total += info.file_size
            if total > max_total:
                raise ValueError('ZIP exceeds the uncompressed size limit')
            path = destination / name
            if not path.resolve().is_relative_to(destination):
                raise ValueError('ZIP path leaves extraction directory')
            path.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with archive.open(info) as src, path.open('wb') as out:
                for block in iter(lambda: src.read(1024 * 1024), b''):
                    written += len(block)
                    if written > info.file_size or written > max_file:
                        raise ValueError('ZIP member exceeds its declared size')
                    out.write(block)
            if written != info.file_size:
                raise ValueError('Truncated ZIP member')
            paths.append(path)
    return sorted(paths)


class PlainHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1
        elif tag in ('p', 'div', 'br', 'li', 'h1', 'h2', 'h3', 'tr'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def read_segments(path):
    """Yield (segment label, page number or None, extracted text). No OCR."""
    suffix = path.suffix.lower()
    if suffix == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError('Encrypted PDF is not supported')
        for i, page in enumerate(reader.pages, 1):
            yield f'page:{i}', i, page.extract_text() or ''
    elif suffix == '.docx':
        from docx import Document
        # Bound the internal ZIP before asking the XML parser to read it.
        with zipfile.ZipFile(path) as z:
            if sum(i.file_size for i in z.infolist()) > 100 * 1024**2:
                raise ValueError('DOCX expanded content exceeds 100 MiB')
        doc = Document(path)
        for i, paragraph in enumerate(doc.paragraphs, 1):
            yield f'paragraph:{i}', None, paragraph.text
        for i, table in enumerate(doc.tables, 1):
            yield f'table:{i}', None, '\n'.join('\t'.join(c.text for c in row.cells) for row in table.rows)
    elif suffix in TEXT_EXTENSIONS or suffix in ('', '.html', '.htm'):
        raw = path.read_bytes()
        encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
        text = raw.decode(encoding)
        if '\x00' in text:
            raise ValueError('Binary content in text file')
        if suffix in ('.html', '.htm'):
            parser = PlainHTML()
            parser.feed(text)
            text = ''.join(parser.parts)
        yield 'text', None, text
    else:
        raise ValueError('Unsupported file type: ' + (suffix or '(none)'))


@dataclass
class Chunk:
    text: str
    start: int
    end: int
    tokens: int


def chunk_text(text, tokenizer, chunk_tokens=384, overlap=64, prefix='', max_sequence=512):
    if chunk_tokens < 8 or overlap < 0 or overlap >= chunk_tokens:
        raise ValueError('Require chunk_tokens >= 8 and 0 <= overlap < chunk_tokens')
    reserve = len(tokenizer.encode(prefix, add_special_tokens=True).ids)
    if chunk_tokens + reserve > max_sequence:
        raise ValueError(f'chunk_tokens exceeds model budget; maximum is {max_sequence - reserve}')
    if len(text) > 5_000_000:
        raise ValueError('Text segment exceeds 5 million characters')
    encoding = tokenizer.encode(text, add_special_tokens=False)
    offsets = encoding.offsets
    start = 0
    while start < len(offsets):
        end = min(start + chunk_tokens, len(offsets))
        while end > start:
            first, last = offsets[start][0], offsets[end - 1][1]
            content = text[first:last]
            # Re-tokenizing a subword boundary may change the token count.
            if len(tokenizer.encode(prefix + content, add_special_tokens=True).ids) <= max_sequence:
                break
            end -= 1
        if end == start:
            raise ValueError('A text fragment cannot fit within the model context')
        if content.strip():
            yield Chunk(content, first, last, len(tokenizer.encode(content, add_special_tokens=False).ids))
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap)


class Embedder:
    def __init__(self, model_dir, model, dtype='q8', threads=2):
        import onnxruntime as ort
        self.model, self.dtype = model, dtype
        self.dimension, self.max_sequence, self.pooling = MODELS[model]
        self.model_dir = Path(model_dir)
        self.chunk_tokenizer = Tokenizer.from_file(str(self.model_dir / 'tokenizer.json'))
        self.chunk_tokenizer.no_padding()
        self.chunk_tokenizer.no_truncation()
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / 'tokenizer.json'))
        self.tokenizer.no_truncation()
        config = json.loads((self.model_dir / 'config.json').read_text())
        pad_id = config.get('pad_token_id', 0)
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token=self.tokenizer.id_to_token(pad_id) or '[PAD]')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_dir / 'onnx' / WEIGHTS[dtype]), sess_options=options, providers=['CPUExecutionProvider'])
        self.inputs = {v.name for v in self.session.get_inputs()}
        self.provenance = {'model': 'Xenova/' + model, 'dtype': dtype, 'dimension': self.dimension, 'pooling': self.pooling, 'normalize': True,
                           'weights_sha256': sha256(self.model_dir / 'onnx' / WEIGHTS[dtype]), 'tokenizer_sha256': sha256(self.model_dir / 'tokenizer.json')}

    def prefix(self, query=False):
        return '' if self.model == 'bge-m3' else ('query: ' if query else 'passage: ')

    def encode(self, texts, query=False):
        encodings = self.tokenizer.encode_batch([self.prefix(query) + text for text in texts])
        if any(sum(e.attention_mask) > self.max_sequence for e in encodings):
            raise ValueError('Input exceeds model context; refusing silent truncation')
        feeds = {'input_ids': np.asarray([e.ids for e in encodings], dtype=np.int64),
                 'attention_mask': np.asarray([e.attention_mask for e in encodings], dtype=np.int64),
                 'token_type_ids': np.asarray([e.type_ids for e in encodings], dtype=np.int64)}
        outputs = self.session.run(['last_hidden_state'], {key: value for key, value in feeds.items() if key in self.inputs})
        hidden = outputs[0].astype(np.float32)
        if self.pooling == 'cls':
            vectors = hidden[:, 0, :]
        else:
            mask = feeds['attention_mask'][..., None].astype(np.float32)
            vectors = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if vectors.shape[1] != self.dimension or not np.isfinite(vectors).all() or (norms <= 0).any():
            raise ValueError('Invalid embedding output')
        return (vectors / norms).astype('<f4')

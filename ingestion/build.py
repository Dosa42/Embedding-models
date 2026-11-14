"""Build and export a native persistent ChromaDB from a ZIP of documents."""
import argparse
from bisect import bisect_right
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
from .core import Embedder, MODELS, WEIGHTS, chunk_text, extract_zip, read_segments, sha256

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = 'chunks'


def chroma_client(path):
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def verify_model(embedder):
    lock = json.loads((ROOT / 'xenova-models.lock.json').read_text())
    entries = [e for e in lock['files'] if e['repo'] == embedder.provenance['model']]
    for filename, key in [('onnx/' + WEIGHTS[embedder.dtype], 'weights_sha256'), ('tokenizer.json', 'tokenizer_sha256')]:
        expected = next(e for e in entries if e['path'] == filename)
        if embedder.provenance[key] != expected['sha256']:
            raise ValueError('Model/tokenizer does not match the repository lock')
    return entries[0]['revision']


def ingest_worker(source_zip, model_dir, model, dtype, output, chunk_tokens=384, overlap=64, batch_size=4, max_chunks=100000, threads=2):
    """Run in a dedicated process; exiting closes all Chroma handles before export."""
    if model not in MODELS or dtype not in WEIGHTS:
        raise ValueError('Unknown model or precision')
    if not 1 <= batch_size <= 16 or not 1 <= threads <= 8 or not 1 <= max_chunks <= 100000:
        raise ValueError('batch_size=1..16, threads=1..8 and max_chunks=1..100000 required')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    embedder = Embedder(model_dir, model, dtype, threads)
    revision = verify_model(embedder)
    list(chunk_text('', embedder.chunk_tokenizer, chunk_tokens, overlap, embedder.prefix(), embedder.max_sequence))
    metadata = {'schema_version': 1, 'database_format': 'chromadb-persistent', 'collection': COLLECTION, **embedder.provenance, 'model_revision': revision,
                'input_zip': Path(source_zip).name, 'input_sha256': sha256(source_zip), 'chunk_tokens': chunk_tokens, 'overlap_tokens': overlap,
                'document_prefix': embedder.prefix(), 'query_prefix': embedder.prefix(query=True), 'distance_metric': 'cosine',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'repository_commit': os.environ.get('GITHUB_SHA'),
                'runtime': {name: version(name) for name in ('chromadb', 'onnxruntime', 'tokenizers', 'numpy', 'pypdf', 'python-docx')}}
    client = chroma_client(output / 'chroma_db')
    try:
        collection = client.create_collection(name=COLLECTION, embedding_function=None,
                                              configuration={'hnsw': {'space': 'cosine'}},
                                              metadata={'model': metadata['model'], 'dtype': dtype, 'dimension': embedder.dimension, 'provenance_json': json.dumps(metadata)})
        documents, total_chunks = [], 0
        with tempfile.TemporaryDirectory(prefix='ingestion-input-') as temporary:
            work = Path(temporary)
            paths = extract_zip(source_zip, work)
            with (output / 'chunks.jsonl').open('w', encoding='utf-8') as jsonl:
                for document_id, path in enumerate(paths, 1):
                    relative = path.relative_to(work).as_posix()
                    record = {'id': document_id, 'path': relative, 'sha256': sha256(path), 'file_type': path.suffix.lower(), 'status': 'indexed', 'detail': None, 'chunk_count': 0}
                    try:
                        if '__MACOSX' in path.parts or path.name.startswith('.'):
                            raise ValueError('Archive metadata/hidden file skipped')
                        segments = list(read_segments(path))
                        if any(len(text) > 5_000_000 for _, _, text in segments) or sum(len(text) for _, _, text in segments) > 20_000_000:
                            raise ValueError('Document exceeds extracted text limit')
                    except Exception as error:
                        record.update(status='skipped', detail=f'{type(error).__name__}: {error}')
                        segments = []
                    pending = []
    
                    def flush():
                        nonlocal total_chunks
                        if not pending:
                            return
                        vectors = embedder.encode([row['text'] for row in pending])
                        ids, metadatas = [], []
                        for row in pending:
                            identity = f"{relative}\0{record['sha256']}\0{row['chunk_index']}"
                            ids.append(hashlib.sha256(identity.encode('utf-8')).hexdigest())
                            meta = {key: value for key, value in row.items() if key != 'text' and value is not None}
                            meta.update(source=relative, source_sha256=record['sha256'], document_id=document_id)
                            metadatas.append(meta)
                        collection.add(ids=ids, embeddings=vectors.tolist(), documents=[row['text'] for row in pending], metadatas=metadatas)
                        for row, identity in zip(pending, ids):
                            row.update(id=identity, source=relative, source_sha256=record['sha256'])
                            jsonl.write(json.dumps(row, ensure_ascii=False) + '\n')
                        total_chunks += len(pending)
                        pending.clear()
    
                    for segment, page, text in segments:
                        newlines = [i for i, character in enumerate(text) if character == '\n']
                        for chunk in chunk_text(text, embedder.chunk_tokenizer, chunk_tokens, overlap, embedder.prefix(), embedder.max_sequence):
                            if total_chunks + len(pending) >= max_chunks:
                                raise ValueError(f'Corpus exceeds the {max_chunks} chunk limit; split the input ZIP')
                            pending.append({'chunk_index': record['chunk_count'], 'segment': segment, 'page': page, 'char_start': chunk.start, 'char_end': chunk.end,
                                            'line_start': bisect_right(newlines, chunk.start - 1) + 1, 'line_end': bisect_right(newlines, chunk.end - 1) + 1,
                                            'token_count': chunk.tokens, 'text': chunk.text})
                            record['chunk_count'] += 1
                            if len(pending) == batch_size:
                                flush()
                    flush()
                    if record['status'] == 'indexed' and record['chunk_count'] == 0:
                        record.update(status='no_text', detail='No extractable text; scanned PDFs need OCR before ingestion')
                    documents.append(record)
                    print(f"{record['status']}: {relative} ({record['chunk_count']} chunks)", flush=True)
        if total_chunks == 0:
            raise ValueError('No chunks were produced; no empty database will be published')
        if collection.count() != total_chunks:
            raise ValueError('Chroma record count mismatch')
        metadata.update(document_count=len(documents), indexed_documents=sum(d['status'] == 'indexed' for d in documents), chunk_count=total_chunks,
                        skipped_documents=sum(d['status'] != 'indexed' for d in documents))
        collection.modify(metadata={'model': metadata['model'], 'dtype': dtype, 'dimension': embedder.dimension, 'provenance_json': json.dumps(metadata)})
        manifest = {'metadata': metadata, 'documents': documents, 'chunks_jsonl_sha256': sha256(output / 'chunks.jsonl')}
        (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return manifest
    finally:
        client.close()


def verify_persisted(output):
    """Open the exported store in a fresh process and check every vector."""
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    metadata = manifest['metadata']
    client = chroma_client(output / 'chroma_db')
    try:
        collection = client.get_collection(COLLECTION, embedding_function=None)
        if collection.count() != metadata['chunk_count']:
            raise ValueError('Persisted Chroma count mismatch')
        checked = 0
        for offset in range(0, collection.count(), 256):
            result = collection.get(limit=256, offset=offset, include=['embeddings', 'documents', 'metadatas'])
            vectors = np.asarray(result['embeddings'], dtype=np.float32)
            if vectors.shape != (len(result['ids']), metadata['dimension']) or not np.isfinite(vectors).all():
                raise ValueError('Persisted embedding shape/value mismatch')
            if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
                raise ValueError('Persisted embeddings are not normalized')
            if not all(text for text in result['documents']) or not all('source' in m for m in result['metadatas']):
                raise ValueError('Persisted source metadata/text missing')
            checked += len(result['ids'])
            if offset == 0:
                hits = collection.query(query_embeddings=[vectors[0].tolist()], n_results=1, include=['distances'])
                if not hits['ids'][0] or abs(hits['distances'][0][0]) > 1e-4:
                    raise ValueError('Persisted Chroma index cannot retrieve its own vector')
        if checked != metadata['chunk_count']:
            raise ValueError('Persisted record traversal mismatch')
        print('CHROMA_REOPEN_VERIFIED', checked, flush=True)
    finally:
        client.close()


def package_database(output, model, dtype):
    """Called only after both Chroma worker processes have exited."""
    output = Path(output)
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['database_files'] = [{'path': p.relative_to(output).as_posix(), 'size': p.stat().st_size, 'sha256': sha256(p)} for p in sorted((output / 'chroma_db').rglob('*')) if p.is_file()]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'README.txt').write_text('''Native ChromaDB embedding database

Keep the complete chroma_db/ directory together. It contains Chroma's metadata
and vector index files; do not rename its internal files or keep only one .db file.
Collection: chunks. Embeddings: normalized float32, cosine distance.
q8 describes the ONNX model precision; Chroma stores the resulting vectors.

Open directly (install the bundled pinned requirements with Python 3.12):
  import chromadb
  client = chromadb.PersistentClient(path="chroma_db")
  collection = client.get_collection("chunks", embedding_function=None)
  print(collection.count())

To query, supply query_embeddings produced by the SAME model/tokenizer/precision.
The bundled command verifies that identity and supplies the correct query prefix:
  python -m ingestion.search --database chroma_db --model-dir /path/to/Xenova/MODEL --query "your question"

chunks.jsonl contains readable chunks and source metadata. manifest.json records
model/source checksums, settings, versions, skipped files and all database files.
Offsets/lines refer to extracted text; PDF page numbers are one-based. No OCR.
The export was reopened and its vectors/index checked before it was packaged.
Use the pinned Chroma version to open this snapshot; upgrades may migrate it.
''', encoding='utf-8')
    package = output / 'ingestion'
    package.mkdir()
    for filename in ('__init__.py', 'core.py', 'search.py'):
        shutil.copy2(ROOT / 'ingestion' / filename, package / filename)
    shutil.copy2(ROOT / 'ingestion/requirements.txt', output / 'requirements.txt')
    archive = output / f'chunked-chromadb-{model}-{dtype}.zip'
    files = sorted(p for p in output.rglob('*') if p.is_file() and p != archive)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path in files:
            z.write(path, path.relative_to(output).as_posix())
    if archive.stat().st_size >= 2 * 1024**3:
        raise ValueError('Output exceeds the GitHub release asset limit; split the input')
    (output / 'SHA256SUMS.txt').write_text(f'{sha256(archive)}  {archive.name}\n')
    m = manifest['metadata']
    (output / 'RELEASE.md').write_text(f"# ChromaDB — {model} ({dtype})\n\n- Indexed documents: {m['indexed_documents']}\n- Chunks: {m['chunk_count']}\n- Vector dimensions: {m['dimension']}\n- Skipped/empty files: {m['skipped_documents']}\n- Input SHA-256: `{m['input_sha256']}`\n\nThe ZIP contains a complete native ChromaDB persistent store, JSONL chunks, source references, provenance and a search utility. Model weights are not duplicated. Keep the full `chroma_db/` directory together.\n")
    print('DATABASE_READY', archive, flush=True)
    return manifest


def build_database(source_zip, model_dir, model, dtype, output, chunk_tokens=384, overlap=64, batch_size=4, max_chunks=100000, threads=2):
    source_zip = Path(source_zip).resolve()
    model_dir = Path(model_dir).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    command = [sys.executable, '-m', 'ingestion.build', '--worker', '--input', str(source_zip), '--model-dir', str(model_dir), '--model', model, '--dtype', dtype,
               '--output', str(output), '--chunk-tokens', str(chunk_tokens), '--overlap', str(overlap), '--batch-size', str(batch_size), '--max-chunks', str(max_chunks), '--threads', str(threads)]
    try:
        subprocess.run(command, cwd=ROOT, check=True)
        subprocess.run([sys.executable, '-m', 'ingestion.build', '--verify-store', '--output', str(output)], cwd=ROOT, check=True)
        return package_database(output, model, dtype)
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--model', choices=list(MODELS))
    parser.add_argument('--dtype', choices=list(WEIGHTS), default='q8')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-tokens', type=int, default=384)
    parser.add_argument('--overlap', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-chunks', type=int, default=100000)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--verify-store', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify_store:
        verify_persisted(args.output)
    else:
        if args.input is None or args.model_dir is None or args.model is None:
            parser.error('--input, --model-dir and --model are required')
        function = ingest_worker if args.worker else build_database
        function(args.input, args.model_dir, args.model, args.dtype, args.output, args.chunk_tokens, args.overlap, args.batch_size, args.max_chunks, args.threads)


if __name__ == '__main__':
    main()

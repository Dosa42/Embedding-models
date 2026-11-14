"""Exhaustive ZIP ingestion: every regular member, including hidden/unknown files.

UTF text is embedded verbatim (including HTML scripts). Non-text bytes are
losslessly represented as base64, NOT claimed to have image/binary semantics.
The original ZIP is retained. Shards use deterministic size-balanced allocation.
"""
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from .build import COLLECTION, chroma_client, ingest_worker, package_database
from .core import sha256


def read_all(path):
    raw = path.read_bytes()
    encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    try:
        text = raw.decode(encoding)
        if '\0' in text:
            raise UnicodeError('NUL in text')
        kind = 'verbatim_text'
    except UnicodeError:
        text = base64.b64encode(raw).decode('ascii')
        kind = 'binary_base64'
    if not text.strip():
        yield 'empty_or_whitespace', None, f'Empty or whitespace-only file: {path.name}. SHA256: {sha256(path)}'
        return
    # Keep bounded segments without dropping characters. Chunk offsets are local
    # to these explicitly labelled segments, not global byte offsets.
    for offset in range(0, len(text), 1_000_000):
        yield f'{kind}:char:{offset}', None, text[offset:offset + 1_000_000]


def allocation(entries, count):
    groups, loads = [[] for _ in range(count)], [0] * count
    for name, size in sorted(entries, key=lambda x: (-x[1], x[0])):
        index = min(range(count), key=lambda i: (loads[i], i))
        groups[index].append(name)
        loads[index] += max(size, 1)
    return [sorted(group) for group in groups]


def inventory(source):
    with zipfile.ZipFile(source) as archive:
        return [(i.filename, i.file_size) for i in archive.infolist() if not i.is_dir()]


def shard(source, model_dir, output, index, count):
    started = time.perf_counter()
    groups = allocation(inventory(source), count)
    wanted = set(groups[index])
    if not wanted:
        raise ValueError('Empty shard; reduce shard count')
    def select(paths, root):
        return [p for p in paths if p.relative_to(root).as_posix() in wanted]
    manifest = ingest_worker(source, model_dir, 'bge-m3', 'fp16', output,
                             threads=4, read_file=read_all, select_paths=select)
    actual = {d['path'] for d in manifest['documents']}
    if actual != wanted or manifest['metadata']['skipped_documents']:
        raise ValueError('Incomplete shard: missing, skipped or empty files')
    subprocess.run([sys.executable, '-m', 'ingestion.build', '--verify-store', '--output', str(output)], check=True)
    manifest['metadata'].update(full_ingestion=True, shard_index=index, shard_count=count,
                                shard_wall_seconds=time.perf_counter() - started,
                                hidden_documents=sum(any(part.startswith('.') for part in Path(p).parts) for p in actual))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('FULL_SHARD_VERIFIED', index, len(actual), flush=True)


def merge(source, shards, output, count):
    started = time.perf_counter()
    roots = sorted(shards.glob('shard-*'))
    if len(roots) != count:
        raise ValueError('Missing shard artifacts')
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in roots]
    if {m['metadata']['shard_index'] for m in manifests} != set(range(count)):
        raise ValueError('Duplicate/missing shard indexes')
    digest = sha256(source)
    reference = manifests[0]['metadata']
    for m in manifests:
        for key in ('model', 'dtype', 'weights_sha256', 'tokenizer_sha256', 'repository_commit', 'chunk_tokens', 'overlap_tokens'):
            if m['metadata'][key] != reference[key]:
                raise ValueError('Incompatible shard: ' + key)
        if m['metadata']['input_sha256'] != digest or m['metadata']['shard_count'] != count:
            raise ValueError('Shard source/count mismatch')
    expected = dict(inventory(source))
    documents = sorted([d for m in manifests for d in m['documents']], key=lambda d: d['path'])
    if len(documents) != len(expected) or {d['path'] for d in documents} != set(expected):
        raise ValueError('Full source coverage verification failed')
    if any(d['status'] != 'indexed' or d['chunk_count'] < 1 for d in documents):
        raise ValueError('A source file was not embedded')
    with zipfile.ZipFile(source) as z:
        import hashlib
        for d in documents:
            if hashlib.sha256(z.read(d['path'])).hexdigest() != d['sha256']:
                raise ValueError('Source content hash mismatch')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(documents, 1):
        d['id'] = i
    ids = {d['path']: d['id'] for d in documents}
    metadata = dict(reference)
    metadata.pop('shard_index')
    metadata.pop('shard_wall_seconds', None)
    metadata.update(elapsed_seconds=sum(m['metadata']['elapsed_seconds'] for m in manifests),
                    inference_seconds=sum(m['metadata']['inference_seconds'] for m in manifests),
                    embedded_tokens=sum(m['metadata']['embedded_tokens'] for m in manifests))
    metadata.update(document_count=len(documents), indexed_documents=len(documents), skipped_documents=0,
                    chunk_count=sum(m['metadata']['chunk_count'] for m in manifests),
                    hidden_documents=sum(any(p.startswith('.') for p in Path(d['path']).parts) for d in documents))
    client = chroma_client(output / 'chroma_db')
    representations = {}
    try:
        target = client.create_collection(COLLECTION, embedding_function=None,
                  configuration={'hnsw': {'space': 'cosine'}}, metadata={'model': metadata['model'], 'dtype': 'fp16', 'dimension': 1024})
        with (output / 'chunks.jsonl').open('w') as chunks:
            for root, manifest in zip(roots, manifests):
                if sha256(root / 'chunks.jsonl') != manifest['chunks_jsonl_sha256']:
                    raise ValueError('Shard JSONL hash mismatch')
                source_client = chroma_client(root / 'chroma_db')
                try:
                    collection = source_client.get_collection(COLLECTION, embedding_function=None)
                    if collection.count() != manifest['metadata']['chunk_count']:
                        raise ValueError('Shard vector count mismatch')
                    for offset in range(0, collection.count(), 256):
                        rows = collection.get(offset=offset, limit=256, include=['embeddings', 'documents', 'metadatas'])
                        for meta in rows['metadatas']:
                            meta['document_id'] = ids[meta['source']]
                        target.add(ids=rows['ids'], embeddings=rows['embeddings'], documents=rows['documents'], metadatas=rows['metadatas'])
                finally:
                    source_client.close()
                with (root / 'chunks.jsonl').open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        representations.setdefault(row['segment'].split(':')[0], set()).add(row['source'])
                        chunks.write(line)
        if target.count() != metadata['chunk_count']:
            raise ValueError('Merged vector count mismatch')
        target.modify(metadata={'model': metadata['model'], 'dtype': 'fp16', 'dimension': 1024, 'provenance_json': json.dumps(metadata)})
    finally:
        client.close()
    manifest = {'metadata': metadata, 'documents': documents, 'chunks_jsonl_sha256': sha256(output / 'chunks.jsonl')}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    subprocess.run([sys.executable, '-m', 'ingestion.build', '--verify-store', '--output', str(output)], check=True)
    inference = sum(m['metadata']['inference_seconds'] for m in manifests)
    worker = sum(m['metadata']['elapsed_seconds'] for m in manifests)
    tokens = sum(m['metadata']['embedded_tokens'] for m in manifests)
    report = {'status': 'complete', 'model': 'Xenova/bge-m3', 'dtype': 'fp16', 'provider': 'CPUExecutionProvider',
        'source_sha256': digest, 'files_expected': len(expected), 'files_embedded': len(documents),
        'hidden_files_embedded': metadata['hidden_documents'], 'excluded_files': 0, 'skipped_files': 0,
        'uncompressed_bytes': sum(expected.values()), 'chunks': metadata['chunk_count'], 'tokens_with_overlap': tokens,
        'shard_count': count, 'threads_per_shard': 4, 'worker_seconds_sum': worker, 'inference_seconds_sum': inference,
        'slowest_shard_seconds': max(m['metadata']['shard_wall_seconds'] for m in manifests),
        'merge_and_verify_seconds': time.perf_counter() - started,
        'chunks_per_inference_second_per_worker_aggregate': metadata['chunk_count'] / inference,
        'tokens_per_inference_second_per_worker_aggregate': tokens / inference,
        'representations_files': {k: len(v) for k, v in representations.items()},
        'quality': {'source_hashes_verified': True, 'complete_file_coverage': True, 'all_vectors_checked_after_reopen': True,
          'finite_1024_dimensions_unit_norm': True, 'self_retrieval_smoke_test': True,
          'semantic_retrieval_benchmark': 'not measured: no labelled relevance dataset',
          'binary_caveat': 'Base64 embeds byte representation; no image understanding or OCR is claimed.'},
        'shards': [m['metadata'] for m in manifests], 'finished_utc': datetime.now(timezone.utc).isoformat()}
    (output / 'performance-quality.json').write_text(json.dumps(report, indent=2) + '\n')
    shutil.copyfile(source, output / 'source-original.zip')
    package_database(output, 'bge-m3', 'fp16')
    summary = (f"\nFull ingestion: {len(documents)}/{len(expected)} files; {metadata['hidden_documents']} hidden; zero exclusions/skips.\n"
               f"Chunks: {metadata['chunk_count']}; embedded tokens including overlap: {tokens}.\n"
               f"Summed inference time: {inference:.2f}s; summed worker time: {worker:.2f}s.\n"
               f"Representations: {report['representations_files']}.\n"
               "All persisted vectors checked. Semantic retrieval quality has not been benchmarked. Binary base64 is not image understanding.\n"
               "See performance-quality.json, workflow-timing.json and per-shard logs for timings and checks.\n")
    with (output / 'RELEASE.md').open('a') as f:
        f.write(summary)
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['shard', 'merge'])
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-dir', type=Path)
    p.add_argument('--shards', type=Path)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--count', type=int, default=16)
    a = p.parse_args()
    if not 0 <= a.index < a.count:
        p.error('Require 0 <= index < count')
    if a.mode == 'shard':
        shard(a.input, a.model_dir, a.output, a.index, a.count)
    else:
        merge(a.input, a.shards, a.output, a.count)

if __name__ == '__main__':
    main()

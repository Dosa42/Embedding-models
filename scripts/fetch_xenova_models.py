#!/usr/bin/env python3
"""Download pinned Xenova ONNX models and build standalone ZIPs (Python 3.10+)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('bge-m3', 'e5-base-v2', 'e5-large-v2')
VARIANTS = {'q8': 'model_quantized.onnx', 'fp16': 'model_fp16.onnx'}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def valid(path, entry):
    return path.is_file() and path.stat().st_size == entry['size'] and sha256(path) == entry['sha256']


def load_lock(path):
    lock = json.loads(path.read_text())
    if lock['schema_version'] != 1:
        raise ValueError('Unsupported lock schema')
    seen = set()
    for entry in lock['files']:
        repo, name = entry['repo'], PurePosixPath(entry['path'])
        if repo not in ['Xenova/' + model for model in MODELS]:
            raise ValueError('Unknown repository')
        if name.is_absolute() or '..' in name.parts or '\\' in entry['path']:
            raise ValueError('Unsafe relative path')
        if not re.fullmatch(r'[0-9a-f]{40}', entry['revision']) or not re.fullmatch(r'[0-9a-f]{64}', entry['sha256']):
            raise ValueError('Invalid pinned revision or checksum')
        if not isinstance(entry['size'], int) or entry['size'] <= 0:
            raise ValueError('Invalid file size')
        key = (repo, entry['path'])
        if key in seen:
            raise ValueError('Duplicate file')
        seen.add(key)
    for model in MODELS:
        entries = [e for e in lock['files'] if e['repo'] == 'Xenova/' + model]
        required = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'}
        required.update('onnx/' + name for name in VARIANTS.values())
        if not required.issubset({e['path'] for e in entries}) or len({e['revision'] for e in entries}) != 1:
            raise ValueError('Incomplete model snapshot')
    return lock


def fetch(entry, destination, verify_only=False):
    path = destination / entry['repo'] / entry['path']
    if valid(path, entry):
        print('Verified:', entry['repo'], entry['path'], flush=True)
        return
    if verify_only:
        raise ValueError('Missing or corrupt file: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    url = f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/{entry['path']}?download=true"
    for attempt in range(3):
        try:
            request = Request(url, headers={'User-Agent': 'Embedding-models-pinned-downloader/1.0'})
            with urlopen(request, timeout=120) as response, temporary.open('wb') as out:
                for block in iter(lambda: response.read(4 * 1024 * 1024), b''):
                    out.write(block)
            if not valid(temporary, entry):
                raise ValueError('Size or SHA-256 mismatch: ' + entry['path'])
            temporary.replace(path)
            print('Downloaded and verified:', entry['repo'], entry['path'], flush=True)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def package(model, dtype, files, destination, output):
    repo = 'Xenova/' + model
    selected = [e for e in files if e['repo'] == repo and (not e['path'].startswith('onnx/') or e['path'] == 'onnx/' + VARIANTS[dtype])]
    archive = output / f'Xenova-{model}-{dtype}-standalone.zip'
    temporary = archive.with_suffix('.zip.partial')
    pooling = 'cls' if model == 'bge-m3' else 'mean'
    notes = f'''{repo} / {dtype}

Extract this archive to a models directory. It includes original ONNX weights,
tokenizer, config and upstream model card. A compatible inference runtime must
be installed separately. These are dense-embedding models.

ONNX: {repo}/onnx/{VARIANTS[dtype]}
Pinned revision: {selected[0]['revision']}
Source: https://huggingface.co/{repo}

Transformers.js 3.x local configuration:
  env.localModelPath = '/absolute/path/to/models/';
  env.allowRemoteModels = false;
  const embed = await pipeline('feature-extraction', '{repo}', {{ dtype: '{dtype}' }});
  const output = await embed(texts, {{ pooling: '{pooling}', normalize: true }});

Import env and pipeline from @huggingface/transformers first. q8 selects
model_quantized.onnx; fp16 selects model_fp16.onnx and requires a compatible
execution provider. Offline browser use also requires local runtime/WASM assets.
E5 retrieval inputs need 'query: ' and 'passage: ' prefixes, mean pooling with
attention masking, and L2 normalization. E5-v2 is English-only, max 512 tokens.
BGE-M3 dense embeddings use CLS pooling and L2 normalization.

Files were verified against the committed size/SHA-256 manifest. The workflow
checks archive integrity; it does not run inference or benchmark model quality.
SHA256SUMS.txt uses paths relative to this archive's root.
'''
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for entry in selected:
            path = destination / repo / entry['path']
            z.write(path, repo + '/' + entry['path'])
        z.writestr('README_LOCAL.txt', notes)
        z.writestr('DOWNLOAD_MANIFEST.json', json.dumps(selected, indent=2) + '\n')
        z.writestr('SHA256SUMS.txt', ''.join(f"{e['sha256']}  {repo}/{e['path']}\n" for e in selected))
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:
            raise ValueError('Archive integrity failure')
    temporary.replace(archive)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['all', *MODELS], default='all')
    parser.add_argument('--dtype', choices=['all', *VARIANTS], default='q8')
    parser.add_argument('--destination', type=Path, default=ROOT / 'models')
    parser.add_argument('--output', type=Path, default=ROOT / 'dist')
    parser.add_argument('--verify-only', action='store_true', help='Verify existing model files without network access')
    parser.add_argument('--no-package', action='store_true')
    args = parser.parse_args()
    lock = load_lock(ROOT / 'xenova-models.lock.json')
    models = MODELS if args.model == 'all' else [args.model]
    dtypes = list(VARIANTS) if args.dtype == 'all' else [args.dtype]
    weights = {'onnx/' + VARIANTS[d] for d in dtypes}
    files = [e for e in lock['files'] if e['repo'].split('/')[1] in models and (not e['path'].startswith('onnx/') or e['path'] in weights)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda entry: fetch(entry, args.destination, args.verify_only), files))
    if args.no_package:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    archives = [package(model, dtype, files, args.destination, args.output) for model in models for dtype in dtypes]
    (args.output / 'SHA256SUMS.txt').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in archives))
    (args.output / 'xenova-models.lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    (args.output / 'RELEASE.md').write_text('## Standalone Xenova ONNX models\n\n' + '\n'.join(f'- `{p.name}` ({p.stat().st_size:,} bytes)' for p in archives) + '\n\nEach ZIP contains the model, tokenizer, configuration, upstream model card, and checksums. Download one precision per model. q8 is the default preset. FP16 requires a compatible runtime. No split files or Git LFS are needed.\n\nAll upstream files are pinned and SHA-256 verified. Inference is not tested by this workflow.\n')
    print('Created', len(archives), 'standalone packages.', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Store release ZIPs as ordinary Git files; restore them offline after cloning."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 48 * 1024 * 1024
MODELS = ('bge-m3', 'e5-base-v2', 'e5-large-v2')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_name(value):
    if not value or value in ('.', '..') or Path(value).name != value or '/' in value or '\\' in value:
        raise ValueError('Unsafe filename: ' + value)
    return value


def verify_parts(folder, manifest):
    if manifest['schema_version'] != 1:
        raise ValueError('Unsupported package manifest')
    safe_name(manifest['archive'])
    total = 0
    full = hashlib.sha256()
    seen = set()
    for part in manifest['parts']:
        name = safe_name(part['name'])
        if name in seen:
            raise ValueError('Duplicate part')
        seen.add(name)
        p = folder / name
        if not p.is_file() or p.stat().st_size != part['size']:
            raise ValueError('Missing or truncated part: ' + str(p))
        h = hashlib.sha256()
        with p.open('rb') as src:
            for block in iter(lambda: src.read(4 * 1024 * 1024), b''):
                h.update(block)
                full.update(block)
                total += len(block)
        if h.hexdigest() != part['sha256']:
            raise ValueError('Part checksum mismatch: ' + str(p))
    if not seen or total != manifest['size'] or full.hexdigest() != manifest['sha256']:
        raise ValueError('Complete archive checksum mismatch')


def stage(source, output, asset, expected_sha, chunk_bytes=CHUNK_BYTES):
    safe_name(asset)
    archive = source / asset
    if not archive.is_file() or digest(archive) != expected_sha:
        raise ValueError('Release archive checksum mismatch: ' + asset)
    folder = output / asset.removesuffix('.zip')
    if folder.exists():
        manifest = json.loads((folder / 'manifest.json').read_text())
        if manifest['archive'] != asset or manifest['sha256'] != expected_sha:
            raise ValueError('Refusing to replace a different existing package')
        verify_parts(folder, manifest)
        print('Already stored and verified:', asset, flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.staging-', dir=output) as temporary:
        work = Path(temporary)
        parts = []
        with archive.open('rb') as src:
            while block := src.read(chunk_bytes):
                name = f'{asset}.part{len(parts) + 1:03d}'
                (work / name).write_bytes(block)
                parts.append({'name': name, 'size': len(block), 'sha256': hashlib.sha256(block).hexdigest()})
        manifest = {'schema_version': 1, 'archive': asset, 'size': archive.stat().st_size, 'sha256': expected_sha, 'parts': parts}
        (work / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        verify_parts(work, manifest)
        shutil.move(str(work), str(folder))
    print('Stored and verified:', asset, len(parts), 'Git parts', flush=True)


def restore(folder, output):
    manifest = json.loads((folder / 'manifest.json').read_text())
    verify_parts(folder, manifest)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / manifest['archive']
    if archive.exists():
        if digest(archive) != manifest['sha256']:
            raise ValueError('Existing ZIP differs; move it aside first: ' + str(archive))
        return archive
    temporary = archive.with_suffix('.zip.partial')
    try:
        with temporary.open('wb') as out:
            for part in manifest['parts']:
                with (folder / part['name']).open('rb') as src:
                    shutil.copyfileobj(src, out, 4 * 1024 * 1024)
        if digest(temporary) != manifest['sha256']:
            raise ValueError('Reassembled archive checksum mismatch')
        with zipfile.ZipFile(temporary) as z:
            if z.testzip() is not None:
                raise ValueError('ZIP integrity failure')
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    print('Restored:', archive, flush=True)
    return archive


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        # Only original model files; per-package documentation stays in the ZIP.
        files = json.loads(z.read('DOWNLOAD_MANIFEST.json'))
        for entry in files:
            relative = PurePosixPath(entry['repo']) / entry['path']
            if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative):
                raise ValueError('Unsafe model path')
            target = destination / relative
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Model path leaves destination')
            if target.exists():
                if target.stat().st_size != entry['size'] or digest(target) != entry['sha256']:
                    raise ValueError('Existing model file differs: ' + str(target))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.partial')
            try:
                with z.open(str(relative)) as src, temporary.open('wb') as out:
                    shutil.copyfileobj(src, out, 4 * 1024 * 1024)
                if temporary.stat().st_size != entry['size'] or digest(temporary) != entry['sha256']:
                    raise ValueError('Extracted file checksum mismatch: ' + str(relative))
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    print('Extracted and verified model files:', archive.name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['stage', 'restore', 'verify'])
    parser.add_argument('--source', type=Path, default=ROOT / 'dist')
    parser.add_argument('--parts', type=Path, default=ROOT / 'standalone-models')
    parser.add_argument('--output', type=Path, default=ROOT / 'dist')
    parser.add_argument('--destination', type=Path, default=ROOT / 'models')
    parser.add_argument('--model', choices=['all', *MODELS], default='all')
    parser.add_argument('--dtype', choices=['all', 'q8', 'fp16'], default='all')
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--asset', help='Exact release asset to stage')
    args = parser.parse_args()
    lock = json.loads((ROOT / 'repo-model-release.lock.json').read_text())
    assets = {e['name']: e for e in lock['assets']}
    if args.command == 'stage':
        if args.asset not in assets:
            parser.error('--asset must match the pinned release manifest')
        stage(args.source, args.parts, args.asset, assets[args.asset]['sha256'])
        return
    models = MODELS if args.model == 'all' else [args.model]
    dtypes = ['q8', 'fp16'] if args.dtype == 'all' else [args.dtype]
    for model in models:
        for dtype in dtypes:
            name = f'Xenova-{model}-{dtype}-standalone.zip'
            folder = args.parts / name.removesuffix('.zip')
            manifest = json.loads((folder / 'manifest.json').read_text())
            if manifest['sha256'] != assets[name]['sha256'] or manifest['archive'] != name:
                raise ValueError('Package differs from pinned release manifest')
            if args.command == 'verify':
                verify_parts(folder, manifest)
                print('Verified:', name, flush=True)
            else:
                archive = restore(folder, args.output)
                if args.extract:
                    extract(archive, args.destination)


if __name__ == '__main__':
    main()

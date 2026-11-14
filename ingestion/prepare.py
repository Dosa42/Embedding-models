"""Resolve the input ZIP and fetch only the selected Git LFS model package."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile
from .core import MODELS, WEIGHTS, sha256

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'standalone-models' / 'ingestion target'


def select_input(target, filename=''):
    if filename:
        if Path(filename).name != filename or any(c in filename for c in '\\/,*?[]\r\n') or not filename.lower().endswith('.zip'):
            raise ValueError('Use a ZIP filename without directories or glob characters')
        selected = target / filename
    else:
        candidates = sorted(p for p in target.glob('*') if p.is_file() and p.suffix.lower() == '.zip')
        if len(candidates) != 1:
            raise ValueError(f'Found {len(candidates)} ZIPs in ingestion target; enter the exact ZIP filename')
        selected = candidates[0]
    if not selected.is_file() or selected.is_symlink() or not selected.resolve().is_relative_to(target.resolve()):
        raise ValueError('Input ZIP is missing or outside ingestion target')
    return selected


def prepare(model, dtype, filename, work):
    source = select_input(TARGET, filename)
    archive = ROOT / 'standalone-models' / f'Xenova-{model}-{dtype}-standalone.zip'
    paths = [source.relative_to(ROOT).as_posix(), archive.relative_to(ROOT).as_posix()]
    subprocess.run(['git', 'lfs', 'pull', '--include=' + ','.join(paths), '--exclude='], cwd=ROOT, check=True)
    if not zipfile.is_zipfile(source):
        raise ValueError('Input is not a readable ZIP; check the LFS upload or archive format')
    release = json.loads((ROOT / 'repo-model-release.lock.json').read_text())
    expected = next(a for a in release['assets'] if a['name'] == archive.name)
    if archive.stat().st_size != expected['size'] or sha256(archive) != expected['sha256']:
        raise ValueError('Model package checksum mismatch')
    work = Path(work)
    model_dir = work / 'models' / 'Xenova' / model
    lock = json.loads((ROOT / 'xenova-models.lock.json').read_text())
    entries = [e for e in lock['files'] if e['repo'] == 'Xenova/' + model and (not e['path'].startswith('onnx/') or e['path'] == 'onnx/' + WEIGHTS[dtype])]
    with zipfile.ZipFile(archive) as z:
        for entry in entries:
            path = model_dir / entry['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(entry['repo'] + '/' + entry['path']) as src, path.open('wb') as dest:
                shutil.copyfileobj(src, dest, 4 * 1024 * 1024)
            if path.stat().st_size != entry['size'] or sha256(path) != entry['sha256']:
                raise ValueError('Model member checksum mismatch: ' + entry['path'])
    outputs = {'input_zip': str(source), 'model_dir': str(model_dir)}
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
            for key, value in outputs.items():
                if '\n' in value or '\r' in value:
                    raise ValueError('Newlines are not supported in input paths')
                stream.write(f'{key}={value}\n')
    print(json.dumps(outputs))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=list(MODELS), required=True)
    parser.add_argument('--dtype', choices=list(WEIGHTS), default='q8')
    parser.add_argument('--zip-filename', default='')
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.model, args.dtype, args.zip_filename, args.work)


if __name__ == '__main__':
    main()

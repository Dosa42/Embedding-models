#!/usr/bin/env python3
"""Verify complete model ZIPs or their Git LFS pointers against the release lock."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def verify_archive(path, asset):
    if not path.is_file() or path.stat().st_size != asset['size']:
        raise ValueError('Missing or incorrect size: ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(4 * 1024 * 1024), b''):
            h.update(block)
    if h.hexdigest() != asset['sha256']:
        raise ValueError('SHA-256 mismatch: ' + str(path))
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity failure: ' + str(path))
    print('Complete ZIP verified:', path.name, flush=True)


def verify_pointer(content, asset):
    expected = [
        'version https://git-lfs.github.com/spec/v1',
        'oid sha256:' + asset['sha256'],
        'size ' + str(asset['size']),
    ]
    if content.splitlines() != expected:
        raise ValueError('Incorrect LFS pointer: ' + asset['name'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'standalone-models')
    parser.add_argument('--git-ref', help='Verify LFS pointers at this Git ref instead of local ZIP bytes')
    parser.add_argument('--lfs-cache', action='store_true', help='Verify the six complete objects in the local Git LFS object store')
    args = parser.parse_args()
    lock = json.loads((ROOT / 'repo-model-release.lock.json').read_text())
    for asset in lock['assets']:
        if args.lfs_cache:
            common = Path(subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], text=True).strip())
            oid = asset['sha256']
            verify_archive(common / 'lfs/objects' / oid[:2] / oid[2:4] / oid, asset)
        elif args.git_ref:
            pointer = subprocess.check_output(['git', 'show', args.git_ref + ':standalone-models/' + asset['name']], text=True)
            verify_pointer(pointer, asset)
            print('LFS pointer verified:', asset['name'], flush=True)
        else:
            verify_archive(args.directory / asset['name'], asset)


if __name__ == '__main__':
    main()

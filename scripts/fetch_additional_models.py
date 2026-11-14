"""Download model data only; never import or execute downloaded model code."""
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.request import urlopen, Request
from urllib.parse import quote
import zipfile

REPOS = ['Xenova/nli-deberta-v3-small', 'vectara/hallucination_evaluation_model', 'Xenova/ms-marco-MiniLM-L-6-v2']
OUT = Path('standalone-models')

def get(url):
    return urlopen(Request(url, headers={'User-Agent': 'standalone-model-downloader'}), timeout=180)

def main():
    OUT.mkdir(exist_ok=True)
    packages = []
    for repo in REPOS:
        with get('https://huggingface.co/api/models/' + repo + '?blobs=true') as response:
            info = json.load(response)
        revision = info['sha']
        variants = {'q8': 'onnx/model_quantized.onnx', 'fp16': 'onnx/model_fp16.onnx'} if repo.startswith('Xenova/') else {'original': None}
        for dtype, weight in variants.items():
            members = [e for e in info['siblings'] if (not e['rfilename'].startswith('onnx/') or e['rfilename'] == weight)] if weight else info['siblings']
            if weight and weight not in [e['rfilename'] for e in members]:
                raise ValueError('Requested ONNX variant missing: ' + repo + '/' + weight)
            filename = repo.replace('/', '-') + '-' + dtype + '-standalone.zip'
            records = []
            with zipfile.ZipFile(OUT / filename, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                for entry in members:
                    name = entry['rfilename']
                    path = PurePosixPath(name)
                    if path.is_absolute() or '..' in path.parts or '\\' in name:
                        raise ValueError('Invalid source path')
                    size = entry['size']
                    digest = hashlib.sha256()
                    gitdigest = hashlib.sha1(('blob ' + str(size) + '\0').encode())
                    count = 0
                    with get('https://huggingface.co/' + repo + '/resolve/' + revision + '/' + quote(name, safe='/') + '?download=true') as response, archive.open(repo + '/' + name, 'w', force_zip64=True) as dest:
                        while block := response.read(4 * 1024 * 1024):
                            count += len(block)
                            digest.update(block)
                            gitdigest.update(block)
                            dest.write(block)
                    if count != size:
                        raise ValueError('Size mismatch: ' + name)
                    if entry.get('lfs'):
                        if digest.hexdigest() != entry['lfs']['sha256']:
                            raise ValueError('LFS checksum mismatch: ' + name)
                    elif gitdigest.hexdigest() != entry['blobId']:
                        raise ValueError('Git checksum mismatch: ' + name)
                    records.append({'path': name, 'size': count, 'sha256': digest.hexdigest()})
                    print(repo, dtype, name, count, flush=True)
                if repo.startswith('vectara/'):
                    dependency = 'google/flan-t5-base'
                    with get('https://huggingface.co/api/models/' + dependency + '?blobs=true') as response:
                        foundation = json.load(response)
                    required = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'spiece.model'}
                    selected = [e for e in foundation['siblings'] if e['rfilename'] in required]
                    if {e['rfilename'] for e in selected} != required:
                        raise ValueError('Incomplete foundation tokenizer/config')
                    for entry in selected:
                        name = entry['rfilename']
                        with get('https://huggingface.co/' + dependency + '/resolve/' + foundation['sha'] + '/' + name) as response:
                            data = response.read()
                        digest = hashlib.sha256(data).hexdigest()
                        expected = entry.get('lfs', {}).get('sha256')
                        actual = digest if expected else hashlib.sha1(('blob ' + str(len(data)) + '\0').encode() + data).hexdigest()
                        if len(data) != entry['size'] or actual != (expected or entry['blobId']):
                            raise ValueError('Foundation checksum mismatch: ' + name)
                        archive.writestr(dependency + '/' + name, data)
                        records.append({'repository': dependency, 'revision': foundation['sha'], 'path': name, 'size': len(data), 'sha256': digest})
                    archive.writestr('OFFLINE-USAGE.txt', 'Vectara original model, not an ONNX/q8 conversion.\nIncludes upstream Python model code, weights and the foundation tokenizer/config.\nAfter extracting, set config.foundation to the absolute local google/flan-t5-base directory before constructing the model.\nLoad the Vectara config/model from its local directory with local_files_only=True.\nThe original custom model implementation requires trust_remote_code=True to load; the downloader never executes it.\nPython, PyTorch and compatible Transformers dependencies must be installed separately.\n')
                manifest = {'repository': repo, 'revision': revision, 'variant': dtype, 'files': records}
                archive.writestr('STANDALONE-MANIFEST.json', json.dumps(manifest, indent=2))
            with zipfile.ZipFile(OUT / filename) as archive:
                if archive.testzip() is not None:
                    raise ValueError('ZIP integrity failure')
            digest = hashlib.sha256()
            with (OUT / filename).open('rb') as stream:
                while block := stream.read(4 * 1024 * 1024):
                    digest.update(block)
            packages.append({'name': filename, 'sha256': digest.hexdigest(), 'size': (OUT / filename).stat().st_size, 'repository': repo, 'revision': revision, 'variant': dtype})
    (OUT / 'additional-models.lock.json').write_text(json.dumps({'packages': packages}, indent=2) + '\n')

if __name__ == '__main__':
    main()

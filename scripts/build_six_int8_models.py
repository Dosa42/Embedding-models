import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from urllib.request import urlopen
import zipfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
from safetensors.numpy import load_file
from tokenizers import Tokenizer


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def graphs(graph):
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from graphs(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    yield from graphs(child)


def quantize_potion(source, target):
    model = onnx.load(source)
    tables = {}
    for tensor in model.graph.initializer:
        if tensor.data_type == TensorProto.FLOAT and len(tensor.dims) == 2:
            weight = numpy_helper.to_array(tensor)
            if weight.shape[0] > 1000:
                scale = np.maximum(np.max(np.abs(weight), axis=1, keepdims=True) / 127, np.finfo(np.float32).tiny)
                quantized = np.clip(np.rint(weight / scale), -127, 127).astype(np.int8)
                tables[tensor.name] = (quantized, scale.astype(np.float32))
    if len(tables) != 1:
        raise ValueError('Expected one static embedding table')
    rewritten = 0
    for graph in reversed(list(graphs(model.graph))):
        nodes = []
        for node in graph.node:
            if node.op_type != 'Gather' or node.input[0] not in tables:
                nodes.append(node)
                continue
            axis = next((a.i for a in node.attribute if a.name == 'axis'), 0)
            if axis != 0:
                raise ValueError('Unsupported embedding gather axis')
            name, indices = node.input[:2]
            out = node.output[0]
            nodes.extend([
                helper.make_node('Gather', [name + '_int8', indices], [out + '_int8'], axis=0),
                helper.make_node('Gather', [name + '_scale', indices], [out + '_scale'], axis=0),
                helper.make_node('Cast', [out + '_int8'], [out + '_float'], to=TensorProto.FLOAT),
                helper.make_node('Mul', [out + '_float', out + '_scale'], [out]),
            ])
            rewritten += 1
        del graph.node[:]
        graph.node.extend(nodes)
    if not rewritten or any(name in node.input for graph in graphs(model.graph) for node in graph.node for name in tables):
        raise ValueError('Embedding table was not fully converted')
    retained = [t for t in model.graph.initializer if t.name not in tables]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    for name, (weight, scale) in tables.items():
        model.graph.initializer.extend([numpy_helper.from_array(weight, name + '_int8'), numpy_helper.from_array(scale, name + '_scale')])
    onnx.checker.check_model(model)
    onnx.save(model, target)
    return {'method': 'symmetric signed INT8 per embedding row; gather before dequantization', 'gathers_rewritten': rewritten}


def complete_ettin(source, target, root):
    model = onnx.load(source)
    if [v.name for v in model.graph.output] != ['last_hidden_state']:
        raise ValueError('Unexpected Ettin backbone output')
    pooling = json.loads((root / '1_Pooling/config.json').read_text())
    if pooling['pooling_mode'] != 'cls':
        raise ValueError('Unexpected Ettin pooling')
    dense = load_file(root / '2_Dense/model.safetensors')
    norm = load_file(root / '3_LayerNorm/model.safetensors')
    score = load_file(root / '4_Dense/model.safetensors')
    print('Ettin head keys', list(dense), list(norm), list(score), flush=True)
    values = {
        'head_cls_index': np.array(0, dtype=np.int64),
        'head_dense_weight': dense['linear.weight'].T.copy(),
        'head_norm_weight': norm['norm.weight'],
        'head_norm_bias': norm['norm.bias'],
        'head_score_weight': score['linear.weight'].T.copy(),
        'head_score_bias': score['linear.bias'],
        'head_sqrt2': np.array(math.sqrt(2), dtype=np.float32),
        'head_one': np.array(1, dtype=np.float32),
        'head_half': np.array(0.5, dtype=np.float32),
    }
    model.graph.initializer.extend(numpy_helper.from_array(v, k) for k, v in values.items())
    model.graph.node.extend([
        helper.make_node('Gather', ['last_hidden_state', 'head_cls_index'], ['head_cls'], axis=1),
        helper.make_node('MatMul', ['head_cls', 'head_dense_weight'], ['head_dense']),
        helper.make_node('Div', ['head_dense', 'head_sqrt2'], ['head_div']),
        helper.make_node('Erf', ['head_div'], ['head_erf']),
        helper.make_node('Add', ['head_erf', 'head_one'], ['head_plus']),
        helper.make_node('Mul', ['head_dense', 'head_half'], ['head_halved']),
        helper.make_node('Mul', ['head_halved', 'head_plus'], ['head_gelu']),
        helper.make_node('LayerNormalization', ['head_gelu', 'head_norm_weight', 'head_norm_bias'], ['head_norm'], axis=-1, epsilon=1e-5),
        helper.make_node('MatMul', ['head_norm', 'head_score_weight'], ['head_score']),
        helper.make_node('Add', ['head_score', 'head_score_bias'], ['scores']),
    ])
    del model.graph.output[:]
    model.graph.output.extend([helper.make_tensor_value_info('scores', TensorProto.FLOAT, ['batch_size', 1])])
    onnx.checker.check_model(model)
    onnx.save(model, target)
    return {'method': 'publisher ARM64 INT8 backbone plus original CLS, Dense/GELU, LayerNorm and scoring head', 'head_precision': 'FP32', 'values': values}


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def feeds(runtime, tokenizer, kind, batch):
    if kind == 'potion':
        encoded = [tokenizer.encode(t, add_special_tokens=False).ids for t in batch]
        return {'input_ids': np.array([i for row in encoded for i in row], dtype=np.int64), 'offsets': np.array(np.cumsum([0] + [len(row) for row in encoded[:-1]]), dtype=np.int64)}
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id('[PAD]') or tokenizer.token_to_id('<pad>') or 0)
    encoded = tokenizer.encode_batch(batch)
    values = {'input_ids': [x.ids for x in encoded], 'attention_mask': [x.attention_mask for x in encoded], 'token_type_ids': [x.type_ids for x in encoded]}
    return {i.name: np.array(values[i.name], dtype=np.int64) for i in runtime.get_inputs()}


def verify(model_path, root, spec, original, conversion):
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    counts = Counter()
    for graph in graphs(model.graph):
        for tensor in graph.initializer:
            counts[TensorProto.DataType.Name(tensor.data_type)] += math.prod(tensor.dims)
    if counts['INT8'] < 1_000_000:
        raise ValueError('Missing substantial signed INT8 weights')
    runtime = session(model_path)
    tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
    texts = ['The cat sits on the mat.', 'Mars is the red planet.', 'Türkçe bir örnek cümle.', 'A longer sentence with several words tests variable sequence lengths.']
    results = []
    reference = session(original) if spec['kind'] in ['potion', 'ettin'] else None
    for size in [1, 3]:
        batch = texts[:size]
        if spec['kind'] in ['ettin', 'reranker', 'nli']:
            batch = [('Which planet is red?', t) for t in batch]
        inputs = feeds(runtime, tokenizer, spec['kind'], batch)
        outputs = runtime.run(None, inputs)
        if any(not np.isfinite(v).all() for v in outputs):
            raise ValueError('Non-finite inference output')
        first = outputs[0]
        if first.shape[0] != size:
            raise ValueError('Incorrect batch dimension')
        if spec['kind'] in ['ettin', 'reranker'] and first.shape != (size, 1):
            raise ValueError('Reranker must output one score per pair')
        if spec['kind'] == 'nli' and first.shape != (size, 3):
            raise ValueError('NLI must output three class logits')
        record = {'batch': size, 'outputs': {o.name: list(v.shape) for o, v in zip(runtime.get_outputs(), outputs)}}
        if spec['kind'] == 'potion':
            baseline = reference.run(None, inputs)[0]
            cosine = np.sum(first * baseline, axis=-1) / np.maximum(np.linalg.norm(first, axis=-1) * np.linalg.norm(baseline, axis=-1), 1e-12)
            record['min_cosine_vs_fp32'] = float(cosine.min())
            record['max_absolute_error'] = float(np.max(np.abs(first - baseline)))
            if cosine.min() < 0.999:
                raise ValueError('Potion quantization accuracy check failed')
        if spec['kind'] == 'ettin':
            values = conversion['values']
            hidden = reference.run(None, inputs)[0][:, 0]
            dense = hidden @ values['head_dense_weight']
            gelu = dense * 0.5 * (1 + np.vectorize(math.erf)(dense / math.sqrt(2)))
            normalized = (gelu - gelu.mean(-1, keepdims=True)) / np.sqrt(gelu.var(-1, keepdims=True) + 1e-5)
            expected = (normalized * values['head_norm_weight'] + values['head_norm_bias']) @ values['head_score_weight'] + values['head_score_bias']
            np.testing.assert_allclose(first, expected, rtol=1e-4, atol=1e-4)
            record['max_head_error_vs_numpy'] = float(np.max(np.abs(first - expected)))
        results.append(record)
    return {'initializer_elements_by_dtype': dict(counts), 'runtime': ort.__version__, 'provider': 'CPUExecutionProvider', 'smoke_tests': results}


def download(spec, root):
    repo, revision = spec['repository'], spec['revision']
    with urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true', timeout=90) as response:
        info = json.load(response)
    records = []
    for entry in info['siblings']:
        name = entry['rfilename']
        wanted = name == spec['source_onnx'] or name.endswith(('.json', '.txt', '.model', '.md')) or name.startswith('LICENSE')
        wanted = wanted or (spec['kind'] == 'ettin' and name in ['2_Dense/model.safetensors', '3_LayerNorm/model.safetensors', '4_Dense/model.safetensors'])
        if not wanted:
            continue
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
        for attempt in range(3):
            try:
                with urlopen(url, timeout=180) as response, path.open('wb') as dest:
                    shutil.copyfileobj(response, dest)
                if path.stat().st_size != entry['size']:
                    raise ValueError('Download size mismatch: ' + name)
                checksum = digest(path)
                if entry.get('lfs') and checksum != entry['lfs']['sha256']:
                    raise ValueError('Download SHA256 mismatch: ' + name)
                if not entry.get('lfs'):
                    data = path.read_bytes()
                    if hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() != entry['blobId']:
                        raise ValueError('Download Git blob mismatch: ' + name)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        records.append({'path': name, 'sha256': checksum, 'size': path.stat().st_size, 'url': url})
    return records


def package(spec, out, work):
    root = work / spec['repository']
    records = download(spec, root)
    source = root / spec['source_onnx']
    target = root / 'onnx/model_int8.onnx'
    conversion = {'method': 'publisher signed INT8 ONNX; unchanged'}
    if spec['kind'] == 'potion':
        conversion = quantize_potion(source, target)
    elif spec['kind'] == 'ettin':
        conversion = complete_ettin(source, target, root)
    report = verify(target, root, spec, source, conversion)
    conversion.pop('values', None)
    if source != target:
        source.unlink()
    files = [{'path': p.relative_to(root).as_posix(), 'size': p.stat().st_size, 'sha256': digest(p)} for p in sorted(root.rglob('*')) if p.is_file()]
    manifest = {**spec, 'variant': 'int8', 'onnx_file': 'onnx/model_int8.onnx', 'conversion': conversion, 'verification': report, 'upstream_files': records, 'files': files}
    (root / 'STANDALONE-MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    note = 'Load onnx/model_int8.onnx with ONNX Runtime and the included tokenizer. INT8 refers to quantized weights; floating-point operators, scales and outputs remain.\n'
    if spec['kind'] == 'potion':
        note += 'Model2Vec static embeddings: tokenize without special tokens; flatten token IDs into input_ids (int64), and supply one starting offset per sentence in offsets (int64). Output embeddings are already pooled and normalized.\n'
    elif spec['kind'] == 'ettin':
        note += 'The original publisher ONNX is only the backbone. This ONNX includes the original FP32 scoring head and outputs scores [batch, 1]. Tokenize query/document pairs. The source repository replaces the inaccessible requested onnx-community repository.\n'
    elif spec['kind'] == 'nli':
        note += 'NLI classification model, not a relevance-trained reranker. Tokenize premise/hypothesis pairs; interpret the three logits using config.json id2label.\n'
    elif spec['kind'] == 'embedding':
        note += 'The graph outputs token embeddings. Apply attention-mask mean pooling for multilingual MiniLM; use the CLS token for BGE, then L2-normalize.\n'
    else:
        note += 'Tokenize query/document pairs; output is the relevance score.\n'
    (root / 'OFFLINE-USAGE.txt').write_text(note)
    name = spec['repository'].replace('/', '-') + '-int8-standalone.zip'
    target_zip = out / name
    with zipfile.ZipFile(target_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                member = spec['repository'] + '/' + path.relative_to(root).as_posix()
                entry = zipfile.ZipInfo(member, (2026, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o644 << 16
                archive.writestr(entry, path.read_bytes())
    with zipfile.ZipFile(target_zip) as archive:
        if archive.testzip() is not None:
            raise ValueError('Invalid model ZIP')
        for entry in files:
            data = archive.read(spec['repository'] + '/' + entry['path'])
            if hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError('Packaged file checksum mismatch')
    result = {**spec, 'name': name, 'size': target_zip.stat().st_size, 'sha256': digest(target_zip), 'onnx_size': target.stat().st_size, 'onnx_sha256': digest(target), 'conversion': conversion, 'verification': report}
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('standalone-models'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    specs = json.loads(Path(__file__).with_name('six_int8_models.json').read_text())
    with tempfile.TemporaryDirectory(prefix='six-int8-') as directory:
        packages = [package(spec, args.output, Path(directory)) for spec in specs]
    result = {'schema': 1, 'packages': packages, 'validation_scope': 'ONNX checker, signed INT8 weights, CPU inference for batch sizes 1 and 3; Potion FP32 comparison and Ettin head equivalence. Not a task-quality benchmark.'}
    (args.output / 'six-int8-models.lock.json').write_text(json.dumps(result, indent=2) + '\n')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
            stream.write('| Model | ONNX MB | ZIP |\n|---|---:|---|\n')
            for p in packages:
                stream.write(f"| {p['repository']} | {p['onnx_size'] / 1e6:.2f} | {p['name']} |\n")


if __name__ == '__main__':
    main()

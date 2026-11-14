import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
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
from onnxruntime.quantization import QuantType, quantize_dynamic

from build_six_int8_models import digest, graphs, session
from next_nine_runtime import colbert_inputs, gliner_inputs, maxsim, text_inputs, tokenizer_at


class QuantizationAccuracyError(ValueError):
    pass


def quantize_embedding_tables(source, target):
    model = onnx.load(source)
    gathered = {n.input[0] for g in graphs(model.graph) for n in g.node if n.op_type == 'Gather'}
    tables = {}
    for tensor in model.graph.initializer:
        if tensor.name in gathered and tensor.data_type == TensorProto.FLOAT and len(tensor.dims) == 2:
            weight = numpy_helper.to_array(tensor)
            scale = np.maximum(np.max(np.abs(weight), axis=1, keepdims=True) / 127, np.finfo(np.float32).tiny)
            tables[tensor.name] = (np.clip(np.rint(weight / scale), -127, 127).astype(np.int8), scale.astype(np.float32))
    if not tables:
        raise ValueError('No embedding tables found')
    for graph in reversed(list(graphs(model.graph))):
        nodes = []
        for node in graph.node:
            if node.op_type != 'Gather' or node.input[0] not in tables:
                nodes.append(node)
                continue
            if next((a.i for a in node.attribute if a.name == 'axis'), 0) != 0:
                raise ValueError('Unexpected embedding gather axis')
            name, indices = node.input[:2]
            out = node.output[0]
            nodes.extend([
                helper.make_node('Gather', [name + '_row_int8', indices], [out + '_row_int8'], axis=0),
                helper.make_node('Gather', [name + '_row_scale', indices], [out + '_row_scale'], axis=0),
                helper.make_node('Cast', [out + '_row_int8'], [out + '_row_float'], to=TensorProto.FLOAT),
                helper.make_node('Mul', [out + '_row_float', out + '_row_scale'], [out]),
            ])
        del graph.node[:]
        graph.node.extend(nodes)
    if any(name in node.input for graph in graphs(model.graph) for node in graph.node for name in tables):
        raise ValueError('Unconverted embedding table use')
    retained = [t for t in model.graph.initializer if t.name not in tables]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    for name, (weight, scale) in tables.items():
        model.graph.initializer.extend([numpy_helper.from_array(weight, name + '_row_int8'), numpy_helper.from_array(scale, name + '_row_scale')])
    onnx.checker.check_model(model)
    onnx.save(model, target)
    return list(tables)


def quantize_matrix_weights(source, target):
    model = onnx.load(source)
    tensors = {t.name: t for t in model.graph.initializer}
    selected = {}
    for graph in graphs(model.graph):
        for node in graph.node:
            if node.op_type not in ['MatMul', 'Gemm'] or node.input[1] not in tensors:
                continue
            tensor = tensors[node.input[1]]
            if tensor.data_type != TensorProto.FLOAT or len(tensor.dims) != 2:
                continue
            transposed = next((a.i for a in node.attribute if a.name == 'transB'), 0)
            axis = 0 if node.op_type == 'Gemm' and transposed else 1
            if tensor.name in selected and selected[tensor.name] != axis:
                raise ValueError('Conflicting matrix weight axes')
            selected[tensor.name] = axis
    if not selected:
        raise ValueError('No matrix weights found')
    new_tensors = []
    dequantizers = []
    for name, axis in selected.items():
        weight = numpy_helper.to_array(tensors[name])
        scale = np.maximum(np.max(np.abs(weight), axis=1 - axis) / 127, np.finfo(np.float32).tiny).astype(np.float32)
        quantized = np.clip(np.rint(weight / (scale[None, :] if axis == 1 else scale[:, None])), -127, 127).astype(np.int8)
        new_tensors.extend([numpy_helper.from_array(quantized, name + '_weight_int8'), numpy_helper.from_array(scale, name + '_weight_scale'), numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), name + '_weight_zero')])
        dequantizers.append(helper.make_node('DequantizeLinear', [name + '_weight_int8', name + '_weight_scale', name + '_weight_zero'], [name], axis=axis))
    retained = [t for t in model.graph.initializer if t.name not in selected]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained + new_tensors)
    nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(dequantizers + nodes)
    onnx.checker.check_model(model)
    onnx.save(model, target)
    return list(selected)


def metadata(repo, revision):
    with urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true', timeout=90) as response:
        return json.load(response)


def download(repo, revision, root, selected):
    info = metadata(repo, revision)
    entries = [e for e in info['siblings'] if selected(e['rfilename'])]
    records = []
    for entry in entries:
        name = entry['rfilename']
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
        for attempt in range(3):
            try:
                with urlopen(url, timeout=180) as response, path.open('wb') as stream:
                    shutil.copyfileobj(response, stream)
                if path.stat().st_size != entry['size']:
                    raise ValueError('Download length mismatch: ' + name)
                checksum = digest(path)
                if entry.get('lfs'):
                    if checksum != entry['lfs']['sha256']:
                        raise ValueError('Download SHA256 mismatch: ' + name)
                else:
                    data = path.read_bytes()
                    if hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() != entry['blobId']:
                        raise ValueError('Git blob mismatch: ' + name)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        records.append({'repository': repo, 'revision': revision, 'path': name, 'size': path.stat().st_size, 'sha256': checksum, 'url': url})
    return records


def verify(root, target, spec, source=None):
    model = onnx.load(target)
    onnx.checker.check_model(model)
    counts = Counter()
    for graph in graphs(model.graph):
        for tensor in graph.initializer:
            counts[TensorProto.DataType.Name(tensor.data_type)] += math.prod(tensor.dims)
    eight = counts['INT8'] + counts['UINT8']
    floating = counts['FLOAT'] + counts['FLOAT16'] + counts['DOUBLE']
    if eight == 0 or eight / (eight + floating) < 0.8:
        raise ValueError('Weights are not predominantly quantized to 8 bit')
    del model
    gc.collect()
    runtime = session(target)
    print('MODEL_IO', spec['key'], [(v.name, v.shape, v.type) for v in runtime.get_inputs()], [(v.name, v.shape) for v in runtime.get_outputs()], flush=True)
    tokenizer = tokenizer_at(root)
    reference = session(source) if source is not None else None
    tests = []
    texts = ['John Smith works at Microsoft in Paris.', 'Ayşe İstanbul şehrinde yaşıyor.', 'Berlin is the capital of Germany.']
    for batch_size in [1, 3]:
        batch = texts[:batch_size]
        if spec['kind'] == 'gliner':
            config = json.loads((root / 'gliner_config.json').read_text())
            inputs = gliner_inputs(runtime, tokenizer, batch, ['person', 'organization', 'location'], config)
        elif spec['group'] == '05_colbert':
            inputs = colbert_inputs(runtime, tokenizer, batch)
        else:
            if spec['kind'] == 'zero_shot':
                batch = [(text, 'This text is about travel.') for text in batch]
            inputs = text_inputs(runtime, tokenizer, batch)
        outputs = runtime.run(None, inputs)
        if any(not np.isfinite(value).all() for value in outputs):
            raise ValueError('Non-finite model outputs')
        value = outputs[0]
        if value.shape[0] != batch_size:
            raise ValueError('Batch dimension mismatch')
        config = json.loads((root / 'config.json').read_text())
        if spec['kind'] == 'ner' and (value.ndim != 3 or value.shape[-1] != len(config['id2label'])):
            raise ValueError('NER label output mismatch')
        if spec['kind'] == 'zero_shot' and value.shape != (batch_size, len(config['id2label'])):
            raise ValueError('NLI class output mismatch')
        if spec['kind'] == 'gliner' and (value.ndim != 4 or value.shape[-1] != 3):
            raise ValueError('GLiNER span/class output mismatch')
        record = {'batch': batch_size, 'outputs': {o.name: list(v.shape) for o, v in zip(runtime.get_outputs(), outputs)}}
        if spec['group'] == '05_colbert':
            if value.ndim != 3 or value.shape[-1] != spec['dimension']:
                raise ValueError('Missing trained ColBERT projection')
            expected = reference.run(None, inputs)[0]
            active = inputs['attention_mask'].astype(bool)
            a, b = value[active], expected[active]
            cosine = np.sum(a * b, axis=-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)
            record.update(min_token_cosine_vs_fp32=float(cosine.min()), mean_token_cosine_vs_fp32=float(cosine.mean()), max_absolute_error=float(np.max(np.abs(a - b))))
            print('COLBERT_ACCURACY', json.dumps(record), flush=True)
            if cosine.min() < 0.98:
                raise QuantizationAccuracyError(f'ColBERT minimum token cosine {float(cosine.min()):.6f} below 0.98')
        tests.append(record)
    if spec['group'] == '05_colbert':
        query = colbert_inputs(runtime, tokenizer, ['Where is the Eiffel Tower?'], query=True)
        docs = colbert_inputs(runtime, tokenizer, ['The Eiffel Tower is in Paris, France.', 'Bananas are yellow fruit.'])
        query_vectors = runtime.run(None, query)[0][0]
        document_vectors = runtime.run(None, docs)[0]
        scores = [maxsim(query_vectors, vectors, ids, tokenizer) for vectors, ids in zip(document_vectors, docs['input_ids'])]
        q_ref = reference.run(None, query)[0][0]
        d_ref = reference.run(None, docs)[0]
        scores_ref = [maxsim(q_ref, vectors, ids, tokenizer) for vectors, ids in zip(d_ref, docs['input_ids'])]
        if not np.isfinite(scores).all() or int(np.argmax(scores)) != int(np.argmax(scores_ref)):
            raise QuantizationAccuracyError('ColBERT MaxSim smoke-test rank changed')
        tests.append({'maxsim_int8': scores, 'maxsim_fp32': scores_ref, 'same_top_document': True})
    return {'initializer_elements_by_dtype': dict(counts), 'quantized_weight_fraction': eight / (eight + floating), 'provider': 'CPUExecutionProvider', 'tests': tests, 'dependencies': {name: importlib.metadata.version(name) for name in ['numpy', 'onnx', 'onnxruntime', 'tokenizers', 'safetensors']}}


def build(spec, out):
    with tempfile.TemporaryDirectory(prefix=spec['key'] + '-') as temp:
        root = Path(temp) / spec['repository']
        is_jina = spec['kind'] == 'jina_colbert'
        def wanted(name):
            return name == spec['source_onnx'] or name.endswith(('.json', '.txt', '.model', '.md', '.metadata')) or name.startswith('LICENSE') or (is_jina and name == 'model.safetensors')
        upstream = download(spec['repository'], spec['revision'], root, wanted)
        if spec['repository'].startswith('libryo-ai/'):
            shutil.copyfile(root / 'answerai_colbert_small_v1_tokenizer.json', root / 'tokenizer.json')
            interface = {'model_type': 'colbert', 'dimension': spec['dimension'], 'repository': spec['repository'], 'configuration_origin': 'package interface metadata; upstream supplies only ONNX and tokenizer', 'query_marker': '[unused0]', 'document_marker': '[unused1]'}
            (root / 'config.json').write_text(json.dumps(interface, indent=2) + '\n')
        target = root / 'onnx/model_int8.onnx'
        target.parent.mkdir(parents=True, exist_ok=True)
        conversion = {'method': 'upstream 8-bit ONNX unchanged'}
        reference = None
        if spec['group'] == '05_colbert':
            if is_jina:
                from export_jina_colbert import export
                source = Path(temp) / 'jina_fp32.onnx'
                conversion = export(root, source)
                (root / 'model.safetensors').unlink()
            else:
                source = root / spec['source_onnx']
                conversion = {'method': 'INT8 weight quantization; preserved trained token projection and normalization'}
            prepared = Path(temp) / 'rowwise_embeddings.onnx'
            conversion['rowwise_int8_embedding_tables'] = quantize_embedding_tables(source, prepared)
            conversion['quantization'] = 'symmetric per-row signed INT8 embedding tables; dynamic per-channel QInt8 MatMul/Gemm'
            reference = source
            rejected = []
            modes = ['weight_only'] if spec.get('quantization_mode') == 'weight_only' else ['dynamic_full', 'dynamic_reduced', 'weight_only']
            for mode in modes:
                if mode == 'weight_only':
                    quantize_matrix_weights(prepared, target)
                else:
                    quantize_dynamic(str(prepared), str(target), per_channel=True, reduce_range=mode == 'dynamic_reduced', weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul', 'Gemm'], extra_options={'MatMulConstBOnly': True})
                try:
                    report = verify(root, target, spec, reference)
                    conversion.update(mode=mode, rejected_candidates=rejected)
                    if mode == 'weight_only':
                        conversion['quantization'] = 'W8A32: symmetric signed INT8 embedding rows and per-output-channel matrix weights; standard DequantizeLinear with FP32 matrix computation'
                        conversion['runtime_memory_note'] = 'Runtime can expand/fold weights to FP32; reduced file size is not a guarantee of reduced runtime memory or faster INT8 arithmetic.'
                    break
                except QuantizationAccuracyError as error:
                    rejected.append({'mode': mode, 'reason': str(error)})
                    gc.collect()
                    if mode == modes[-1]:
                        raise
            prepared.unlink()
        else:
            source = root / spec['source_onnx']
            if source != target:
                shutil.copyfile(source, target)
            report = verify(root, target, spec)
        if source != target:
            source.unlink()
        shutil.copyfile(Path(__file__).with_name('next_nine_runtime.py'), root / 'onnx_runtime_inputs.py')
        usage = 'Load onnx/model_int8.onnx with ONNX Runtime. onnx_runtime_inputs.py provides the tested model-specific input builders. Runtime dependencies: numpy, tokenizers, onnxruntime.\n'
        if spec['kind'] == 'gliner':
            usage += 'GLiNER needs entity-label prompts, first-subtoken word masks, text lengths and candidate spans. Use gliner_inputs with gliner_config.json and the desired labels. Its first output contains span logits [batch, words, max_width, labels]; apply sigmoid and span decoding for entity extraction.\n'
        elif spec['kind'] == 'ner':
            usage += 'Use text_inputs. Output logits are per token; map IDs using config.json id2label and aggregate BIO labels into entities as needed.\n'
        elif spec['kind'] == 'zero_shot':
            usage += 'Use text_inputs on premise/hypothesis pairs, e.g. (text, This text is about travel.). Read id2label from config.json: some models have two classes, others three. Use entailment/non-entailment according to those labels when scoring candidate labels.\n'
        else:
            usage += 'Use colbert_inputs(query=True) for query marker and MASK augmentation; query=False for document marker. Output is a vector per token, retaining the trained projection. maxsim applies document punctuation/padding filtering and sums the maximum similarity for each query token. This is not a pooled sentence encoder.\n'
        usage += 'The INT8/UINT8/FLOAT tensor counts and exact validation scope are recorded in STANDALONE-MANIFEST.json. These CPU smoke checks do not constitute a task-quality benchmark or Android/Web validation.\n'
        if conversion.get('mode') == 'weight_only':
            usage += 'This model is W8A32: weights are stored as signed INT8, while activations/matrix computation remain FP32. The runtime may expand weights in memory. This preserves accuracy better than the rejected dynamic activation-quantized candidates; it does not promise full-INT8 arithmetic speedups.\n'
        (root / 'OFFLINE-USAGE.txt').write_text(usage)
        files = [{'path': p.relative_to(root).as_posix(), 'size': p.stat().st_size, 'sha256': digest(p)} for p in sorted(root.rglob('*')) if p.is_file()]
        manifest = {**spec, 'onnx_file': 'onnx/model_int8.onnx', 'conversion': conversion, 'verification': report, 'upstream_files': upstream, 'files': files}
        (root / 'STANDALONE-MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
        name = spec['repository'].replace('/', '-') + '-int8-standalone.zip'
        archive_path = out / name
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    entry = zipfile.ZipInfo(spec['repository'] + '/' + path.relative_to(root).as_posix(), (2026, 1, 1, 0, 0, 0))
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    entry.external_attr = 0o644 << 16
                    archive.writestr(entry, path.read_bytes())
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError('ZIP CRC mismatch')
            for file in files:
                if hashlib.sha256(archive.read(spec['repository'] + '/' + file['path'])).hexdigest() != file['sha256']:
                    raise ValueError('Packaged file hash mismatch')
        result = {**spec, 'name': name, 'size': archive_path.stat().st_size, 'sha256': digest(archive_path), 'onnx_size': target.stat().st_size, 'onnx_sha256': digest(target), 'conversion': conversion, 'verification': report}
        (out / ('package-' + spec['key'] + '.json')).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('standalone-models'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    specs = json.loads(Path(__file__).with_name('next_nine_int8_models.json').read_text())
    if args.finalize:
        packages = [json.loads((args.output / ('package-' + s['key'] + '.json')).read_text()) for s in specs]
        for spec, package in zip(specs, packages):
            if any(package[k] != spec[k] for k in ['repository', 'revision', 'key']):
                raise ValueError('Package identity mismatch')
            if digest(args.output / package['name']) != package['sha256']:
                raise ValueError('Collected ZIP hash mismatch')
            conversion = package['conversion']
            if conversion.get('mode') == 'weight_only' and conversion.get('method', '').startswith('dynamic INT8'):
                conversion['method'] = 'INT8 weight quantization; preserved trained token projection and normalization'
                archive_path = args.output / package['name']
                replacement = archive_path.with_suffix('.tmp')
                manifest_path = package['repository'] + '/STANDALONE-MANIFEST.json'
                with zipfile.ZipFile(archive_path) as archive, zipfile.ZipFile(replacement, 'w', zipfile.ZIP_DEFLATED) as updated:
                    manifest = json.loads(archive.read(manifest_path))
                    manifest['conversion'] = conversion
                    for member in archive.infolist():
                        data = (json.dumps(manifest, indent=2) + '\n').encode() if member.filename == manifest_path else archive.read(member)
                        updated.writestr(member, data)
                with zipfile.ZipFile(replacement) as archive:
                    if archive.testzip() is not None:
                        raise ValueError('Metadata correction ZIP check failed')
                    for item in manifest['files']:
                        data = archive.read(package['repository'] + '/' + item['path'])
                        if hashlib.sha256(data).hexdigest() != item['sha256']:
                            raise ValueError('Model bytes changed during metadata correction')
                replacement.replace(archive_path)
                package.update(size=archive_path.stat().st_size, sha256=digest(archive_path))
                print('Corrected descriptive method; verified all model and tokenizer bytes unchanged:', package['name'], flush=True)
        (args.output / 'next-nine-int8-models.lock.json').write_text(json.dumps({'schema': 1, 'packages': packages}, indent=2) + '\n')
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
                stream.write('| Model | ONNX MB | ZIP MB |\n|---|---:|---:|\n')
                for p in packages:
                    stream.write(f"| {p['repository']} | {p['onnx_size']/1e6:.2f} | {p['size']/1e6:.2f} |\n")
    else:
        build(next(spec for spec in specs if spec['key'] == args.model), args.output)


if __name__ == '__main__':
    main()

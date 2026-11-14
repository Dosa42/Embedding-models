import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
import onnx
from onnx import TensorProto
from onnxruntime.quantization import quantize_dynamic, QuantType

from build_six_int8_models import digest, graphs
from build_next_nine_int8 import download, quantize_embedding_tables, quantize_matrix_weights
from final_six_runtime import encode, generate, session


def inspect(path):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    counts = Counter()
    for graph in graphs(model.graph):
        for tensor in graph.initializer:
            counts[TensorProto.DataType.Name(tensor.data_type)] += math.prod(tensor.dims)
    eight = counts['INT8'] + counts['UINT8']
    fraction = eight / max(1, eight + counts['FLOAT'] + counts['FLOAT16'])
    if eight == 0 or fraction < 0.8:
        raise ValueError(f'Insufficient 8-bit weights: {counts}')
    return {'initializer_elements_by_dtype': dict(counts), 'quantized_weight_fraction': fraction, 'size': path.stat().st_size, 'sha256': digest(path)}


def export_sparse(root, target):
    import torch
    from transformers import AutoModelForMaskedLM
    torch.set_num_threads(2)
    backbone = AutoModelForMaskedLM.from_pretrained(root, local_files_only=True, attn_implementation='eager').eval()
    class SparseModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, token_type_ids):
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
            return torch.max(torch.log1p(torch.relu(logits)) * attention_mask.unsqueeze(-1), dim=1).values
    wrapper = SparseModel(backbone).eval()
    _, data = encode(root, ['The Eiffel Tower is in Paris.', 'A banana.'])
    names = ['input_ids', 'attention_mask', 'token_type_ids']
    tensors = tuple(torch.from_numpy(data[n]) for n in names)
    with torch.no_grad():
        expected = wrapper(*tensors).numpy()
        torch.onnx.export(wrapper, tensors, str(target), input_names=names, output_names=['sparse_weights'], dynamic_axes={**{n: {0: 'batch', 1: 'sequence'} for n in names}, 'sparse_weights': {0: 'batch'}}, opset_version=17, dynamo=False)
    actual = session(target).run(None, data)[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-4)
    return {'method': 'Original masked-LM weights exported with max(log1p(relu(logits)) * attention_mask) pooling', 'fp32_export_max_absolute_error': float(np.max(np.abs(actual - expected)))}


def sparse_checks(root, target, reference):
    runtime, baseline = session(target), session(reference)
    records = []
    texts = ['Where is the Eiffel Tower?', 'The Eiffel Tower is in Paris, France.', 'Bananas are yellow fruit.']
    for batch in [texts[:1], texts, ['Paris. ' * 90, 'A short query.']]:
        _, inputs = encode(root, batch)
        actual = runtime.run(None, inputs)[0]
        expected = baseline.run(None, inputs)[0]
        if actual.shape != (len(batch), 30522) or not np.isfinite(actual).all() or (actual < 0).any():
            raise ValueError('Invalid SPLADE output')
        cosine = np.sum(actual * expected, axis=1) / np.maximum(np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1), 1e-12)
        overlaps = [len(set(np.argsort(a)[-20:]) & set(np.argsort(b)[-20:])) / 20 for a, b in zip(actual, expected)]
        record = {'batch': len(batch), 'sequence': inputs['input_ids'].shape[1], 'min_cosine': float(cosine.min()), 'top20_overlap': overlaps, 'nonzero_counts': np.count_nonzero(actual, axis=1).tolist()}
        print('SPARSE_ACCURACY', json.dumps(record), flush=True)
        if cosine.min() < 0.98 or min(overlaps) < 0.75:
            raise ValueError('SPLADE accuracy threshold failed')
        if len(batch) == 3:
            scores = actual[1:] @ actual[0]
            ref_scores = expected[1:] @ expected[0]
            if scores.argmax() != ref_scores.argmax():
                raise ValueError('SPLADE ranking changed')
            record.update(scores=scores.tolist(), reference_scores=ref_scores.tolist())
        records.append(record)
    return records


def build(spec, out):
    with tempfile.TemporaryDirectory(prefix=spec['key'] + '-') as temp:
        root = Path(temp) / spec['repository']
        sparse_model = spec['group'] == '06_sparse_splade'
        def wanted(name):
            if sparse_model:
                return name == 'pytorch_model.bin' or ('/' not in name and name.endswith(('.json', '.txt', '.md'))) or name == spec['pooling_config']
            return name in spec['onnx_sources'] or ('/' not in name and name.endswith(('.json', '.txt', '.md', '.model', '.spm')))
        upstream = download(spec['repository'], spec['revision'], root, wanted)
        (root / 'onnx').mkdir(exist_ok=True)
        if sparse_model:
            pooling = json.loads((root / spec['pooling_config']).read_text())
            if pooling['pooling_strategy'] != 'max' or pooling['activation_function'] != 'relu':
                raise ValueError('Unsupported upstream pooling')
            source = Path(temp) / 'fp32.onnx'
            conversion = export_sparse(root, source)
            (root / 'pytorch_model.bin').unlink()
            prepared = Path(temp) / 'embeddings_int8.onnx'
            quantize_embedding_tables(source, prepared)
            target = root / 'onnx/model_int8.onnx'
            rejected = []
            for mode in ['dynamic', 'weight_only']:
                if mode == 'dynamic':
                    quantize_dynamic(str(prepared), str(target), per_channel=True, weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul', 'Gemm'], extra_options={'MatMulConstBOnly': True})
                else:
                    quantize_matrix_weights(prepared, target)
                try:
                    checks = sparse_checks(root, target, source)
                    break
                except ValueError as error:
                    rejected.append({'mode': mode, 'error': str(error)})
                    if mode == 'weight_only':
                        raise
                    gc.collect()
            conversion.update(mode=mode, rejected_candidates=rejected, pooling_config=pooling)
            verification = {'tests': checks, 'graphs': {'onnx/model_int8.onnx': inspect(target)}}
        else:
            conversion = {'method': 'Upstream 8-bit ONNX graph bytes unchanged; normalized filenames', 'graph_mapping': dict(zip(spec['onnx_sources'], ['onnx/encoder_model_int8.onnx', 'onnx/decoder_model_int8.onnx']))}
            reference_root = Path(temp) / 'reference'
            if spec.get('convert_weight_only'):
                reference_root.mkdir()
                (reference_root / 'onnx').mkdir()
                for path in root.iterdir():
                    if path.is_file():
                        shutil.copyfile(path, reference_root / path.name)
                conversion = {'method': 'W8A32 from original FP32 ONNX: signed INT8 embedding rows and per-channel matrix weights with FP32 computation', 'mode': 'weight_only', 'reason': 'Both upstream 8-bit Flan variants produced degraded generation in smoke examples'}
            for source_name, target_name in zip(spec['onnx_sources'], ['onnx/encoder_model_int8.onnx', 'onnx/decoder_model_int8.onnx']):
                source = root / source_name
                if spec.get('convert_weight_only'):
                    shutil.copyfile(source, reference_root / target_name)
                    prepared = Path(temp) / 'prepared.onnx'
                    quantize_embedding_tables(source, prepared)
                    quantize_matrix_weights(prepared, root / target_name)
                    source.unlink()
                elif source_name != target_name:
                    source.rename(root / target_name)
            verification = {'graphs': {p.relative_to(root).as_posix(): inspect(p) for p in sorted((root / 'onnx').glob('*.onnx'))}, 'tests': []}
            for texts in [spec['prompts'][:1], spec['prompts']]:
                result = generate(root, texts)
                print('GENERATION', json.dumps(result), flush=True)
                if not all(t.strip() for t in result['texts']) or not all(result['ended_with_eos']):
                    raise ValueError('Generation did not produce nonempty completed text')
                record = {'inputs': texts, **result}
                if spec.get('convert_weight_only'):
                    baseline = generate(reference_root, texts)
                    print('REFERENCE_GENERATION', json.dumps(baseline), flush=True)
                    record['fp32_reference'] = baseline
                    record['exact_text_matches'] = [a == b for a, b in zip(result['texts'], baseline['texts'])]
                    if not all(record['exact_text_matches']):
                        raise ValueError('W8A32 generation differs from FP32 on smoke examples')
                verification['tests'].append(record)
        shutil.copyfile(Path(__file__).with_name('final_six_runtime.py'), root / 'onnx_runtime.py')
        usage = 'Offline CPU runtime: pip install numpy tokenizers onnxruntime sentencepiece\n'
        usage += 'python onnx_runtime.py . "Your text" --task ' + ('sparse' if sparse_model else 'generate') + '\n'
        usage += 'All graphs have predominantly 8-bit weights; tensor types and tests are recorded in STANDALONE-MANIFEST.json. Smoke tests are not retrieval/generation benchmarks or Android/Web validation.\n'
        if sparse_model:
            usage += 'Output sparse_weights is a dense [batch,30522] array with sparse nonnegative vocabulary weights. Convert nonzero entries to token IDs/weights for sparse dot-product retrieval. Pooling is already in the ONNX graph.\n'
            if spec['key'] == 'efficient-splade-VI-BT-large-query':
                usage += 'This package contains only the requested query encoder. Pair with its separately supplied document encoder for retrieval. Upstream router metadata is retained as provenance, not an instruction to load an absent document model.\n'
            if conversion['mode'] == 'weight_only':
                usage += 'W8A32: INT8 stored weights with FP32 matrix computation; runtime may expand weights. No full-INT8 speed or runtime-memory promise.\n'
        else:
            usage += 'Complete encoder plus non-cached decoder. No decoder-with-past or merged graph is required. The helper performs greedy autoregressive decoding up to 48 new tokens and reports EOS completion; it does not reproduce beam-search defaults or use a KV cache. Original SentencePiece files are included.\n'
        if not sparse_model and spec.get('convert_weight_only'):
            usage += 'W8A32: INT8 stored weights with FP32 matrix computation; runtime may expand weights. No full-INT8 speed or runtime-memory promise.\n'
        (root / 'OFFLINE-USAGE.txt').write_text(usage)
        files = [{'path': p.relative_to(root).as_posix(), 'size': p.stat().st_size, 'sha256': digest(p)} for p in sorted(root.rglob('*')) if p.is_file()]
        manifest = {**spec, 'upstream_files': upstream, 'conversion': conversion, 'verification': verification, 'files': files}
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
                raise ValueError('ZIP CRC failure')
            for item in files:
                if hashlib.sha256(archive.read(spec['repository'] + '/' + item['path'])).hexdigest() != item['sha256']:
                    raise ValueError('ZIP file hash mismatch')
        report = {**spec, 'name': name, 'size': archive_path.stat().st_size, 'sha256': digest(archive_path), 'onnx_size': sum(v['size'] for v in verification['graphs'].values()), 'conversion': conversion, 'verification': verification}
        (out / ('package-' + spec['key'] + '.json')).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('standalone-models'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    specs = json.loads(Path(__file__).with_name('final_six_int8_models.json').read_text())
    if args.finalize:
        packages = [json.loads((args.output / ('package-' + s['key'] + '.json')).read_text()) for s in specs]
        for spec, package in zip(specs, packages):
            if any(package.get(k) != v for k, v in spec.items()) or digest(args.output / package['name']) != package['sha256']:
                raise ValueError('Package identity/hash mismatch')
        (args.output / 'final-six-int8-models.lock.json').write_text(json.dumps({'schema': 1, 'packages': packages}, indent=2) + '\n')
    else:
        build(next(s for s in specs if s['key'] == args.model), args.output)


if __name__ == '__main__':
    main()

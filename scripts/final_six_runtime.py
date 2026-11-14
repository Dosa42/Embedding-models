import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


class MarianDecoder:
    def __init__(self, root, config):
        import sentencepiece as spm
        self.processor = spm.SentencePieceProcessor(model_file=str(root / 'target.spm'))
        self.tokens = {value: key for key, value in json.loads((root / 'vocab.json').read_text()).items()}
        self.special = {config['pad_token_id'], config['eos_token_id'], config['decoder_start_token_id']}

    def decode_batch(self, rows, skip_special_tokens=True):
        return [self.processor.decode_pieces([self.tokens[i] for i in row if not skip_special_tokens or i not in self.special]) for row in rows]


def encode(root, texts):
    root = Path(root)
    config = json.loads((root / 'config.json').read_text())
    if config.get('model_type') == 'marian':
        import sentencepiece as spm
        processor = spm.SentencePieceProcessor(model_file=str(root / 'source.spm'))
        vocab = json.loads((root / 'vocab.json').read_text())
        rows = [[vocab.get(piece, vocab['<unk>']) for piece in processor.encode_as_pieces(text)][:255] + [config['eos_token_id']] for text in texts]
        length = max(map(len, rows))
        data = {'input_ids': np.asarray([row + [config['pad_token_id']] * (length - len(row)) for row in rows], dtype=np.int64), 'attention_mask': np.asarray([[1] * len(row) + [0] * (length - len(row)) for row in rows], dtype=np.int64)}
        data['token_type_ids'] = np.zeros_like(data['input_ids'])
        return MarianDecoder(root, config), data
    tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(pad_id=config.get('pad_token_id', 0), pad_token=tokenizer.id_to_token(config.get('pad_token_id', 0)))
    rows = tokenizer.encode_batch(texts)
    return tokenizer, {name: np.asarray([getattr(row, attr) for row in rows], dtype=np.int64) for name, attr in [('input_ids', 'ids'), ('attention_mask', 'attention_mask'), ('token_type_ids', 'type_ids')]}


def sparse(root, texts):
    runtime = session(Path(root) / 'onnx/model_int8.onnx')
    tokenizer, inputs = encode(root, texts)
    values = runtime.run(None, {v.name: inputs[v.name] for v in runtime.get_inputs()})[0]
    return [{tokenizer.id_to_token(int(i)): float(row[i]) for i in np.flatnonzero(row)} for row in values]


def generate(root, texts, max_new_tokens=48):
    root = Path(root)
    config = json.loads((root / 'config.json').read_text())
    generation = json.loads((root / 'generation_config.json').read_text()) if (root / 'generation_config.json').exists() else {}
    settings = {**config, **generation}
    tokenizer, data = encode(root, texts)
    encoder = session(root / 'onnx/encoder_model_int8.onnx')
    decoder = session(root / 'onnx/decoder_model_int8.onnx')
    hidden = encoder.run(None, {v.name: data[v.name] for v in encoder.get_inputs()})[0]
    ids = np.full((len(texts), 1), settings['decoder_start_token_id'], dtype=np.int64)
    ended = np.zeros(len(texts), dtype=bool)
    eos = settings['eos_token_id']
    for step in range(max_new_tokens):
        available = {'input_ids': ids, 'decoder_input_ids': ids, 'encoder_hidden_states': hidden, 'encoder_attention_mask': data['attention_mask'], 'attention_mask': data['attention_mask'], 'decoder_attention_mask': np.ones_like(ids)}
        logits = decoder.run(['logits'], {v.name: available[v.name] for v in decoder.get_inputs()})[0][:, -1, :].copy()
        if not np.isfinite(logits).all():
            raise ValueError('Non-finite decoder logits')
        for token in settings.get('suppress_tokens', []) or []:
            logits[:, token] = -np.inf
        for token in (settings.get('begin_suppress_tokens', []) or []) if step == 0 else []:
            logits[:, token] = -np.inf
        next_ids = logits.argmax(axis=-1)
        if step == 0 and settings.get('forced_bos_token_id') is not None:
            next_ids[:] = settings['forced_bos_token_id']
        next_ids[ended] = settings.get('pad_token_id', eos)
        ids = np.concatenate([ids, next_ids[:, None]], axis=1)
        ended |= np.isin(next_ids, eos)
        if ended.all():
            break
    return {'texts': tokenizer.decode_batch(ids[:, 1:].tolist(), skip_special_tokens=True), 'token_ids': ids.tolist(), 'ended_with_eos': ended.tolist(), 'steps': ids.shape[1] - 1, 'strategy': 'greedy without KV cache; bounded by max_new_tokens'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('text', nargs='+')
    parser.add_argument('--task', choices=['sparse', 'generate'], required=True)
    args = parser.parse_args()
    result = sparse(args.root, args.text) if args.task == 'sparse' else generate(args.root, args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

import json
import re
import string
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def tokenizer_at(root):
    return Tokenizer.from_file(str(Path(root) / 'tokenizer.json'))


def text_inputs(runtime, tokenizer, texts):
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id('[PAD]') or tokenizer.token_to_id('<pad>') or 0)
    encodings = tokenizer.encode_batch(texts)
    data = {'input_ids': [e.ids for e in encodings], 'attention_mask': [e.attention_mask for e in encodings], 'token_type_ids': [e.type_ids for e in encodings]}
    return {v.name: np.asarray(data[v.name], dtype=np.int64) for v in runtime.get_inputs()}


def gliner_inputs(runtime, tokenizer, texts, labels, config):
    words = [re.findall(r'\w+(?:[-_]\w+)*|\S', text) for text in texts]
    prompt = [token for label in labels for token in [config['ent_token'], label]] + [config['sep_token']]
    tokenizer.no_truncation()
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id('[PAD]') or 0)
    encoded = tokenizer.encode_batch([prompt + row for row in words], is_pretokenized=True)
    masks = []
    for encoding in encoded:
        seen = set()
        row = []
        for word in encoding.word_ids:
            selected = word is not None and word >= len(prompt) and word not in seen
            row.append(word - len(prompt) + 1 if selected else 0)
            seen.add(word)
        masks.append(row)
    width = config['max_width']
    max_words = max(map(len, words))
    spans = np.asarray([(start, start + offset) for start in range(max_words) for offset in range(width)], dtype=np.int64)
    span_mask = np.asarray([[end < len(row) for start, end in spans] for row in words], dtype=np.bool_)
    span_idx = np.broadcast_to(spans, (len(words), len(spans), 2)).copy()
    span_idx[~span_mask] = 0
    data = {'input_ids': np.asarray([e.ids for e in encoded], dtype=np.int64), 'attention_mask': np.asarray([e.attention_mask for e in encoded], dtype=np.int64), 'words_mask': np.asarray(masks, dtype=np.int64), 'text_lengths': np.asarray([[len(row)] for row in words], dtype=np.int64), 'span_idx': span_idx, 'span_mask': span_mask}
    return {v.name: data[v.name] for v in runtime.get_inputs()}


def colbert_inputs(runtime, tokenizer, texts, query=False, max_length=None):
    tokenizer.no_padding()
    tokenizer.no_truncation()
    length = max_length or (32 if query else 180)
    marker = tokenizer.token_to_id('[unused0]' if query else '[unused1]')
    if marker is None:
        raise ValueError('Missing ColBERT query/document marker')
    rows = []
    masks = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False).ids[:length - 3]
        ids = [tokenizer.token_to_id('[CLS]'), marker] + ids + [tokenizer.token_to_id('[SEP]')]
        rows.append(ids)
        masks.append([1] * len(ids))
    padding_length = length if query else max(map(len, rows))
    pad = tokenizer.token_to_id('[MASK]' if query else '[PAD]')
    data = {'input_ids': np.asarray([row + [pad] * (padding_length - len(row)) for row in rows], dtype=np.int64), 'attention_mask': np.asarray([row + [0] * (padding_length - len(row)) for row in masks], dtype=np.int64), 'token_type_ids': np.zeros((len(rows), padding_length), dtype=np.int64)}
    return {v.name: data[v.name] for v in runtime.get_inputs()}


def maxsim(query_vectors, document_vectors, document_ids, tokenizer):
    punctuation = {tokenizer.encode(symbol, add_special_tokens=False).ids[0] for symbol in string.punctuation if tokenizer.encode(symbol, add_special_tokens=False).ids}
    excluded = punctuation | {tokenizer.token_to_id('[PAD]')}
    active = np.asarray([int(token) not in excluded for token in document_ids])
    if not active.any():
        raise ValueError('No document tokens remain')
    query_vectors = query_vectors / np.maximum(np.linalg.norm(query_vectors, axis=-1, keepdims=True), 1e-12)
    document_vectors = document_vectors / np.maximum(np.linalg.norm(document_vectors, axis=-1, keepdims=True), 1e-12)
    return float((query_vectors @ document_vectors[active].T).max(axis=-1).sum())

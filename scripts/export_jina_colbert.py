import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from build_six_int8_models import session
from next_nine_runtime import colbert_inputs, tokenizer_at


CODE_REPO = 'jinaai/jina-bert-implementation'
CODE_REVISION = 'f3ec4cf7de7e561007f27c9efc7148b0bd713f81'


class ColbertExport(torch.nn.Module):
    def __init__(self, backbone, projection):
        super().__init__()
        self.bert = backbone
        self.linear = torch.nn.Linear(projection.shape[1], projection.shape[0], bias=False)
        self.linear.weight.data.copy_(projection)
        self.register_buffer('slopes', backbone.encoder.alibi[:, :, 0:1, 1:2].clone())

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        distances = torch.abs(positions[:, None] - positions[None, :]).to(self.slopes.dtype)
        self.bert.encoder.alibi = self.slopes * distances[None, None]
        self.bert.encoder._current_alibi_size = 8192
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=torch.zeros_like(input_ids), return_dict=False)[0]
        return torch.nn.functional.normalize(self.linear(hidden), p=2, dim=-1)


def export(root, target):
    torch.set_num_threads(2)
    config_class = get_class_from_dynamic_module('configuration_bert.JinaBertConfig', CODE_REPO, revision=CODE_REVISION)
    model_class = get_class_from_dynamic_module('modeling_bert.JinaBertModel', CODE_REPO, revision=CODE_REVISION)
    data = json.loads((root / 'config.json').read_text())
    data['_name_or_path'] = str(root)
    data['max_position_embeddings'] = 64
    data['attn_implementation'] = 'eager'
    config = config_class(**data)
    backbone = model_class(config).eval()
    weights = load_file(root / 'model.safetensors')
    state = {k.removeprefix('bert.'): v for k, v in weights.items() if k.startswith('bert.')}
    backbone.load_state_dict(state, strict=True)
    model = ColbertExport(backbone, weights['linear.weight']).eval()
    del weights, state
    ids = torch.tensor([[101, 1, 2054, 2003, 1996, 102], [101, 2, 3000, 2003, 1996, 102]], dtype=torch.long)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        original = torch.nn.functional.normalize(model.linear(backbone(input_ids=ids, attention_mask=mask, return_dict=False)[0]), p=2, dim=-1)
        np.testing.assert_allclose(model(ids, mask).numpy(), original.numpy(), rtol=1e-5, atol=1e-6)
        torch.onnx.export(model, (ids, mask), str(target), input_names=['input_ids', 'attention_mask'], output_names=['contextual'], dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'}, 'attention_mask': {0: 'batch_size', 1: 'sequence_length'}, 'contextual': {0: 'batch_size', 1: 'sequence_length'}}, opset_version=17, do_constant_folding=True, dynamo=False)
    runtime = session(target)
    tokenizer = tokenizer_at(root)
    checks = []
    for texts, limit in [(['Where is Paris?'], 32), (['The Eiffel Tower is in Paris.', 'A longer document about yellow fruit.'], 64), (['word ' * 254], 256)]:
        feed = colbert_inputs(runtime, tokenizer, texts, max_length=limit)
        with torch.no_grad():
            expected = model(torch.from_numpy(feed['input_ids']), torch.from_numpy(feed['attention_mask'])).numpy()
        actual = runtime.run(None, feed)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=2e-5)
        checks.append({'shape': list(actual.shape), 'max_absolute_error': float(np.max(np.abs(actual - expected)))})
    return {'code_repository': CODE_REPO, 'code_revision': CODE_REVISION, 'export': 'original JinaBERT ALiBi/GEGLU backbone and trained 128-dimensional linear projection, token L2 normalization; dynamically constructed equivalent ALiBi', 'pytorch_fp32_vs_onnx_fp32': checks}

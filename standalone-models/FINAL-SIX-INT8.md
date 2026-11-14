# Final six SPLADE and seq2seq INT8 ONNX packages

These packages complete the requested 21-model set together with [the first six](SIX-INT8.md) and [the next nine](NEXT-NINE-INT8.md).

| Group | Requested model | Download |
|---|---|---|
| 06_sparse_splade | `naver/splade-cocondenser-ensembledistil` | [ZIP](https://media.githubusercontent.com/media/Dosa42/Embedding-models/main/standalone-models/naver-splade-cocondenser-ensembledistil-int8-standalone.zip) |
| 06_sparse_splade | `prithivida/Splade_PP_en_v1` | [ZIP](https://media.githubusercontent.com/media/Dosa42/Embedding-models/main/standalone-models/prithivida-Splade_PP_en_v1-int8-standalone.zip) |
| 06_sparse_splade | `naver/efficient-splade-VI-BT-large-query` | [ZIP](https://media.githubusercontent.com/media/Dosa42/Embedding-models/main/standalone-models/naver-efficient-splade-VI-BT-large-query-int8-standalone.zip) |
| 07_seq2seq | `Xenova/flan-t5-small` | [ZIP](https://media.githubusercontent.com/media/Dosa42/Embedding-models/main/standalone-models/Xenova-flan-t5-small-int8-standalone.zip) |
| 07_seq2seq | `Xenova/t5-small` | [ZIP](https://media.githubusercontent.com/media/Dosa42/Embedding-models/main/standalone-models/Xenova-t5-small-int8-standalone.zip) |
| 07_seq2seq | `Xenova/opus-mt-nl-en` | [ZIP](https://media.githubusercontent.com/media/Dosa42/Embedding-models/main/standalone-models/Xenova-opus-mt-nl-en-int8-standalone.zip) |

## Package contents

Each ZIP includes the pinned original tokenizer/configuration files, ONNX graph(s), `onnx_runtime.py`, `OFFLINE-USAGE.txt`, and a per-file SHA256 manifest. The [lock file](final-six-int8-models.lock.json) records exact sources, archive hashes, tensor types, conversion decisions and CPU tests.

SPLADE exports the original masked-language model and its configured max pooling into `onnx/model_int8.onnx`. The output is `max(log1p(relu(logits)) * attention_mask)` over the token axis: a dense array of sparse vocabulary weights shaped `[batch,30522]`. It is not a pooled dense sentence embedding. Efficient SPLADE contains the requested **query encoder only**; its matching document encoder is a separate model. Upstream router files are preserved as provenance.

SPLADE uses signed INT8 embedding tables and dynamic per-channel INT8 matrix quantization. If that fails the fixed accuracy thresholds, the workflow tests W8A32 weight-only quantization and records the selected mode explicitly. W8A32 stores INT8 weights but computes matrices in FP32 and may expand weights at runtime. Check the lock file for the actual mode of each completed package.

Seq2seq packages include `onnx/encoder_model_int8.onnx` and `onnx/decoder_model_int8.onnx`. T5-small and Marian graph bytes are unchanged from upstream 8-bit variants; filenames are normalized. T5-small and Marian use the smaller upstream `decoder_model_quantized.onnx` variants. Flan-T5 is converted from the original FP32 ONNX pair to W8A32 because both tested upstream 8-bit variants degraded generation. Its example generations must exactly match the FP32 reference. W8A32 stores INT8 weights but computes in FP32; runtime memory and speed savings are not guaranteed. INT8/UINT8 tensor counts are checked rather than inferred from filenames. Original SentencePiece files are included. Marian uses source/target SentencePiece and the original vocabulary directly because its upstream tokenizer JSON has a null precompiled normalizer.

The included generator performs greedy autoregressive decoding with the full decoder, without a KV cache. This is a complete usable encoder/decoder pair; merged and decoder-with-past graphs are not needed for this runtime. It does not reproduce beam-search defaults and trades cache acceleration for a smaller complete package.

## Offline use

From the extracted model directory:

```sh
pip install numpy tokenizers onnxruntime sentencepiece
python onnx_runtime.py . "The Eiffel Tower is in Paris." --task sparse
```

For a seq2seq model, use `--task generate` with a suitable prompt; for Marian Dutch-to-English, pass Dutch text.

## Validation scope

GitHub Actions verifies source hashes, ONNX structure, predominantly 8-bit weights, actual CPU inference, ZIP CRC and every packaged file SHA256. SPLADE is compared against its FP32 export with minimum cosine 0.98, top-20 overlap 0.75 and consistent example ranking, using multiple batch/sequence sizes. The FP32 export is compared to PyTorch. Seq2seq generates nonempty text through EOS for individual and padded two-example batches. These are smoke tests, not retrieval/translation benchmarks or Android/Web execution tests.

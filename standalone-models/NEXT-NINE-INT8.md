# NER, zero-shot and ColBERT INT8 ONNX packages

[Build next nine INT8 ONNX models](../.github/workflows/build-next-nine-int8-models.yml) prepares and validates each model in an independent job, then publishes all nine complete ZIPs through Git LFS. Sources are pinned in [next_nine_int8_models.json](../scripts/next_nine_int8_models.json). The generated [next-nine-int8-models.lock.json](next-nine-int8-models.lock.json) records sizes, hashes, conversion details and actual CPU inference checks. Workflow artifacts remain available for 30 days.

| Group | Model | ZIP |
| --- | --- | --- |
| 03_token_classifiers | onnx-community/gliner_small-v2.1 | [INT8](onnx-community-gliner_small-v2.1-int8-standalone.zip) |
| 03_token_classifiers | Xenova/bert-base-NER | [INT8](Xenova-bert-base-NER-int8-standalone.zip) |
| 03_token_classifiers | Xenova/bert-base-NER-uncased | [INT8](Xenova-bert-base-NER-uncased-int8-standalone.zip) |
| 04_zero_shot | Xenova/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary | [INT8](Xenova-DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary-int8-standalone.zip) |
| 04_zero_shot | Xenova/distilbert-base-uncased-mnli | [INT8](Xenova-distilbert-base-uncased-mnli-int8-standalone.zip) |
| 04_zero_shot | MoritzLaurer/mDeBERTa-v3-base-mnli-xnli | [INT8](MoritzLaurer-mDeBERTa-v3-base-mnli-xnli-int8-standalone.zip) |
| 05_colbert | libryo-ai/answerai-colbert-small-v1 | [INT8](libryo-ai-answerai-colbert-small-v1-int8-standalone.zip) |
| 05_colbert | colbert-ir/colbertv2.0 | [INT8](colbert-ir-colbertv2.0-int8-standalone.zip) |
| 05_colbert | jinaai/jina-colbert-v1-en | [INT8](jinaai-jina-colbert-v1-en-int8-standalone.zip) |

The inaccessible requested `Xenova/mDeBERTa-v3-base-mnli-xnli` is sourced from its original publisher, `MoritzLaurer`, using the existing `onnx/model_quantized.onnx`. Its bytes are preserved under the common filename `onnx/model_int8.onnx`. The other five models in the first two groups use their existing `model_int8.onnx` unchanged. Quantized models can combine signed INT8 matrix weights, unsigned UINT8 tables, and floating-point operations; manifests report actual tensor counts rather than assuming all tensors have one dtype.

The two existing ColBERT FP32 graphs are quantized with symmetric per-row INT8 embedding tables and ONNX Runtime dynamic per-channel QInt8 matrix operations. Full-range weights are checked first; reduced-range INT8 is tried if the accuracy check rejects that candidate. Manifests identify the accepted setting and rejected candidates. Libryo's repository supplies only ONNX and tokenizer files; its package adds explicitly labelled minimal interface configuration, without claiming it is an original Transformers configuration. The 96-dimensional AnswerAI projection and 128-dimensional ColBERTv2 projection remain intact. Jina is exported from its original safetensors using pinned JinaBERT implementation code, preserving GEGLU, ALiBi and the trained 128-dimensional projection. An equivalent dynamically computed ALiBi avoids embedding a huge fixed attention-bias tensor in ONNX.

Each ZIP includes its tokenizer, configuration, original model documentation, `OFFLINE-USAGE.txt`, `onnx_runtime_inputs.py`, and a manifest with provenance and checksums. The helper provides tested GLiNER prompt/span inputs, ordinary classification inputs, and ColBERT query/document markers, query MASK augmentation and MaxSim scoring. GLiNER emits span logits and requires entity span decoding; NER emits token logits; zero-shot models emit either two or three class logits according to their label configuration. ColBERT produces multiple vectors per text and must not be treated as a pooled sentence encoder.

Checks include ONNX structural validation, predominantly 8-bit weight storage, finite CPU outputs at batch sizes 1 and 3, correct task dimensions, ColBERT FP32/INT8 token-vector similarity and MaxSim ranking consistency on a small example. Jina also compares PyTorch FP32 with exported ONNX FP32 at multiple lengths, including 256 tokens. These are conversion/inference checks, not retrieval or entity-recognition benchmarks. Linux CPU verification does not establish Android/Web compatibility or a test of Jina's full 8192-token limit.

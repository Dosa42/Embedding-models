# Six INT8 ONNX models

The [Build six INT8 ONNX models workflow](../.github/workflows/build-six-int8-models.yml) downloads pinned upstream revisions, converts Potion, completes Ettin's scoring head, executes all six models, and stores complete ZIP files here with Git LFS. Each ZIP includes tokenizer/configuration files, upstream model documentation, usage notes and a checksum/validation manifest. The workflow also provides a `six-int8-onnx-models` artifact for 30 days.

| Group | Model | Complete ZIP |
| --- | --- | --- |
| 01_bi_encoders | minishlab/potion-base-8M | [INT8](minishlab-potion-base-8M-int8-standalone.zip) |
| 01_bi_encoders | Xenova/bge-small-en-v1.5 | [INT8](Xenova-bge-small-en-v1.5-int8-standalone.zip) |
| 01_bi_encoders | Xenova/paraphrase-multilingual-MiniLM-L12-v2 | [INT8](Xenova-paraphrase-multilingual-MiniLM-L12-v2-int8-standalone.zip) |
| 02_cross_encoders | Xenova/ms-marco-TinyBERT-L-2-v2 | [INT8](Xenova-ms-marco-TinyBERT-L-2-v2-int8-standalone.zip) |
| 02_cross_encoders | cross-encoder/ettin-reranker-17m-v1 | [INT8](cross-encoder-ettin-reranker-17m-v1-int8-standalone.zip) |
| 02_cross_encoders | Xenova/nli-deberta-v3-xsmall | [INT8](Xenova-nli-deberta-v3-xsmall-int8-standalone.zip) |

The requested `onnx-community/ettin-reranker-17m-v1` repository was inaccessible. The package uses the same model's publisher repository, `cross-encoder/ettin-reranker-17m-v1`. Its published ARM64 INT8 ONNX produces token embeddings only; the packaged `onnx/model_int8.onnx` also includes the original CLS pooling and FP32 scoring head, producing one relevance score per query/document pair. Original head weights and configurations are retained.

Potion uses signed INT8 quantization per embedding row. Its graph gathers quantized rows and their scales before dequantization, preserving the original pooling and normalization. It accepts flattened `input_ids` plus sentence-start `offsets`, rather than the padded transformer input convention. Tokenize without special tokens.

The four Xenova ONNX files are the publisher's signed `model_int8.onnx`, without reconversion. NLI DeBERTa produces three NLI class logits, not a trained relevance score. Follow `config.json` for label order.

INT8 describes quantized weights; floating-point scales, normalization, activations and outputs remain. Validation checks ONNX structure, substantial signed INT8 tensors, finite CPU outputs at batch sizes 1 and 3, Potion similarity against its original FP32 model, and Ettin scoring-head equivalence against a NumPy implementation. This is a conversion/inference check, not a retrieval or NLI benchmark. Android/Web execution is not claimed by the Linux CPU check.

The workflow writes [six-int8-models.lock.json](six-int8-models.lock.json) with exact upstream revisions, ZIP and ONNX sizes, SHA-256 values, and test results. Clone/pull with Git LFS, or download the complete files from GitHub. Extract a ZIP into a models directory; its paths are `<publisher>/<model>/`. Every final graph is named `onnx/model_int8.onnx`.

# Embedding models

## Xenova BGE-M3 and E5-v2 — models included in this repository

All six **q8 and FP16 model packages are stored as ordinary Git files** in
[standalone-models/](standalone-models/). A normal clone includes their bytes;
no release download or Git LFS is required. Each ZIP is divided into parts of
at most 48 MiB to fit GitHub's per-file limit.

```sh
git clone https://github.com/Dosa42/Embedding-models.git
cd Embedding-models
python3 scripts/repo_model_packages.py restore --extract
```

This works offline after cloning and needs only Python 3.10+. It reconstructs
six complete ZIPs in `dist/`, verifies their SHA-256 checksums and extracts the
model files into `models/Xenova/<model>/`. Add `--dtype q8` or `--dtype fp16`
to select a precision, and `--model bge-m3`, `--model e5-base-v2`, or
`--model e5-large-v2` to select one model. Existing files with different contents
are preserved and reported as errors.

| Model | q8 weight bytes | FP16 weight bytes | Pooling |
| --- | ---: | ---: | --- |
| Xenova/bge-m3 | 569,694,530 | 1,133,992,936 | CLS + L2 |
| Xenova/e5-base-v2 | 110,083,279 | 218,108,177 | Masked mean + L2 |
| Xenova/e5-large-v2 | 336,983,104 | 668,828,712 | Masked mean + L2 |

Each package includes the ONNX weights, tokenizer, configuration and upstream
model card. Use `dtype: 'q8'` for `model_quantized.onnx` or `dtype: 'fp16'` for
`model_fp16.onnx`. Install a compatible inference runtime separately. E5-v2
retrieval uses `query: ` / `passage: ` prefixes, English text and at most 512 tokens.
BGE-M3 packages target dense embeddings. These files do not change the Android
app's bundled model selection.

The [upstream model lock](xenova-models.lock.json) pins original model files;
the [release lock](repo-model-release.lock.json) pins the ZIP bytes copied into
Git. Each package directory also records checksums for every part. See
[standalone-models/README.md](standalone-models/README.md) for offline verification.

The [repository import workflow](https://github.com/Dosa42/Embedding-models/actions/workflows/import-models-to-repo.yml)
imports the pinned release and commits the model bytes package by package. It
can be rerun safely to finish an interrupted import. The original
[release downloads](https://github.com/Dosa42/Embedding-models/releases/tag/xenova-onnx-1-1)
remain available as an additional download option.

For a fresh download from the pinned Hugging Face versions, the original
`python3 scripts/fetch_xenova_models.py --dtype all` command also remains available.
See `README_LOCAL.txt` inside each ZIP for local loading settings. Integrity is
verified; inference is not tested by the import workflow.

## Multilingual E5 small (int8 ONNX)

`multilingual-e5-small-int8/` contains the tokenizer and configuration files for the [Xenova/multilingual-e5-small](https://huggingface.co/Xenova/multilingual-e5-small) ONNX model, revision `761b726dd34fb83930e26aab4e9ac3899aa1fa78`. The base model is [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small), licensed under MIT.

The 118,054,593-byte ONNX file is stored as `onnx/model_int8.onnx.zip` because the original file exceeds GitHub's 100 MB file limit. Extract the archive into the same `onnx/` directory before loading the model:

```sh
unzip multilingual-e5-small-int8/onnx/model_int8.onnx.zip -d multilingual-e5-small-int8/onnx
```

The extracted `model_int8.onnx` SHA-256 is `4d24e2bc01a447951524466ef533e52944bf48509e6552810bcee1a2711cb02c`.

For retrieval, prefix queries with `query: ` and indexed text with `passage: `. The model produces 384-dimensional embeddings with attention-mask mean pooling and L2 normalization; its maximum sequence length is 512 tokens.

## All MiniLM L6 v2

`all-MiniLM-L6-v2/` contains the [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, licensed under Apache 2.0. It includes the original Safetensors weights, ONNX weights, tokenizer, and Sentence Transformers pooling configuration.

The model produces 384-dimensional embeddings. Apply attention-mask mean pooling and L2 normalization; inputs longer than 256 word pieces are truncated. The ONNX file SHA-256 is `6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452` and the Safetensors file SHA-256 is `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.

## Android app

The [android-app](android-app/) directory packages both models, the official Chroma Android library, embedded Python, FastAPI, and LangChain Core. Its foreground service owns text embedding and retrieval, and the interface uses the real FastAPI endpoints. Projects retain original file and line citations.

## Official framework sources

The full source snapshots below are included alongside the two model directories. Their upstream licenses, tests, examples, documentation, and package sources are preserved. They contain source at the pinned commits, without Git history or installed dependencies.

| Directory | Official upstream | Commit | Files |
| --- | --- | --- | ---: |
| [langchain](langchain/) | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | `771d11ba650b6fc43306d8f303e574814e433804` | 3,296 |
| [fastapi](fastapi/) | [fastapi/fastapi](https://github.com/fastapi/fastapi) | `5add7a458d629b07311d48261f696b5ac2bb9a5f` | 3,241 |

[upstream-sources.lock.json](upstream-sources.lock.json) records the upstream commits, Git tree hashes, archive checksums, file counts, and source sizes. The Android build packages FastAPI and LangChain Core from these pinned source snapshots. See [the Android runtime and API documentation](android-app/README.md).


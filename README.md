# Embedding models

## Xenova BGE-M3 and E5-v2 — standalone q8 / FP16

The [Fetch Xenova ONNX models workflow](https://github.com/Dosa42/Embedding-models/actions/workflows/fetch-xenova-models.yml)
downloads the three models directly on a GitHub-hosted runner and publishes **six
complete ZIP packages** under [Releases](https://github.com/Dosa42/Embedding-models/releases). Each package contains one
ONNX precision, its tokenizer, configuration, upstream model card and checksums.
Large weights are release assets, not Git blobs; no Git LFS or split ZIPs are needed.

| Model | q8 weight bytes | FP16 weight bytes | Pooling |
| --- | ---: | ---: | --- |
| Xenova/bge-m3 | 569,694,530 | 1,133,992,936 | CLS + L2 |
| Xenova/e5-base-v2 | 110,083,279 | 218,108,177 | Masked mean + L2 |
| Xenova/e5-large-v2 | 336,983,104 | 668,828,712 | Masked mean + L2 |

The [lock file](xenova-models.lock.json) pins the exact upstream commits, file sizes
and SHA-256 values. Every downloaded file is verified before packaging; an invalid
download fails the job. Each ZIP includes its own file manifest, and the release
includes checksums for the ZIPs. The workflow publishes only after all six uploads
succeed. A failed upload may leave an unpublished draft for inspection.

The workflow starts when its downloader, tests, lock file or workflow changes on
`main`. To run it again, open **Actions → Fetch Xenova ONNX models → Run workflow**
on `main`. It uses the built-in GitHub token; no Hugging Face token or extra secret
is required for these public models. A new release is created for each run.

### Clone and download from GitHub

With Git and the GitHub CLI installed:

```sh
git clone https://github.com/Dosa42/Embedding-models.git
cd Embedding-models
gh release download --repo Dosa42/Embedding-models --pattern 'Xenova-*-q8-standalone.zip' --dir downloads
```

Change the pattern to `Xenova-*-fp16-standalone.zip` for FP16 or
`Xenova-*-standalone.zip` for both. To select an older model release explicitly,
pass its `xenova-onnx-…` tag to `gh release download`. Extract ZIPs into `models/`;
they create `models/Xenova/<model>/config.json` and the matching `onnx/` file.
A Git clone alone does not download these release assets.

### Fetch directly from the pinned upstream versions

Python 3.10+ is sufficient; the downloader has no third-party dependencies:

```sh
python3 scripts/fetch_xenova_models.py                 # all three, q8 by default
python3 scripts/fetch_xenova_models.py --dtype all     # all six variants
python3 scripts/fetch_xenova_models.py --model e5-base-v2 --dtype fp16
```

Files are downloaded into `models/`, packages into `dist/`. Valid cached files are
reused. Add `--no-package` to download just the model directory, or `--verify-only`
to verify existing files without using the network. Choose `--destination` and
`--output` to use other directories.

Use `dtype: 'q8'` for `model_quantized.onnx`, or `dtype: 'fp16'` for
`model_fp16.onnx`. A compatible inference runtime is required separately. E5-v2
retrieval uses `query: ` / `passage: ` prefixes, English text and at most 512 tokens.
BGE-M3 packages here target dense embeddings. The workflow verifies file and ZIP
integrity; it does not run inference or modify the Android app's bundled models.
See `README_LOCAL.txt` inside each ZIP for local loading settings.

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


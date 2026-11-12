# Embedding models

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

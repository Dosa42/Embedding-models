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

## Android app starter

The [android-app](android-app/) directory is a buildable Kotlin Android project with a Gradle wrapper and a Samsung SDK library slot. Its placeholder screen does not package either model yet.

# Complete standalone model ZIPs

Each model/precision is one complete ZIP file, tracked with Git LFS. There are
no parts to assemble. The Git history stores an LFS pointer; GitHub LFS holds
the complete ZIP bytes and Git LFS downloads them to the same path when cloning.

| Model | q8 | FP16 |
| --- | --- | --- |
| BGE-M3 | [ZIP](Xenova-bge-m3-q8-standalone.zip) | [ZIP](Xenova-bge-m3-fp16-standalone.zip) |
| E5-base-v2 | [ZIP](Xenova-e5-base-v2-q8-standalone.zip) | [ZIP](Xenova-e5-base-v2-fp16-standalone.zip) |
| E5-large-v2 | [ZIP](Xenova-e5-large-v2-q8-standalone.zip) | [ZIP](Xenova-e5-large-v2-fp16-standalone.zip) |

Install Git LFS, then:

```sh
git lfs install
git clone --depth 1 https://github.com/Dosa42/Embedding-models.git
cd Embedding-models
```

For an existing clone:

```sh
git lfs install
git pull
git lfs pull
```

Extract any ZIP normally. Each includes the original ONNX model, tokenizer,
configuration, upstream model card and SHA-256 checksums. Extract into `models/`
to create `models/Xenova/<model>/`. Both precisions may share that directory.
A compatible inference runtime is required separately.

To verify all six downloaded ZIPs with Python 3.10+:

```sh
python3 scripts/verify_repo_models.py
```

Small text files starting with `version https://git-lfs.github.com/spec/v1` are
LFS pointers, not the model bytes: run `git lfs pull` to download the full files.
GitHub's source ZIP may contain pointers depending on repository archive settings;
use a Git LFS clone or the individual file's download button for complete models.

[repo-model-release.lock.json](../repo-model-release.lock.json) records exact
ZIP sizes and SHA-256 values. The import workflow verifies the complete ZIPs,
uploads all LFS objects, then replaces the multipart paths on `main` in one commit.
Old multipart data remains in Git history; `--depth 1` avoids that old history.

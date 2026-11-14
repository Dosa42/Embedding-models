# Standalone model packages stored in Git

This directory holds the actual bytes of all six Xenova q8/FP16 model packages.
Each package has its own directory containing `manifest.json` and ordered
`.zip.part001`, `.zip.part002`, … files of at most 48 MiB each. These are ordinary
Git blobs, not Git LFS pointers or links to release downloads.

After cloning, restore and extract all models with Python 3.10+:

```sh
python3 scripts/repo_model_packages.py restore --extract
```

For only the q8 versions:

```sh
python3 scripts/repo_model_packages.py restore --dtype q8 --extract
```

For one model, add `--model bge-m3`, `--model e5-base-v2`, or `--model e5-large-v2`.
The script verifies every part, the complete ZIP and each extracted model file.
ZIPs are restored into `dist/`; model files into `models/Xenova/<model>/`.
Use `--output` and `--destination` to choose other folders. Omit `--extract`
to restore only the ZIPs. No network access or third-party Python package is needed.

The committed [release lock](../repo-model-release.lock.json) records the original
release and exact ZIP checksums. [The import workflow](../.github/workflows/import-models-to-repo.yml)
imports those bytes and pushes each package separately. An interrupted import can
be rerun: it verifies and skips already committed packages. Incomplete imports
are visible as missing model directories and make `verify`/`restore` fail.

```sh
python3 scripts/repo_model_packages.py verify
```

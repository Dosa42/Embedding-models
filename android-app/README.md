# Source Atlas for Android

This APK embeds multilingual E5 small and MiniLM L6 v2, runs a localhost RAG retrieval API in an Android foreground service, and stores project collections with the official Chroma Android library. It works without a network connection after installation. The Chroma store is private to the app at `app_chroma/`; Chroma Android creates `chroma_metadata.db` and index directories there. The app does not create a separate SQLite output database.

The interface imports selected text files, a source folder with relative paths, existing 384 dimensional vectors in JSONL, or the bundled app source corpus. It chunks source by line, embeds passages with the selected model, and returns citations such as `src/tools/BashTool/bashSecurity.ts:1459`. E5 uses `passage:` for source and `query:` for search. Each project has separate Chroma collections for E5 and MiniLM.

## Build

Use JDK 17 or newer, Android SDK platform 36, and Android SDK Build Tools 36.0.0:

```sh
./gradlew :app:assembleDebug
```

The build fetches pinned Chroma Android 0.0.1 and Transformers.js 3.8.1 runtime files, verifies their SHA256 checksums, expands the E5 ONNX archive, and packages both model files. This build step needs network access; the installed app does not. The APK is at `app/build/outputs/apk/debug/app-debug.apk`.

In Termux, its `LD_PRELOAD` setting conflicts with AGP's AAPT2 binary. Use:

```sh
ANDROID_HOME="$HOME/android-sdk" env -u LD_PRELOAD ./gradlew :app:assembleDebug
```

The Chroma Android beta AAR currently contains ARM64 native libraries. Its 0.0.1 libraries are not aligned for 16 KB memory pages, so current Samsung firmware may show a compatibility warning. The app was exercised successfully on the connected Samsung device despite that warning.

## Local RAG API

The API listens on `127.0.0.1:8765` after the app is opened. The foreground service owns the listener and Chroma client, so the API remains available when the screen closes or the app moves to the background. Its ongoing notification opens the app. Android may restart the sticky service after process reclamation; force stopping the app or stopping it through Android's running services controls ends it until the app is opened again. It accepts JSON and returns JSON.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Chroma engine and store path |
| GET | `/api/projects` | List projects |
| POST | `/api/projects` | Create project with `{"name":"My project"}` |
| POST | `/api/ingest` | Save a batch of passages and 384 dimensional vectors |
| POST | `/api/search` | Search one project and model, returning paths, lines, text, and distances |
| POST | `/api/rag` | Return a citation formatted context block and source hits |
| GET | `/api/corpus/files` | List bundled knowledge files |
| GET | `/api/folder/files` | List the folder selected in the app |

`/api/ingest` accepts `{"project":"id","model":"e5","items":[{"path":"src/file.ts","line":1,"text":"source passage","vector":[384 numbers]}]}`. `/api/search` and `/api/rag` accept `{"project":"id","model":"e5","vector":[384 numbers],"limit":8}`. A LangChain or FastAPI client can call these localhost endpoints with embeddings from the same selected model. The RAG endpoint assembles retrieval context; it does not generate an LLM answer.

## Bundled corpus

The build also packages this app's Kotlin and interface source as an example knowledge corpus. Choose **Use bundled corpus**, then **Embed & save to Chroma** to index it. Choose **Choose source folder** to index any other source tree while keeping its relative paths. Files over 4 MB are not accepted by the current importer.

## Samsung SDKs

The app builds for Samsung Galaxy devices with the standard Android SDK. A Samsung feature SDK can be added to `app/libs/` when a specific device feature needs it.

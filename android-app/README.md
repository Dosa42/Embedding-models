# Source Atlas for Android

Source Atlas packages multilingual E5 small, MiniLM L6 v2, CPython, FastAPI, LangChain Core, and the official Chroma Android library in one APK. Models and Python dependencies are bundled for offline operation. Source passages retain their project, original file path, and starting line.

## Runtime

`FastApiService` owns the Chroma client, an embedding WebView, and the embedded Python interpreter. Python runs the real FastAPI application with Uvicorn. LangChain's embeddings and retriever interfaces connect its retrieval pipeline to the bundled models and native Chroma collections.

The app screen is served by FastAPI and calls the same endpoints as other clients. Embedding inference belongs to the foreground service, so plain-text requests can execute while the screen is closed. Opening Source Atlas starts the service. It does not start at boot. The service returns `START_STICKY`, which requests Android recreation after process reclamation. Device verification confirmed both models still answer plain-text queries after the screen activity is destroyed. A forced SIGKILL on the tested Samsung device did not recreate the process within 55 seconds; reopening the app restores the service. Immediate recovery after a process kill is not guaranteed. Force stopping the app stops it until it is opened again.

Chroma persists at the original private `app_chroma/` path with `chroma_metadata.db` and its index directories. Existing projects and collections are preserved. Each project has separate E5 and MiniLM collections. E5 inputs use `query: ` and `passage: ` prefixes; both models use mean pooling and L2 normalization.

## API

Uvicorn listens on `0.0.0.0:8765` with its WebSocket transport enabled. Use `http://127.0.0.1:8765` on the phone or `http://PHONE_LAN_IP:8765` from the local network. Offline Swagger is at `/docs`; the generated schema is at `/openapi.json`. No application WebSocket route is registered.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | Service state, embedding readiness, actual framework versions |
| GET | `/models` | Bundled model IDs and dimensions |
| GET / POST | `/projects` | List projects or create one with `{"name":"Sources"}` |
| POST | `/documents` | Ingest source text or existing vectors |
| POST | `/search` | Retrieve passages using a text query or vector |
| POST | `/rag` | Return retrieval context and citations |
| POST | `/rag/invoke` | Invoke the LangChain retrieval pipeline with an `input` body |
| POST | `/v1/embeddings` | Return OpenAI-shaped floating-point embeddings |
| GET | `/corpus/files` | List bundled example sources |
| GET | `/corpus/file/{id}` | Read a bundled source |
| GET | `/folder/files` | List the folder selected in the app |
| GET | `/folder/file/{id}` | Read a selected folder source |

Plain-text retrieval:

```json
{
  "project": "PROJECT_ID",
  "model": "e5",
  "query": "Where are shell commands checked before execution?"
}
```

Send this body to `/search` or `/rag`. `/rag/invoke` accepts `{"input":{...}}` with the same fields. Each hit contains `path`, `line`, `citation`, `text`, and `distance`. The RAG response contains `context` and `citations`. No answer generation model is configured.

Source ingestion:

```json
{
  "project": "PROJECT_ID",
  "model": "e5",
  "documents": [
    {
      "path": "src/example.ts",
      "line": 1,
      "text": "export function example() { return true }"
    }
  ],
  "chunk_lines": 80,
  "overlap_lines": 12
}
```

An optional `vector` on a document accepts exactly 384 numbers and preserves that supplied passage. Search accepts either `query` or `vector`, never both. Models are `e5` or `minilm`. Omit `limit` or set it to `null` to return every passage in the selected collection ranked by similarity. An optional positive `limit` requests a specific result count without an application-imposed upper ceiling. FastAPI validates requests and documents them in OpenAPI.

Embedding requests use `{"model":"e5","input":"some text","input_type":"query"}`. Input may also be a nonempty array of texts, with no application-imposed array length cap. The response has `object`, `model`, and `data` entries with `index` and `embedding`. This endpoint supports float embeddings; it does not implement token-array inputs, base64 output, or usage counts.

## Request sizes and lifecycle

Source files have no application-imposed byte limit. Document requests and embedding input arrays have no application-imposed item limit. The interface sends the selected passages together; the API embeds and stores the supplied passages without fixed eight-item batches or a native 32-item ceiling.

Embedding initialization and queued embedding operations wait until completion or a reported failure, with no fixed 60-second or 180-second deadline. The CPU wake lock is acquired for an embedding operation and released in its cleanup, without a timed expiry. Shutdown waits for the Uvicorn thread to finish before closing Chroma, without a 10-second join deadline.

## Build

Use JDK 17+, Android SDK 36, Build Tools 36.0.0, and build Python 3.14:

```sh
./gradlew :app:assembleDebug
```

The framework packages are built from the pinned source snapshots at `../fastapi` and `../langchain/libs/core`. Chaquopy 17.2 embeds Python 3.14.8 for Android arm64. Python dependency pins are in `python-requirements.txt`; prepared Android wheels and their source checksums are in `python-wheels/`.

The build downloads checksum-pinned Chroma Android 0.0.1, Transformers.js 3.8.1, and Swagger UI 5.34.1 assets. Both model directories are staged into the APK. Build time requires network access for uncached dependencies; installed inference and retrieval use local files.

In Termux:

```sh
ANDROID_HOME="$HOME/android-sdk" env -u LD_PRELOAD ./gradlew :app:assembleDebug --no-daemon
```

Rebuild native Python dependencies on Android arm64 with Python 3.14, Rust, Clang, and patchelf:

```sh
python scripts/build_android_wheels.py
```

Verify the prepared wheels:

```sh
python scripts/build_android_wheels.py --verify
```

The script verifies upstream archive hashes, builds Android wheels, checks native dependencies and segment alignment, removes build-host RUNPATH entries, and records artifact checksums. These build tools are not runtime requirements.

The debug APK is at `app/build/outputs/apk/debug/app-debug.apk`. The official Chroma Android 0.0.1 beta AAR supports ARM64; its existing native libraries are not aligned for 16 KB memory pages. The current connected Samsung uses 4 KB pages.

## Bundled example

Choose **Use bundled example**, then **Index selected source** to index the app's Kotlin, Python, and interface code. The source folder picker preserves relative paths for other projects. The interface accepts text/source files and existing vectors; it does not limit knowledge input to JSON documents.

## Samsung SDKs

The app targets Samsung Galaxy devices through the standard Android SDK. Add a Samsung feature SDK to `app/libs/` when a specific device feature requires it.

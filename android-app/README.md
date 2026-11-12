# Source Atlas for Android

This APK embeds multilingual E5 small and MiniLM L6 v2, stores project collections with the official Chroma Android library. It works without a network connection after installation. The Chroma store is private to the app at `app_chroma/`; Chroma Android creates `chroma_metadata.db` and index directories there. The app does not create a separate SQLite output database.

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

## Embedded Chroma access

The app uses the official Chroma Android library directly from the Android process. The WebView loads packaged assets through Android resource interception at `https://appassets.androidplatform.net/` and reaches the database through an internal JavaScript bridge. No HTTP server, listening port, RAG endpoint, or foreground server service is included.

Existing Chroma collections and project files remain at their original app-private paths. Opening this version preserves those stored projects. The two embedding models still run locally in the app interface.

## Bundled corpus

The build also packages this app's Kotlin and interface source as an example knowledge corpus. Choose **Use bundled example**, then **Index selected source** to index it. Choose **Choose source folder** to index any other source tree while keeping its relative paths. Files over 4 MB are not accepted by the current importer.

## Samsung SDKs

The app builds for Samsung Galaxy devices with the standard Android SDK. A Samsung feature SDK can be added to `app/libs/` when a specific device feature needs it.

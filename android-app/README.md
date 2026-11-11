# Android app starter

This is a Kotlin Android application with Gradle Kotlin DSL. It currently opens a single placeholder screen. The app identifier is `com.ayrvokan.embeddingmodels`.

## Build

Use JDK 17 or newer, Android SDK platform 36, and Android SDK Build Tools 36.0.0. From this directory, run:

```sh
./gradlew :app:assembleDebug
```

The debug APK is generated at `app/build/outputs/apk/debug/app-debug.apk`. `local.properties` and build outputs are ignored by Git.

In Termux, its `LD_PRELOAD` setting conflicts with AGP's AAPT2 binary. Use:

```sh
ANDROID_HOME="$HOME/android-sdk" env -u LD_PRELOAD ./gradlew :app:assembleDebug
```

## Samsung SDKs

The project builds for Samsung Galaxy devices using the standard Android SDK. Samsung distributes feature SDKs separately. The `app/libs/` directory is wired for local Samsung `.aar` and `.jar` libraries; add a particular SDK and its required manifest entries when the app's functions are chosen. No Samsung feature or permission is selected in this starter.

## Model files

The repository's E5 and MiniLM models remain in their own top-level directories. They are not packaged into this placeholder APK. Model loading and packaging can be added after the app's behavior is defined.

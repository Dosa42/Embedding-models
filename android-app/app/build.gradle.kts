import java.io.File
import java.net.URI
import java.security.MessageDigest

plugins {
    id("com.android.application")
    id("com.chaquo.python")
}

val runtimeAssets = layout.buildDirectory.dir("generated/runtime-assets/vendor")
val fetchRuntimeAssets = tasks.register("fetchRuntimeAssets") {
    doLast {
        val downloads = listOf(
            Triple("https://unpkg.com/swagger-ui-dist@5.34.1/swagger-ui-bundle.js", runtimeAssets.get().file("swagger-ui-bundle.js").asFile, "36eeb4d5640b9c036b1d0f854499af1cceebbdb260b790a03a18be5e8488fbf5"),
            Triple("https://unpkg.com/swagger-ui-dist@5.34.1/swagger-ui.css", runtimeAssets.get().file("swagger-ui.css").asFile, "c3cfe8248c4b0b89673600fb829924eafbac6de4bc4b2b06f5f5243483074d0a"),
            Triple("https://github.com/chroma-core/chroma-android/releases/download/v0.0.1/chroma-android-release-0.0.1.aar", project.file("libs/chroma-android-release-0.0.1.aar"), "dcb4d2b4a6dac1f9784c629a2ee85ebb53467d2c38cfa4f73ebbaf254933fb2e"),
            Triple("https://unpkg.com/@huggingface/transformers@3.8.1/dist/transformers.min.js", runtimeAssets.get().file("transformers.min.js").asFile, "aa5002b70e789798da263f5f99c62bd3e8fcd0c119258a493c40c180648365fa"),
            Triple("https://unpkg.com/@huggingface/transformers@3.8.1/dist/ort-wasm-simd-threaded.jsep.mjs", runtimeAssets.get().file("ort-wasm-simd-threaded.jsep.mjs").asFile, "08fb86ec433c78bfb032c5d84a68b8e8e5a8d81268fa39e24314179a5767a5b9"),
            Triple("https://unpkg.com/@huggingface/transformers@3.8.1/dist/ort-wasm-simd-threaded.jsep.wasm", runtimeAssets.get().file("ort-wasm-simd-threaded.jsep.wasm").asFile, "c46655e8a94afc45338d4cb2b840475f88e5012d524509916e505079c00bfa39")
        )
        fun digest(file: File): String = MessageDigest.getInstance("SHA-256").digest(file.readBytes()).joinToString("") { "%02x".format(it) }
        for ((url, file, expected) in downloads) {
            if (file.exists() && digest(file) == expected) continue
            file.parentFile.mkdirs()
            val temporary = File(file.parentFile, "${file.name}.download")
            URI(url).toURL().openStream().use { source -> temporary.outputStream().use { source.copyTo(it) } }
            require(digest(temporary) == expected) { "Downloaded runtime checksum mismatch: ${file.name}" }
            temporary.renameTo(file)
        }
    }
}

val stageModelAssets by tasks.registering(Sync::class) {
    dependsOn(fetchRuntimeAssets)
    from(zipTree(rootProject.file("../multilingual-e5-small-int8/onnx/model_int8.onnx.zip"))) {
        into("models/Xenova/multilingual-e5-small/onnx")
    }
    from(rootProject.file("../multilingual-e5-small-int8")) {
        include("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        into("models/Xenova/multilingual-e5-small")
    }
    from(rootProject.file("../all-MiniLM-L6-v2")) {
        include("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "onnx/model.onnx", "config_sentence_transformers.json", "modules.json", "sentence_bert_config.json", "1_Pooling/config.json")
        into("models/sentence-transformers/all-MiniLM-L6-v2")
    }
    from(project.file("src/main/java")) {
        include("**/*.kt")
        into("corpus/src/main/java")
    }
    from(project.file("src/main/assets")) {
        include("app.js", "index.html", "app.css", "embedding.js", "embedding.html", "settings.js")
        into("corpus/src/main/assets")
    }
    from(project.file("src/main/python")) {
        include("**/*.py")
        into("corpus/src/main/python")
    }
    from(runtimeAssets) {
        into("vendor")
    }
    into(layout.buildDirectory.dir("generated/model-assets"))
}

android {
    namespace = "com.ayrvokan.embeddingmodels"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.ayrvokan.embeddingmodels"
        minSdk = 26
        targetSdk = 36
        versionCode = 7
        versionName = "0.6.0"
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets.getByName("main").assets.srcDir(layout.buildDirectory.dir("generated/model-assets").get().asFile)
}

tasks.named("preBuild") {
    dependsOn(stageModelAssets)
}

dependencies {
    implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.jar", "*.aar"))))
    implementation("androidx.documentfile:documentfile:1.0.1")
}

chaquopy {
    defaultConfig {
        version = "3.14"
        pip {
            options("--find-links", rootProject.file("python-wheels").absolutePath)
            install("-r", rootProject.file("python-requirements.txt").absolutePath)
            install(rootProject.file("../fastapi").absolutePath)
            install(rootProject.file("../langchain/libs/core").absolutePath)
        }
    }
}

package com.ayrvokan.embeddingmodels

import android.content.Context
import org.json.JSONObject
import java.io.File

class RagRuntime(private val context: Context, val chroma: ChromaBridge, val embeddings: EmbeddingEngine) {
    @Volatile var state = "starting"
    @Volatile var failure: String? = null

    fun listProjects(): String = chroma.listProjects()
    fun createProject(name: String): String = chroma.createProject(name)
    fun addDocuments(project: String, model: String, itemsJson: String): String = chroma.addDocuments(project, model, itemsJson)
    fun query(project: String, model: String, vectorJson: String, limit: Int?): String = chroma.query(project, model, vectorJson, limit)
    fun listCorpusFiles(): String = chroma.listCorpusFiles()
    fun readCorpusFile(id: String): String = chroma.readCorpusFile(id)
    fun listFolderFiles(): String = chroma.listFolderFiles()
    fun readFolderFile(id: String): String = chroma.readFolderFile(id)
    fun embedTexts(model: String, textsJson: String, kind: String): String = embeddings.embedTexts(model, textsJson, kind)
    fun assetText(path: String): String = context.assets.open(path).bufferedReader().use { it.readText() }

    @Synchronized
    fun assetFile(path: String): String {
        require(path.isNotEmpty() && !path.startsWith("/") && path.split('/').none { it == ".." }) { "Invalid asset path" }
        val version = File(context.applicationInfo.sourceDir).lastModified()
        val file = File(context.cacheDir, "rag-assets-$version/$path")
        if (!file.exists()) {
            file.parentFile?.mkdirs()
            val temporary = File(file.parentFile, file.name + ".pending")
            context.assets.open(path).use { source -> temporary.outputStream().use { source.copyTo(it) } }
            check(temporary.renameTo(file)) { "Could not prepare asset" }
        }
        return file.absolutePath
    }

    fun runtimeStatus(): String = JSONObject().put("service", "source-atlas-rag").put("database", "chroma-android").put("state", state).put("error", failure ?: JSONObject.NULL).put("embedding", embeddings.state()).toString()
    fun serverReady() { state = "running" }
    fun serverStopped() { state = "stopped" }
}

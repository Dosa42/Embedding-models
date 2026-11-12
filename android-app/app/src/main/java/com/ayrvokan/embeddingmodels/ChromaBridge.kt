package com.ayrvokan.embeddingmodels

import android.content.Context
import android.net.Uri
import android.webkit.JavascriptInterface
import androidx.documentfile.provider.DocumentFile
import com.trychroma.android.Document
import com.trychroma.android.PersistentClient
import org.json.JSONArray
import org.json.JSONObject
import java.security.MessageDigest
import java.util.UUID

class ChromaBridge(private val context: Context) {
    private val chroma = PersistentClient(context.getDir("chroma", Context.MODE_PRIVATE).absolutePath)
    private val lock = Any()
    private val projectsFile = context.filesDir.resolve("projects.json")
    private val projects = if (projectsFile.exists()) JSONArray(projectsFile.readText()) else JSONArray()
    private val folderFiles = LinkedHashMap<String, Pair<String, Uri>>()
    private val corpusFiles = ArrayList<String>()

    init {
        fun visit(path: String) {
            for (name in context.assets.list(path).orEmpty()) {
                val child = "$path/$name"
                if (context.assets.list(child).orEmpty().isNotEmpty()) visit(child) else corpusFiles.add(child)
            }
        }
        visit("corpus")
    }

    private fun result(action: () -> JSONObject): String = try {
        synchronized(lock) { action().toString() }
    } catch (error: Exception) {
        JSONObject().put("error", error.message ?: "Chroma operation failed").toString()
    }

    @JavascriptInterface
    fun listProjects(): String = result { JSONObject().put("projects", JSONArray(projects.toString())) }

    @JavascriptInterface
    fun createProject(name: String): String = result {
        val trimmed = name.trim()
        require(trimmed.isNotEmpty()) { "Project name required" }
        val project = JSONObject().put("id", UUID.randomUUID().toString().replace("-", "")).put("name", trimmed)
        projects.put(project)
        projectsFile.writeText(projects.toString())
        project
    }

    @JavascriptInterface
    fun addDocuments(project: String, model: String, itemsJson: String): String = result {
        val items = JSONArray(itemsJson)
        require(items.length() > 0) { "Send at least one passage" }
        val docs = ArrayList<Document>()
        for (i in 0 until items.length()) {
            val item = items.getJSONObject(i)
            val vector = floats(item.getJSONArray("vector"))
            require(vector.size == 384) { "Expected 384 embedding values" }
            val path = item.getString("path")
            val line = item.getInt("line")
            val text = item.getString("text")
            val metadata = mapOf<String, Any>("path" to path, "line" to line, "model" to model)
            val hash = MessageDigest.getInstance("SHA-256").digest("$project:$model:$path:$line:$text".toByteArray()).joinToString("") { "%02x".format(it) }
            docs.add(Document(hash, vector, text, metadata, path))
        }
        collection(project, model).add(docs)
        JSONObject().put("saved", docs.size)
    }

    @JavascriptInterface
    fun query(project: String, model: String, vectorJson: String, limit: Int?): String = result {
        val vector = floats(JSONArray(vectorJson))
        require(vector.size == 384) { "Expected 384 embedding values" }
        val hits = JSONArray()
        require(limit == null || limit > 0) { "Result limit must be positive" }
        val sources = collection(project, model)
        val count = sources.count()
        val requested = limit?.coerceAtMost(count) ?: count
        val matches = if (requested > 0) sources.query(arrayOf(vector), requested) else emptyList()
        for (hit in matches) {
            val path = hit.metadata?.get("path")?.toString() ?: hit.uri ?: "source"
            val line = (hit.metadata?.get("line") as? Number)?.toInt() ?: 1
            hits.put(JSONObject().put("path", path).put("line", line).put("citation", "$path:$line").put("text", hit.data).put("distance", hit.distance))
        }
        JSONObject().put("hits", hits)
    }

    @JavascriptInterface
    fun listCorpusFiles(): String = result {
        val files = JSONArray()
        corpusFiles.forEachIndexed { index, name -> files.put(JSONObject().put("id", index).put("path", name.removePrefix("corpus/"))) }
        JSONObject().put("files", files)
    }

    @JavascriptInterface
    fun readCorpusFile(id: String): String = result {
        val path = corpusFiles[id.toInt()]
        JSONObject().put("path", path.removePrefix("corpus/")).put("content", context.assets.open(path).bufferedReader().use { it.readText() })
    }

    @JavascriptInterface
    fun listFolderFiles(): String = result {
        if (folderFiles.isEmpty()) {
            context.getSharedPreferences("rag_service", Context.MODE_PRIVATE).getString("selected_tree", null)?.let { selectTree(Uri.parse(it)) }
        }
        val files = JSONArray()
        folderFiles.forEach { (id, entry) -> files.put(JSONObject().put("id", id).put("path", entry.first)) }
        JSONObject().put("files", files)
    }

    @JavascriptInterface
    fun readFolderFile(id: String): String = result {
        val entry = folderFiles[id] ?: throw IllegalArgumentException("File not found")
        val content = context.contentResolver.openInputStream(entry.second)?.use { source ->
            source.bufferedReader(Charsets.UTF_8).readText()
        } ?: throw IllegalArgumentException("File unavailable")
        JSONObject().put("path", entry.first).put("content", content)
    }

    fun selectTree(uri: Uri) {
        val root = DocumentFile.fromTreeUri(context, uri) ?: throw IllegalArgumentException("Folder unavailable")
        val found = LinkedHashMap<String, Pair<String, Uri>>()
        fun visit(directory: DocumentFile, prefix: String) {
            for (child in directory.listFiles()) {
                val name = child.name ?: continue
                if (child.isDirectory) {
                    if (name !in setOf(".git", "node_modules", "build", "dist")) visit(child, "$prefix$name/")
                } else if (child.isFile) found[UUID.randomUUID().toString()] = "$prefix$name" to child.uri
            }
        }
        visit(root, "")
        synchronized(lock) {
            folderFiles.clear()
            folderFiles.putAll(found)
        }
        context.getSharedPreferences("rag_service", Context.MODE_PRIVATE).edit().putString("selected_tree", uri.toString()).apply()
    }

    private fun collection(project: String, model: String): com.trychroma.android.Collection {
        require((0 until projects.length()).any { projects.getJSONObject(it).getString("id") == project }) { "Project not found" }
        require(model == "e5" || model == "minilm") { "Unknown model" }
        val name = "project_${project}_$model"
        return try { chroma.getCollection(name) } catch (_: Exception) { chroma.createCollection(name) }
    }

    private fun floats(array: JSONArray): FloatArray = FloatArray(array.length()) { array.getDouble(it).toFloat() }

    fun close() {
        synchronized(lock) { chroma.close() }
    }
}

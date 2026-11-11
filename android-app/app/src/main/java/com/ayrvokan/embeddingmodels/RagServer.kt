package com.ayrvokan.embeddingmodels

import android.content.Context
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import com.trychroma.android.Document
import com.trychroma.android.PersistentClient
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.net.URLDecoder
import java.security.MessageDigest
import java.util.UUID
import kotlin.concurrent.thread

class RagServer(private val context: Context) {
    private val chroma = PersistentClient(context.getDir("chroma", Context.MODE_PRIVATE).absolutePath)
    private val lock = Any()
    private val projectsFile = context.filesDir.resolve("projects.json")
    private val projects = if (projectsFile.exists()) JSONArray(projectsFile.readText()) else JSONArray()
    private val folderFiles = LinkedHashMap<String, Pair<String, Uri>>()
    private val corpusFiles = ArrayList<String>()
    private var socket: ServerSocket? = null

    fun start() {
        fun visit(path: String) {
            for (name in context.assets.list(path).orEmpty()) {
                val child = "$path/$name"
                if (context.assets.list(child).orEmpty().isNotEmpty()) visit(child) else corpusFiles.add(child)
            }
        }
        visit("corpus")
        socket = ServerSocket(8765, 16, InetAddress.getByName("127.0.0.1"))
        thread(name = "rag-server") {
            while (socket?.isClosed == false) {
                try {
                    val client = socket!!.accept()
                    thread(name = "rag-client") { handle(client) }
                } catch (_: Exception) {
                    break
                }
            }
        }
    }

    fun close() {
        socket?.close()
        synchronized(lock) { chroma.close() }
    }

    fun selectTree(uri: Uri) {
        val root = DocumentFile.fromTreeUri(context, uri) ?: return
        val found = LinkedHashMap<String, Pair<String, Uri>>()
        fun visit(directory: DocumentFile, prefix: String) {
            for (child in directory.listFiles()) {
                val name = child.name ?: continue
                if (child.isDirectory) {
                    if (name !in setOf(".git", "node_modules", "build", "dist")) visit(child, "$prefix$name/")
                } else if (child.isFile) {
                    found[UUID.randomUUID().toString()] = "$prefix$name" to child.uri
                }
            }
        }
        visit(root, "")
        synchronized(lock) {
            folderFiles.clear()
            folderFiles.putAll(found)
        }
    }

    private fun handle(client: Socket) {
        client.use { connection ->
            connection.soTimeout = 60000
            try {
                val input = BufferedInputStream(connection.getInputStream())
                val output = BufferedOutputStream(connection.getOutputStream())
                val requestLine = readLine(input) ?: return
                val parts = requestLine.split(' ')
                if (parts.size < 2) return
                val headers = mutableMapOf<String, String>()
                while (true) {
                    val line = readLine(input) ?: return
                    if (line.isEmpty()) break
                    val cut = line.indexOf(':')
                    if (cut > 0) headers[line.substring(0, cut).lowercase()] = line.substring(cut + 1).trim()
                }
                val length = headers["content-length"]?.toIntOrNull() ?: 0
                if (length > 8_000_000) {
                    respond(output, 413, JSONObject().put("error", "Request too large").toString())
                    return
                }
                val bodyBytes = ByteArray(length)
                var offset = 0
                while (offset < length) {
                    val count = input.read(bodyBytes, offset, length - offset)
                    if (count < 0) throw IllegalArgumentException("Incomplete request")
                    offset += count
                }
                val body = if (length > 0) JSONObject(String(bodyBytes, Charsets.UTF_8)) else JSONObject()
                val path = URLDecoder.decode(parts[1].substringBefore('?'), "UTF-8")
                if (path.startsWith("/api/")) {
                    val result = route(parts[0], path, body)
                    respond(output, 200, result.toString())
                } else {
                    asset(output, path)
                }
            } catch (error: Exception) {
                try {
                    respond(BufferedOutputStream(connection.getOutputStream()), 400, JSONObject().put("error", error.message ?: "Request failed").toString())
                } catch (_: Exception) {
                }
            }
        }
    }

    private fun route(method: String, path: String, body: JSONObject): JSONObject {
        if (method == "GET" && path == "/api/health") return JSONObject().put("status", "ok").put("database", "chroma-android").put("store", chroma.persistPath)
        if (method == "GET" && path == "/api/projects") return JSONObject().put("projects", synchronized(lock) { JSONArray(projects.toString()) })
        if (method == "GET" && path == "/api/folder/files") {
            val files = JSONArray()
            synchronized(lock) { folderFiles.forEach { (id, entry) -> files.put(JSONObject().put("id", id).put("path", entry.first)) } }
            return JSONObject().put("files", files)
        }
        if (method == "GET" && path == "/api/corpus/files") {
            val files = JSONArray()
            corpusFiles.forEachIndexed { index, name -> files.put(JSONObject().put("id", index).put("path", name.removePrefix("corpus/"))) }
            return JSONObject().put("files", files)
        }
        if (method == "GET" && path.startsWith("/api/corpus/file/")) {
            val index = path.substringAfterLast('/').toInt()
            val name = corpusFiles[index]
            return JSONObject().put("path", name.removePrefix("corpus/")).put("content", context.assets.open(name).bufferedReader().use { it.readText() })
        }
        if (method == "GET" && path.startsWith("/api/folder/file/")) {
            val entry = synchronized(lock) { folderFiles[path.substringAfterLast('/')] } ?: throw IllegalArgumentException("File not found")
            val content = context.contentResolver.openInputStream(entry.second)?.use { source ->
                val bytes = source.readBytes()
                require(bytes.size <= 4_000_000) { "File exceeds 4 MB" }
                String(bytes, Charsets.UTF_8)
            } ?: throw IllegalArgumentException("File unavailable")
            return JSONObject().put("path", entry.first).put("content", content)
        }
        if (method == "POST" && path == "/api/projects") {
            val name = body.getString("name").trim()
            require(name.isNotEmpty()) { "Project name required" }
            val project = JSONObject().put("id", UUID.randomUUID().toString().replace("-", "")).put("name", name)
            synchronized(lock) {
                projects.put(project)
                projectsFile.writeText(projects.toString())
            }
            return project
        }
        if (method == "POST" && path == "/api/ingest") {
            val project = body.getString("project")
            val model = body.getString("model")
            val items = body.getJSONArray("items")
            require(items.length() in 1..32) { "Send 1 to 32 chunks per request" }
            val docs = ArrayList<Document>()
            for (i in 0 until items.length()) {
                val item = items.getJSONObject(i)
                val vector = floats(item.getJSONArray("vector"))
                require(vector.size == 384) { "Expected 384 embedding values" }
                val pathValue = item.getString("path")
                val line = item.getInt("line")
                val metadata = mapOf<String, Any>("path" to pathValue, "line" to line, "model" to model)
                val documentText = item.getString("text")
                val hash = MessageDigest.getInstance("SHA-256").digest("$project:$model:$pathValue:$line:$documentText".toByteArray()).joinToString("") { "%02x".format(it) }
                docs.add(Document(hash, vector, documentText, metadata, pathValue))
            }
            synchronized(lock) { collection(project, model).add(docs) }
            return JSONObject().put("saved", docs.size)
        }
        if (method == "POST" && path == "/api/search") {
            val project = body.getString("project")
            val model = body.getString("model")
            val vector = floats(body.getJSONArray("vector"))
            require(vector.size == 384) { "Expected 384 embedding values" }
            val limit = body.optInt("limit", 8).coerceIn(1, 30)
            val results = synchronized(lock) { collection(project, model).query(arrayOf(vector), limit) }
            val hits = JSONArray()
            for (result in results) {
                val pathValue = result.metadata?.get("path")?.toString() ?: result.uri ?: "source"
                val line = (result.metadata?.get("line") as? Number)?.toInt() ?: 1
                hits.put(JSONObject().put("path", pathValue).put("line", line).put("citation", "$pathValue:$line").put("text", result.data).put("distance", result.distance))
            }
            return JSONObject().put("hits", hits)
        }
        if (method == "POST" && path == "/api/rag") {
            val search = route("POST", "/api/search", body)
            val hits = search.getJSONArray("hits")
            val contextText = buildString {
                for (i in 0 until hits.length()) {
                    val hit = hits.getJSONObject(i)
                    append("[${hit.getString("citation")}]\n${hit.getString("text")}\n\n")
                }
            }
            return JSONObject().put("context", contextText.trim()).put("citations", hits)
        }
        throw IllegalArgumentException("Unknown API endpoint")
    }

    private fun collection(project: String, model: String): com.trychroma.android.Collection {
        require((0 until projects.length()).any { projects.getJSONObject(it).getString("id") == project }) { "Project not found" }
        require(model == "e5" || model == "minilm") { "Unknown model" }
        val name = "project_${project}_$model"
        return try { chroma.getCollection(name) } catch (_: Exception) { chroma.createCollection(name) }
    }

    private fun floats(array: JSONArray): FloatArray = FloatArray(array.length()) { array.getDouble(it).toFloat() }

    private fun asset(output: BufferedOutputStream, requestPath: String) {
        val path = if (requestPath == "/") "index.html" else requestPath.removePrefix("/")
        require(!path.split('/').contains("..")) { "Invalid path" }
        val mime = when {
            path.endsWith(".html") -> "text/html; charset=utf-8"
            path.endsWith(".js") || path.endsWith(".mjs") -> "text/javascript; charset=utf-8"
            path.endsWith(".css") -> "text/css; charset=utf-8"
            path.endsWith(".json") -> "application/json"
            path.endsWith(".wasm") -> "application/wasm"
            else -> "application/octet-stream"
        }
        output.write("HTTP/1.1 200 OK\r\nContent-Type: $mime\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n".toByteArray())
        context.assets.open(path).use { source ->
            val buffer = ByteArray(65536)
            while (true) {
                val count = source.read(buffer)
                if (count < 0) break
                output.write("${count.toString(16)}\r\n".toByteArray())
                output.write(buffer, 0, count)
                output.write("\r\n".toByteArray())
            }
        }
        output.write("0\r\n\r\n".toByteArray())
        output.flush()
    }

    private fun respond(output: BufferedOutputStream, status: Int, body: String) {
        val bytes = body.toByteArray(Charsets.UTF_8)
        output.write("HTTP/1.1 $status ${if (status == 200) "OK" else "Error"}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: ${bytes.size}\r\nConnection: close\r\n\r\n".toByteArray())
        output.write(bytes)
        output.flush()
    }

    private fun readLine(input: BufferedInputStream): String? {
        val bytes = ArrayList<Byte>()
        while (bytes.size < 8192) {
            val next = input.read()
            if (next < 0) return null
            if (next == 10) return String(bytes.toByteArray(), Charsets.ISO_8859_1).trimEnd('\r')
            bytes.add(next.toByte())
        }
        throw IllegalArgumentException("Header too long")
    }
}

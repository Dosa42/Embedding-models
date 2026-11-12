package com.ayrvokan.embeddingmodels

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.webkit.JavascriptInterface
import android.webkit.RenderProcessGoneDetail
import android.webkit.WebView
import android.webkit.WebViewClient
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import org.json.JSONObject
import java.util.UUID
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch

class EmbeddingEngine(private val context: Context) {
    private val main = Handler(Looper.getMainLooper())
    private val initializedLatch = CountDownLatch(1)
    private val pending = ConcurrentHashMap<String, CompletableFuture<String>>()
    private val web = WebView(context)
    @Volatile private var failure: String? = null
    @Volatile private var initialized = false
    @Volatile private var closed = false

    init {
        check(Looper.myLooper() == Looper.getMainLooper())
        web.settings.javaScriptEnabled = true
        web.settings.domStorageEnabled = true
        web.setRendererPriorityPolicy(WebView.RENDERER_PRIORITY_IMPORTANT, false)
        val resources = AssetResources(context)
        web.webViewClient = object : WebViewClient() {
            override fun shouldInterceptRequest(view: WebView?, request: WebResourceRequest?): WebResourceResponse? = resources.shouldInterceptRequest(view, request)
            override fun onRenderProcessGone(view: WebView?, detail: RenderProcessGoneDetail?): Boolean {
                fail("Embedding renderer stopped")
                return true
            }
        }
        web.addJavascriptInterface(object {
            @JavascriptInterface
            fun ready() {
                initialized = true
                initializedLatch.countDown()
            }
            @JavascriptInterface
            fun failed(message: String) {
                fail(message)
            }
            @JavascriptInterface
            fun complete(id: String, value: String) {
                pending.remove(id)?.complete(value)
            }
        }, "EmbeddingCallbacks")
        web.loadUrl("https://appassets.androidplatform.net/embedding.html")
    }

    private fun fail(message: String) {
        failure = message
        initializedLatch.countDown()
        pending.values.forEach { it.completeExceptionally(IllegalStateException(message)) }
        pending.clear()
    }

    fun state(): JSONObject = JSONObject().put("ready", initialized && failure == null && !closed).put("error", failure ?: JSONObject.NULL)

    fun embedTexts(model: String, textsJson: String, kind: String): String {
        check(Looper.myLooper() != Looper.getMainLooper()) { "Embedding must run off the UI thread" }
        check(!closed) { "Embedding runtime is closed" }
        require(model == "e5" || model == "minilm") { "Unknown embedding model" }
        require(kind == "query" || kind == "passage") { "Unknown embedding input type" }
        initializedLatch.await()
        failure?.let { throw IllegalStateException(it) }
        val id = UUID.randomUUID().toString()
        val future = CompletableFuture<String>()
        pending[id] = future
        val power = context.getSystemService(PowerManager::class.java)
        val wake = power.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "SourceAtlas:Embedding")
        wake.acquire()
        try {
            main.post {
                if (closed) future.completeExceptionally(IllegalStateException("Embedding runtime is closed"))
                else web.evaluateJavascript("window.embedRequest(${JSONObject.quote(id)},${JSONObject.quote(model)},${JSONObject.quote(textsJson)},${JSONObject.quote(kind)})", null)
            }
            val result = future.get()
            if (result.startsWith("{")) throw IllegalStateException(JSONObject(result).optString("error", "Embedding failed"))
            return result
        } finally {
            pending.remove(id)
            if (wake.isHeld) wake.release()
        }
    }

    fun close() {
        closed = true
        fail("Embedding runtime closed")
        main.post { web.destroy() }
    }
}

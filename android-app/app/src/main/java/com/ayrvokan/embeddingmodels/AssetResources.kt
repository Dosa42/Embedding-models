package com.ayrvokan.embeddingmodels

import android.content.Context
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient

class AssetResources(private val context: Context) : WebViewClient() {
    override fun shouldInterceptRequest(view: WebView?, request: WebResourceRequest?): WebResourceResponse? {
        val uri = request?.url ?: return null
        if (uri.host != "appassets.androidplatform.net") return super.shouldInterceptRequest(view, request)
        val path = if (uri.path == "/") "index.html" else uri.path.orEmpty().removePrefix("/")
        val mime = when {
            path.endsWith(".html") -> "text/html"
            path.endsWith(".js") || path.endsWith(".mjs") -> "text/javascript"
            path.endsWith(".css") -> "text/css"
            path.endsWith(".json") -> "application/json"
            path.endsWith(".wasm") -> "application/wasm"
            else -> "application/octet-stream"
        }
        return try {
            WebResourceResponse(mime, if (mime.startsWith("text/") || mime == "application/json") "UTF-8" else null, context.assets.open(path))
        } catch (_: Exception) {
            WebResourceResponse("text/plain", "UTF-8", 404, "Not Found", emptyMap(), "Asset not found".byteInputStream())
        }
    }
}

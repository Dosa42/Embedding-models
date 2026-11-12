package com.ayrvokan.embeddingmodels

import android.app.Activity
import android.content.pm.ApplicationInfo
import android.content.Intent
import android.os.Bundle
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.net.Uri
import android.webkit.JavascriptInterface
import org.json.JSONObject
import kotlin.concurrent.thread

class MainActivity : Activity() {
    private lateinit var web: WebView
    private var fileCallback: ValueCallback<Array<Uri>>? = null
    private lateinit var chroma: ChromaBridge

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0) WebView.setWebContentsDebuggingEnabled(true)
        web = WebView(this)
        web.settings.javaScriptEnabled = true
        web.settings.domStorageEnabled = true
        chroma = ChromaBridge(applicationContext)
        web.addJavascriptInterface(chroma, "NativeChroma")
        web.webViewClient = object : WebViewClient() {
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
                    WebResourceResponse(mime, if (mime.startsWith("text/") || mime == "application/json") "UTF-8" else null, assets.open(path))
                } catch (_: Exception) {
                    WebResourceResponse("text/plain", "UTF-8", 404, "Not Found", emptyMap(), "Asset not found".byteInputStream())
                }
            }
        }
        web.addJavascriptInterface(object {
            @JavascriptInterface
            fun pickFolder() {
                runOnUiThread { startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT_TREE), 811) }
            }
        }, "FolderPicker")
        web.webChromeClient = object : WebChromeClient() {
            override fun onShowFileChooser(view: WebView?, callback: ValueCallback<Array<Uri>>?, params: FileChooserParams?): Boolean {
                fileCallback?.onReceiveValue(null)
                fileCallback = callback
                val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    type = "*/*"
                    addCategory(Intent.CATEGORY_OPENABLE)
                    putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
                }
                startActivityForResult(intent, 810)
                return true
            }
        }
        setContentView(web)
        web.loadUrl("https://appassets.androidplatform.net/")
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 811) {
            if (resultCode == RESULT_OK) data?.data?.let {
                try { contentResolver.takePersistableUriPermission(it, Intent.FLAG_GRANT_READ_URI_PERMISSION) } catch (_: SecurityException) { }
                thread(name = "chroma-folder-scan") {
                    val error = try { chroma.selectTree(it); null } catch (failure: Exception) { failure.message ?: "Could not read folder" }
                    runOnUiThread {
                        if (isDestroyed) return@runOnUiThread
                        if (error == null) web.evaluateJavascript("window.onFolderReady()", null)
                        else web.evaluateJavascript("document.getElementById('selected').textContent = ${JSONObject.quote(error)}", null)
                    }
                }
            }
            return
        }
        if (requestCode != 810) return
        val files = if (resultCode == RESULT_OK) {
            data?.clipData?.let { clip -> Array(clip.itemCount) { clip.getItemAt(it).uri } }
                ?: data?.data?.let { arrayOf(it) }
        } else null
        fileCallback?.onReceiveValue(files)
        fileCallback = null
    }

    override fun onDestroy() {
        web.destroy()
        chroma.close()
        super.onDestroy()
    }
}

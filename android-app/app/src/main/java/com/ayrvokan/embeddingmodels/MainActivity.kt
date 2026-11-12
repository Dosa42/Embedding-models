package com.ayrvokan.embeddingmodels

import android.Manifest
import android.app.Activity
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.text.Html
import android.webkit.JavascriptInterface
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URI
import kotlin.concurrent.thread

class MainActivity : Activity() {
    private lateinit var web: WebView
    private var fileCallback: ValueCallback<Array<Uri>>? = null
    private var service: FastApiService? = null
    private var bound = false
    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            service = (binder as FastApiService.LocalBinder).service
            thread(name = "rag-ui-startup") {
                var error: String? = null
                for (attempt in 0 until 120) {
                    if (isDestroyed) return@thread
                    val current = service ?: return@thread
                    current.runtime.failure?.let { error = it }
                    if (error != null) break
                    try {
                        val request = URI("http://127.0.0.1:8765/health").toURL().openConnection() as HttpURLConnection
                        request.connectTimeout = 500
                        request.readTimeout = 500
                        val available = try { request.responseCode == 200 } finally { request.disconnect() }
                        if (available) {
                            runOnUiThread { if (!isDestroyed) web.loadUrl("http://127.0.0.1:8765/") }
                            return@thread
                        }
                    } catch (_: Exception) { }
                    Thread.sleep(500)
                }
                runOnUiThread { if (!isDestroyed) showStartup(error ?: "The RAG service did not start. Reopen Source Atlas to retry.") }
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) { service = null }
    }

    private fun showStartup(message: String) {
        web.loadDataWithBaseURL(null, "<html><head><meta name='viewport' content='width=device-width,initial-scale=1'></head><body style='margin:0;padding:48px 24px;background:#f6f4ef;color:#243c30;font:16px/1.6 sans-serif'><h1 style='font-family:serif;font-weight:400'>Source Atlas</h1><p>${Html.escapeHtml(message)}</p></body></html>", "text/html", "UTF-8", null)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0) WebView.setWebContentsDebuggingEnabled(true)
        web = WebView(this)
        web.settings.javaScriptEnabled = true
        web.settings.domStorageEnabled = true
        web.webViewClient = WebViewClient()
        web.addJavascriptInterface(object {
            @JavascriptInterface
            fun pickFolder() { runOnUiThread { startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT_TREE), 811) } }
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
        showStartup("Starting FastAPI, LangChain and your local Chroma library…")
        val intent = Intent(this, FastApiService::class.java)
        startForegroundService(intent)
        bound = bindService(intent, connection, Context.BIND_AUTO_CREATE)
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 812)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 811) {
            if (resultCode == RESULT_OK) data?.data?.let { uri ->
                try { contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) } catch (_: SecurityException) { }
                thread(name = "source-folder-scan") {
                    val error = try {
                        val current = service ?: throw IllegalStateException("RAG service is unavailable")
                        current.runtime.chroma.selectTree(uri)
                        null
                    } catch (failure: Exception) { failure.message ?: "Could not read folder" }
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
            data?.clipData?.let { clip -> Array(clip.itemCount) { clip.getItemAt(it).uri } } ?: data?.data?.let { arrayOf(it) }
        } else null
        fileCallback?.onReceiveValue(files)
        fileCallback = null
    }

    override fun onDestroy() {
        if (bound) unbindService(connection)
        web.destroy()
        super.onDestroy()
    }
}

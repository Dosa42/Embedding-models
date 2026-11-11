package com.ayrvokan.embeddingmodels

import android.app.Activity
import android.content.pm.ApplicationInfo
import android.content.Intent
import android.os.Bundle
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import android.net.Uri
import android.webkit.JavascriptInterface
import kotlin.concurrent.thread

class MainActivity : Activity() {
    private lateinit var server: RagServer
    private lateinit var web: WebView
    private var fileCallback: ValueCallback<Array<Uri>>? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        server = RagServer(this)
        server.start()
        if (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0) WebView.setWebContentsDebuggingEnabled(true)
        web = WebView(this)
        web.settings.javaScriptEnabled = true
        web.settings.domStorageEnabled = true
        web.webViewClient = WebViewClient()
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
        web.loadUrl("http://127.0.0.1:8765/")
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 811) {
            if (resultCode == RESULT_OK) data?.data?.let {
                thread {
                    server.selectTree(it)
                    runOnUiThread { web.evaluateJavascript("window.onFolderReady()", null) }
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
        server.close()
        super.onDestroy()
    }
}

package com.ayrvokan.embeddingmodels

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlin.concurrent.thread

class FastApiService : Service() {
    inner class LocalBinder : Binder() {
        val service: FastApiService get() = this@FastApiService
    }

    private val binder = LocalBinder()
    lateinit var runtime: RagRuntime
        private set
    @Volatile private var module: PyObject? = null
    private var serverThread: Thread? = null

    override fun onCreate() {
        super.onCreate()
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel("fastapi_rag", "Source Atlas RAG", NotificationManager.IMPORTANCE_LOW))
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val notification = Notification.Builder(this, "fastapi_rag")
            .setSmallIcon(R.drawable.ic_source_atlas)
            .setContentTitle("Source Atlas RAG")
            .setContentText("Local FastAPI, LangChain and Chroma service")
            .setContentIntent(open)
            .setOngoing(true)
            .setCategory(Notification.CATEGORY_SERVICE)
            .build()
        if (Build.VERSION.SDK_INT >= 34) startForeground(1, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        else startForeground(1, notification)
        runtime = RagRuntime(applicationContext, ChromaBridge(applicationContext), EmbeddingEngine(applicationContext))
        serverThread = thread(name = "fastapi-uvicorn") {
            try {
                synchronized(Python::class.java) {
                    if (!Python.isStarted()) Python.start(AndroidPlatform(applicationContext))
                }
                val entry = Python.getInstance().getModule("atlas_rag")
                module = entry
                entry.callAttr("run", runtime, "127.0.0.1", 8765)
            } catch (error: Throwable) {
                runtime.failure = error.message ?: error.javaClass.simpleName
                runtime.state = "failed"
                Log.e("SourceAtlasRag", "FastAPI startup failed", error)
                stopSelf()
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = START_STICKY
    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        runtime.embeddings.close()
        thread(name = "fastapi-shutdown") {
            try { module?.callAttr("stop") } catch (error: Exception) { Log.e("SourceAtlasRag", "FastAPI shutdown failed", error) }
            serverThread?.join(10000)
            runtime.chroma.close()
        }
        super.onDestroy()
    }
}

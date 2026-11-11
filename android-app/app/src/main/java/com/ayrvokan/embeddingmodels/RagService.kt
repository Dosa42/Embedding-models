package com.ayrvokan.embeddingmodels

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.Uri
import android.os.Binder
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import kotlin.concurrent.thread

class RagService : Service() {
    inner class LocalBinder : Binder() {
        val service: RagService get() = this@RagService
    }

    private val binder = LocalBinder()
    private lateinit var server: RagServer
    private val main = Handler(Looper.getMainLooper())

    override fun onCreate() {
        super.onCreate()
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel("rag_server", "Local RAG server", NotificationManager.IMPORTANCE_LOW))
        val openApp = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val notification = Notification.Builder(this, "rag_server")
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentTitle("Source Atlas RAG is running")
            .setContentText("Chroma search is available on this device")
            .setContentIntent(openApp)
            .setOngoing(true)
            .setCategory(Notification.CATEGORY_SERVICE)
            .build()
        if (Build.VERSION.SDK_INT >= 34) startForeground(1, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        else startForeground(1, notification)
        server = RagServer(applicationContext)
        server.start()
        getSharedPreferences("rag_service", MODE_PRIVATE).getString("selected_tree", null)?.let { saved ->
            thread(name = "rag-folder-restore") {
                try { server.selectTree(Uri.parse(saved)) } catch (_: Exception) { }
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = START_STICKY

    override fun onBind(intent: Intent?): IBinder = binder

    fun selectTree(uri: Uri, finished: (String?) -> Unit) {
        thread(name = "rag-folder-scan") {
            val error = try {
                server.selectTree(uri)
                getSharedPreferences("rag_service", MODE_PRIVATE).edit().putString("selected_tree", uri.toString()).apply()
                null
            } catch (failure: Exception) {
                failure.message ?: "Could not read folder"
            }
            main.post { finished(error) }
        }
    }

    override fun onDestroy() {
        if (::server.isInitialized) server.close()
        super.onDestroy()
    }
}

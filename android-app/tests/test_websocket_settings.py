import asyncio
import json
import socket
import threading
import time
from types import SimpleNamespace
import unittest
import urllib.request

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient
import uvicorn
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect

from test_atlas_rag import FakeRuntime, atlas_rag
from atlas_settings import WebSocketSettings


class WebSocketSettingsTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.app = atlas_rag.configure(self.runtime)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.runtime.assets.cleanup()

    def test_settings_report_saved_choices_and_effective_runtime_values(self):
        response = self.client.get("/settings/websocket")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["saved"]["max_message_size"], {"mode": "manual", "value": 16777216})
        self.assertEqual(result["effective"]["ws_ping_interval"], 20)
        self.assertEqual(result["active_connections"], 0)
        self.assertEqual(result["device"], self.runtime.metrics)

    def test_unlimited_and_disabled_choices_are_persisted_and_reloaded(self):
        settings = WebSocketSettings().model_dump()
        settings["max_message_size"] = {"mode": "unlimited"}
        settings["max_queue"] = {"mode": "unlimited"}
        settings["ping_interval"] = {"mode": "disabled"}
        settings["ping_timeout"] = {"mode": "disabled"}
        settings["compression"] = "disabled"
        response = self.client.put("/settings/websocket", json=settings)
        self.assertEqual(response.status_code, 200)
        effective = response.json()["effective"]
        for key in ["ws_max_size", "ws_max_queue", "ws_ping_interval", "ws_ping_timeout"]:
            self.assertIsNone(effective[key])
        self.assertFalse(effective["ws_per_message_deflate"])
        with TestClient(atlas_rag.configure(self.runtime)) as client:
            self.assertEqual(client.get("/settings/websocket").json()["saved"], response.json()["saved"])

    def test_manual_values_have_no_arbitrary_upper_ceiling(self):
        settings = WebSocketSettings().model_dump()
        settings["max_message_size"]["value"] = 2 ** 40
        settings["max_queue"]["value"] = 100000
        settings["ping_interval"]["value"] = 0.125
        settings["ping_timeout"]["value"] = 3600
        result = self.client.put("/settings/websocket", json=settings).json()
        self.assertEqual(result["effective"]["ws_max_size"], 2 ** 40)
        self.assertEqual(result["effective"]["ws_max_queue"], 100000)
        self.assertEqual(result["effective"]["ws_ping_interval"], 0.125)
        self.assertEqual(result["effective"]["ws_ping_timeout"], 3600)

    def test_invalid_manual_values_leave_saved_choices_intact(self):
        for field in ["max_message_size", "max_queue", "ping_interval", "ping_timeout"]:
            for value in [0, -1, None]:
                settings = WebSocketSettings().model_dump()
                settings[field]["value"] = value
                self.assertEqual(self.client.put("/settings/websocket", json=settings).status_code, 422)
        self.assertEqual(self.runtime.websocket_settings, {})

    def test_adaptive_values_follow_memory_connections_and_power(self):
        settings = WebSocketSettings(max_message_size={"mode": "adaptive"}, max_queue={"mode": "adaptive"}, ping_interval={"mode": "adaptive"}, ping_timeout={"mode": "adaptive"}, compression="adaptive")
        initial = settings.resolve(self.runtime.metrics, 1)
        busy = settings.resolve(self.runtime.metrics, 4)
        self.assertEqual(busy["ws_max_size"], initial["ws_max_size"] // 4)
        self.runtime.metrics["available_memory_bytes"] //= 2
        self.runtime.metrics["power_save"] = True
        power = settings.resolve(self.runtime.metrics, 1)
        self.assertEqual(power["ws_max_size"], initial["ws_max_size"] // 2)
        self.assertEqual(power["ws_ping_interval"], 60)
        self.assertEqual(power["ws_ping_timeout"], 120)
        self.assertFalse(power["ws_per_message_deflate"])

    def test_settings_assets_and_routes_are_available_without_websocket_app_route(self):
        self.assertEqual(self.client.get("/settings.js").text, "asset:settings.js")
        self.assertIn("/settings/websocket", self.client.get("/openapi.json").json()["paths"])
        self.assertFalse(any(route.path == "/rag/ws" for route in self.app.routes))


class AdaptiveRefreshTests(unittest.IsolatedAsyncioTestCase):
    async def test_adaptive_refresh_updates_server_without_ui_polling(self):
        runtime = FakeRuntime()
        app = atlas_rag.configure(runtime)
        controller = app.state.websocket
        controller.server = SimpleNamespace(config=SimpleNamespace(loaded=True), server_state=SimpleNamespace(connections=set()))
        settings = WebSocketSettings(max_message_size={"mode": "adaptive"}, adaptive_refresh_seconds=0.01)
        await controller.update(settings)
        await controller.start()
        try:
            initial = controller.server.config.ws_max_size
            runtime.metrics["available_memory_bytes"] //= 4
            for attempt in range(100):
                if controller.server.config.ws_max_size != initial:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(controller.server.config.ws_max_size, initial // 4)
        finally:
            await controller.close()
            runtime.assets.cleanup()


class LiveWebSocketSettingsTests(unittest.TestCase):
    def test_live_configuration_changes_real_uvicorn_connections(self):
        runtime = FakeRuntime()
        app = atlas_rag.configure(runtime)

        @app.websocket("/test-transport")
        async def echo(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    await websocket.send_text(await websocket.receive_text())
            except WebSocketDisconnect:
                pass

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        config = uvicorn.Config(app, host="127.0.0.1", port=port, lifespan="on", log_level="error", **app.state.websocket.effective)
        server = uvicorn.Server(config)
        app.state.websocket.server = server
        thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
        thread.start()

        def update(settings):
            request = urllib.request.Request(f"http://127.0.0.1:{port}/settings/websocket", data=json.dumps(settings).encode(), headers={"Content-Type": "application/json"}, method="PUT")
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.load(response)

        try:
            for attempt in range(500):
                if server.started:
                    break
                time.sleep(0.01)
            self.assertTrue(server.started)
            settings = WebSocketSettings().model_dump()
            uri = f"ws://127.0.0.1:{port}/test-transport"
            with connect(uri, compression=None, ping_interval=None, open_timeout=5) as connection:
                connection.send("before")
                self.assertEqual(connection.recv(timeout=5), "before")
                for field in ["max_message_size", "max_queue"]:
                    settings[field] = {"mode": "unlimited"}
                for field in ["ping_interval", "ping_timeout"]:
                    settings[field] = {"mode": "disabled"}
                settings["compression"] = "disabled"
                update(settings)
                connection.send("after")
                self.assertEqual(connection.recv(timeout=5), "after")
                for protocol in server.server_state.connections:
                    if hasattr(protocol, "keepalive_ping_task"):
                        self.assertIsNone(protocol.max_size)
                        self.assertIsNone(protocol.max_queue)
                        self.assertIsNone(protocol.ping_interval)
                        self.assertIsNone(protocol.ping_timeout)
                settings["enabled"] = False
                update(settings)
                with self.assertRaises(ConnectionClosed):
                    connection.recv(timeout=5)
            with self.assertRaises(InvalidStatus):
                connect(uri, open_timeout=5)
            settings["enabled"] = True
            settings["max_message_size"] = {"mode": "manual", "value": 64}
            update(settings)
            with connect(uri, compression=None, ping_interval=None, open_timeout=5) as connection:
                connection.send("x" * 100)
                with self.assertRaises(ConnectionClosed) as closed:
                    connection.recv(timeout=5)
                self.assertEqual(closed.exception.rcvd.code, 1009)
            self.assertEqual(server.config.ws_max_size, 64)
            self.assertFalse(server.config.ws_per_message_deflate)
        finally:
            server.should_exit = True
            thread.join(10)
            listener.close()
            runtime.assets.cleanup()
            self.assertFalse(thread.is_alive())

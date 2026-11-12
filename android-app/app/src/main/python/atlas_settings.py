import asyncio
from contextlib import suppress
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator
from uvicorn.protocols.websockets.websockets_impl import WebSocketProtocol


PositiveNumber = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class CapacitySetting(BaseModel):
    mode: Literal["manual", "adaptive", "unlimited"] = "manual"
    value: Annotated[int, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def manual_value(self):
        if self.mode == "manual" and self.value is None:
            raise ValueError("Enter a positive value for manual capacity")
        return self


class TimingSetting(BaseModel):
    mode: Literal["manual", "adaptive", "disabled"] = "manual"
    value: PositiveNumber | None = None

    @model_validator(mode="after")
    def manual_value(self):
        if self.mode == "manual" and self.value is None:
            raise ValueError("Enter a positive duration for manual timing")
        return self


class WebSocketSettings(BaseModel):
    enabled: bool = True
    max_message_size: CapacitySetting = Field(default_factory=lambda: CapacitySetting(value=16 * 1024 * 1024))
    max_queue: CapacitySetting = Field(default_factory=lambda: CapacitySetting(value=32))
    ping_interval: TimingSetting = Field(default_factory=lambda: TimingSetting(value=20))
    ping_timeout: TimingSetting = Field(default_factory=lambda: TimingSetting(value=20))
    compression: Literal["enabled", "disabled", "adaptive"] = "enabled"
    adaptive_refresh_seconds: PositiveNumber = 5

    def adaptive(self):
        return self.compression == "adaptive" or any(value.mode == "adaptive" for value in [self.max_message_size, self.max_queue, self.ping_interval, self.ping_timeout])

    def resolve(self, metrics, connections):
        available = int(metrics["available_memory_bytes"])
        peers = max(1, connections)
        size = self.max_message_size.value
        if self.max_message_size.mode == "unlimited":
            size = None
        elif self.max_message_size.mode == "adaptive":
            size = max(1, available // (64 * peers))
        queue = self.max_queue.value
        if self.max_queue.mode == "unlimited":
            queue = None
        elif self.max_queue.mode == "adaptive":
            message_budget = size or max(1, available // (64 * peers))
            queue = max(1, available // (8 * peers * message_budget))
        interval = self.ping_interval.value
        if self.ping_interval.mode == "disabled":
            interval = None
        elif self.ping_interval.mode == "adaptive":
            interval = 60.0 if metrics.get("power_save") or not metrics.get("interactive", True) else 20.0
        timeout = self.ping_timeout.value
        if self.ping_timeout.mode == "disabled":
            timeout = None
        elif self.ping_timeout.mode == "adaptive":
            timeout = 2 * (interval or 20.0)
        compression = self.compression == "enabled"
        if self.compression == "adaptive":
            compression = not metrics.get("power_save", False) and not metrics.get("low_memory", False)
        return {"ws": "websockets" if self.enabled else "none", "ws_max_size": size, "ws_max_queue": queue, "ws_ping_interval": interval, "ws_ping_timeout": timeout, "ws_per_message_deflate": compression}


class WebSocketController:
    def __init__(self, runtime, bridge):
        self.runtime = runtime
        self.bridge = bridge
        self.settings = WebSocketSettings.model_validate(bridge(runtime, "webSocketSettings"))
        self.server = None
        self.task = None
        self.changed = asyncio.Event()
        self.lock = asyncio.Lock()
        self.metrics = bridge(runtime, "webSocketMetrics")
        self.connections = 0
        self.effective = self.settings.resolve(self.metrics, 0)

    def snapshot(self):
        return {"saved": self.settings.model_dump(), "effective": self.effective, "device": self.metrics, "active_connections": self.connections, "compression_applies": "next_connection"}

    async def refresh(self):
        self.metrics = self.bridge(self.runtime, "webSocketMetrics")
        connections = list(self.server.server_state.connections) if self.server is not None else []
        sockets = [connection for connection in connections if isinstance(connection, WebSocketProtocol)]
        self.connections = len(sockets)
        self.effective = self.settings.resolve(self.metrics, self.connections)
        if self.server is None:
            return
        config = self.server.config
        for key, value in self.effective.items():
            setattr(config, key, value)
        if config.loaded:
            config.ws_protocol_class = WebSocketProtocol if self.settings.enabled else None
        for connection in connections:
            if hasattr(connection, "ws_protocol_class"):
                connection.ws_protocol_class = WebSocketProtocol if self.settings.enabled else None
        for connection in sockets:
            connection.max_size = self.effective["ws_max_size"]
            connection.max_queue = self.effective["ws_max_queue"]
            waiter = getattr(connection, "_put_message_waiter", None)
            if waiter is not None and not waiter.done():
                waiter.set_result(None)
            timing_changed = connection.ping_interval != self.effective["ws_ping_interval"] or connection.ping_timeout != self.effective["ws_ping_timeout"]
            connection.ping_interval = self.effective["ws_ping_interval"]
            connection.ping_timeout = self.effective["ws_ping_timeout"]
            if timing_changed and hasattr(connection, "keepalive_ping_task"):
                connection.keepalive_ping_task.cancel()
                if connection.ping_interval is not None and not connection.closed:
                    connection.keepalive_ping_task = asyncio.create_task(connection.keepalive_ping())
            if not self.settings.enabled:
                connection.fail_connection(1001, "WebSockets disabled in Settings")

    async def read(self):
        async with self.lock:
            await self.refresh()
            return self.snapshot()

    async def update(self, settings):
        async with self.lock:
            self.bridge(self.runtime, "saveWebSocketSettings", settings.model_dump_json())
            self.settings = settings
            await self.refresh()
            self.changed.set()
            return self.snapshot()

    async def watch(self):
        while True:
            if self.settings.adaptive():
                with suppress(TimeoutError):
                    await asyncio.wait_for(self.changed.wait(), self.settings.adaptive_refresh_seconds)
            else:
                await self.changed.wait()
            self.changed.clear()
            async with self.lock:
                await self.refresh()

    async def start(self):
        await self.refresh()
        self.task = asyncio.create_task(self.watch())

    async def close(self):
        if self.task is not None:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task

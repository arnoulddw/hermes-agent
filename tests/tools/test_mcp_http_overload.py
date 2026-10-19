"""HTTP overload must not trigger SSE or bypass the server's retry deadline."""

import asyncio
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tools import mcp_tool
from tools.mcp_tool import MCPServerTask
from tools.mcp_tool_errors import _make_http_rejection_recorder


@pytest.mark.parametrize("status", [401, 403, 429, 502, 503])
def test_recorded_http_status_overrides_opaque_sdk_transport_error(monkeypatch, status):
    """An SDK's generic -32603 must not turn overload/auth/outages into SSE discovery."""
    task = MCPServerTask("overload")
    task._config = {}
    attempted = []

    class OpaqueError(Exception):
        error = type("Error", (), {"code": -32603})()

    async def serve(self, transport, label, timeout):
        attempted.append(label)
        raise OpaqueError("Server returned an error response")

    def streamable(self, *args, **kwargs):
        self._http_rejection.update(status=status, method="POST", url="http://localhost/mcp", body="rejected")
        return object()

    monkeypatch.setattr(MCPServerTask, "_serve_transport", serve)
    monkeypatch.setattr(MCPServerTask, "_streamable_http_transport", streamable)
    monkeypatch.setattr(MCPServerTask, "_sse_transport", lambda *args: object())
    monkeypatch.setattr(MCPServerTask, "_build_oauth_auth", lambda *args: None)
    with pytest.raises(ConnectionError, match=f"HTTP {status}"):
        asyncio.run(task._run_http({"url": "http://localhost/mcp"}))
    assert "SSE" not in attempted


@pytest.mark.parametrize("protocol", ["legacy", "stateless"])
def test_real_http_retry_after_survives_new_task_and_discovery_failure(monkeypatch, tmp_path, protocol):
    """Both the SDK wire path and a fresh task must honour the shared scoped cooldown."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_tool, "_CONNECT_RETRY_BASE_BACKOFF_SEC", 0.01)
    posts = []
    gets = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", "0")))
            posts.append(time.monotonic())
            body = b'{"error":"Too many MCP sessions"}'
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Retry-After", "3")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            gets.append(self.path)
            self.send_error(400)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = {"url": f"http://127.0.0.1:{server.server_port}/mcp", "protocol": protocol, "connect_timeout": 5}

    async def scenario():
        from tools.mcp_tool_discovery import _record_connect_failure

        for attempt in range(2):
            task = MCPServerTask("retry-after")
            task._config = config
            with pytest.raises(Exception):
                await task._run_http(config)
            if attempt == 0:
                # Initial discovery also records its own short exponential cooldown.
                with mcp_tool._lock:
                    _record_connect_failure(task.name)
        cancelled = MCPServerTask("retry-after")
        cancelled._config = config
        cancelled._shutdown_event.set()
        assert await cancelled._run_http(config) == "shutdown"

    try:
        asyncio.run(asyncio.wait_for(scenario(), timeout=20))
        assert len(posts) == 2
        assert posts[1] - posts[0] >= 3
        assert gets == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("header,expected", [
    ("60", 60), ("0", 0), ("-1", None), ("nan", None), ("invalid", None),
    ("Sun, 18 Oct 2026 12:01:00 GMT", 60),
    ("Sun, 18 Oct 2026 11:59:00 GMT", 0),
])
def test_retry_after_recorder_accepts_valid_delays_only(monkeypatch, header, expected):
    """The actual HTTP hook retains a delay, not an unvalidated raw header."""
    monkeypatch.setattr(time, "time", lambda: datetime(2026, 10, 18, 12, tzinfo=UTC).timestamp())
    async def scenario():
        httpx = mcp_tool.sdk_httpx()
        sink = {}
        transport = httpx.MockTransport(lambda request: httpx.Response(
            429, headers={"retry-after": header}, json={"error": "busy"}))
        async with httpx.AsyncClient(transport=transport, event_hooks={
                "response": [_make_http_rejection_recorder(sink)]}) as client:
            await client.post("http://localhost/mcp")
        assert sink.get("retry_after") == expected

    asyncio.run(scenario())


def test_stale_rejection_does_not_turn_timeout_into_sse():
    from tools.mcp_tool_errors import _is_streamable_http_rejection

    assert not _is_streamable_http_rejection(TimeoutError(), {"status": 400})
    assert not _is_streamable_http_rejection(asyncio.CancelledError(), {"status": 400})


def test_protocol_fallback_clears_previous_http_rejection():
    task = MCPServerTask("stale-rejection")
    task._config = {"protocol": "stateless"}

    class Session:
        async def discover(self):
            task._http_rejection.update(status=400)
            raise ValueError("method not found")

        async def initialize(self):
            assert task._http_rejection == {}
            raise TimeoutError()

    with pytest.raises(TimeoutError):
        asyncio.run(task._negotiate_session(Session(), 1))


@pytest.mark.parametrize("status", [401, 403, 429, 503])
def test_stateless_negotiation_does_not_fallback_on_http_rejection(status):
    task = MCPServerTask("negotiation-overload")
    task._config = {"protocol": "stateless"}
    calls = []

    class Session:
        async def discover(self):
            calls.append("discover")
            task._http_rejection.update(status=status)
            raise ConnectionError("rejected")

        async def initialize(self):
            calls.append("initialize")

    with pytest.raises(ConnectionError, match="rejected"):
        asyncio.run(task._negotiate_session(Session(), 1))
    assert calls == ["discover"]

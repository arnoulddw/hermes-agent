"""Python SDK session termination must release strict-JSON HTTP servers immediately."""

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tools import mcp_tool
from tools.mcp_tool import MCPServerTask


def test_sdk_bodyless_delete_releases_session_without_json_content_type(monkeypatch):
    """Exercise the real SDK through Hermes's owned client, not a hand-built DELETE."""
    mcp_tool._ensure_mcp_sdk()
    active = set()
    deletes = []
    posts = []
    clients = []
    sdk_transport = mcp_tool.streamable_http_client

    @asynccontextmanager
    async def capture_client(*args, http_client=None, **kwargs):
        clients.append(http_client)
        async with sdk_transport(*args, http_client=http_client, **kwargs) as streams:
            yield streams

    monkeypatch.setattr(mcp_tool, "streamable_http_client", capture_client)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body=b"", headers=None):
            self.send_response(status)
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            posts.append(self.headers.get("content-type"))
            if body["method"] == "initialize":
                active.add("fixture-session")
                response = {"jsonrpc": "2.0", "id": body["id"], "result": {
                    "protocolVersion": body["params"]["protocolVersion"],
                    "capabilities": {}, "serverInfo": {"name": "fixture", "version": "1"}}}
                self.reply(200, json.dumps(response).encode(), {
                    "Content-Type": "application/json", "Mcp-Session-Id": "fixture-session"})
            else:
                self.reply(202)

        def do_GET(self):
            self.reply(405)

        def do_DELETE(self):
            self.rfile.read(int(self.headers.get("content-length", "0")))
            content_type = self.headers.get("content-type")
            deletes.append(content_type)
            # Fastify rejects empty JSON before the MCP handler can release a slot.
            if content_type and content_type.split(";")[0] == "application/json":
                self.reply(400)
            else:
                active.discard(self.headers.get("mcp-session-id"))
                self.reply(200)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    task = MCPServerTask("cleanup-fixture")
    url = f"http://127.0.0.1:{server.server_port}/api/mcp"

    async def scenario():
        for _ in range(3):
            async with task._streamable_http_transport(
                    url, {"mcp-protocol-version": mcp_tool.LATEST_HANDSHAKE_VERSION},
                    5, True, None, None, False, set()) as (read, write, *_):
                async with mcp_tool.ClientSession(read, write) as session:
                    await session.initialize()
                    assert active == {"fixture-session"}
                    invalid = await clients[-1].request("DELETE", url, content=b"{", headers={
                        "content-type": "application/json", "mcp-session-id": "fixture-session"})
                    assert invalid.status_code == 400
                    assert active == {"fixture-session"}, "Nonempty JSON DELETE headers must be preserved"
            assert not active, "SDK close leaked a session slot"

    try:
        asyncio.run(asyncio.wait_for(scenario(), timeout=20))
        assert deletes == ["application/json", None] * 3
        assert posts and all(value == "application/json" for value in posts)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)

# SPDX-License-Identifier: Apache-2.0
"""Exercise OpenCode's real tool path against a local stub, with zero model calls."""

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def probe(config, report, review_path, output):
    blocked = output / "must-not-be-written.txt"

    class Handler(BaseHTTPRequestHandler):
        stage = 0

        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            delta = {"role": "assistant", "content": "probe complete"}
            finish = "stop"
            if any(t.get("function", {}).get("name") == "read" for t in body.get("tools", [])) and Handler.stage < 3:
                stage = Handler.stage
                Handler.stage += 1
                name = "read" if stage == 0 else "write"
                params = (
                    {"filePath": str(report), "limit": 1}
                    if stage == 0
                    else {"filePath": str(review_path if stage == 1 else blocked), "content": "permission probe"}
                )
                delta = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"probe_{stage}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(params)},
                        }
                    ],
                }
                finish = "tool_calls"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for d, f in ((delta, None), ({}, finish)):
                chunk = {
                    "id": "probe",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "deepseek-flash",
                    "choices": [{"index": 0, "delta": d, "finish_reason": f}],
                }
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    stub_config = json.loads(json.dumps(config))
    stub_config.update(
        model="deepseek/deepseek-flash", small_model="deepseek/deepseek-flash", enabled_providers=["deepseek"]
    )
    stub_config["provider"]["deepseek"]["options"] = {
        "apiKey": "offline-probe-only",
        "baseURL": f"http://127.0.0.1:{server.server_port}/v1",
    }
    path = output / "probe-config.json"
    path.write_text(json.dumps(stub_config))
    env = dict(os.environ, OPENCODE_CONFIG=str(path), MAIN2MAIN_API_KEY="offline-probe-only")
    try:
        result = subprocess.run(
            [
                "opencode",
                "run",
                "--auto",
                "--format",
                "json",
                "--print-logs",
                "--log-level",
                "DEBUG",
                "--model",
                "deepseek/deepseek-flash",
                "--",
                "Execute only the provided permission probes.",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        (output / "permission-probe.jsonl").write_text(result.stdout, encoding="utf-8")
        # The child has only a dummy API key and a localhost provider.
        (output / "permission-probe.log").write_text(result.stderr, encoding="utf-8")
        events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        calls = {e["part"]["callID"]: e["part"]["state"] for e in events if e.get("type") == "tool_use"}
        assert calls.get("probe_0", {}).get("status") == "completed", "Report read probe failed; no model called"
        assert calls.get("probe_1", {}).get("status") == "completed", "Review write probe failed; no model called"
        assert calls.get("probe_2", {}).get("status") == "error" and not blocked.exists(), "Write boundary probe failed"
        assert review_path.read_text() == "permission probe"
        (output / "permission-probe-status.json").write_text(json.dumps({"passed": True, "real_model_calls": 0}))
    finally:
        server.shutdown()
        path.unlink(missing_ok=True)
        if review_path.exists() and review_path.read_text() == "permission probe":
            review_path.unlink()

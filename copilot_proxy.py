"""OpenAI-compatible HTTP proxy that calls the GitHub Copilot CLI.

Maps POST /v1/chat/completions -> `copilot -p <prompt> --model <model>`.

Usage:
    python copilot_proxy.py --port 9001
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 9001
COPILOT_BIN = os.environ.get(
    "COPILOT_BIN",
    "/data/users/zacharie/.vscode-server/data/User/globalStorage/github.copilot-chat/copilotCli/copilot",
)
TIMEOUT_S = int(os.environ.get("COPILOT_TIMEOUT", "180"))
TRACE_DIR = os.environ.get("COPILOT_TRACE_DIR", "/data/users/zacharie/CheeseBench/results/copilot_eval/_proxy_traces")
os.makedirs(TRACE_DIR, exist_ok=True)

# Concurrent CLI invocations (the CLI is heavy; cap to avoid OOM/quota spikes)
MAX_CONCURRENT = int(os.environ.get("COPILOT_MAX_CONCURRENT", "8"))
_sema = threading.Semaphore(MAX_CONCURRENT)
_counter_lock = threading.Lock()
_counter = {"req": 0, "ok": 0, "err": 0, "tokens": 0}

# Footer printed by `copilot` after the actual reply. We strip it before returning.
FOOTER_RE = re.compile(
    r"(?:^|\n)\s*Changes\s+\+?-?\d+.*?(?:\n.*)*?Tokens\s+.*$",
    re.DOTALL,
)
HEADER_PATTERNS = [
    re.compile(r"^!\s+Third-party MCP servers.*$", re.MULTILINE),
    re.compile(r"^\s+Only built-in servers are available\.\s*$", re.MULTILINE),
    re.compile(r"^●\s.*$", re.MULTILINE),  # spinner residue
]


def flatten_messages(messages: list[dict]) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        # Multimodal: list of parts -> concatenate text only (images dropped, copilot CLI is text-only)
        if isinstance(content, list):
            text_bits = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    text_bits.append(p.get("text", ""))
                elif isinstance(p, dict) and p.get("type") in ("image_url", "image"):
                    text_bits.append("[image omitted]")
            content = "\n".join(text_bits)
        parts.append(f"[{role}]\n{content}")
    parts.append(
        "\n[INSTRUCTION]\nReply ONLY with the assistant message in the requested format. "
        "Do not call tools, do not edit files, do not run commands. Output text only."
    )
    return "\n\n".join(parts)


def call_copilot(model: str, prompt: str, req_id: str) -> tuple[str, int]:
    cmd = [
        COPILOT_BIN,
        "-p", prompt,
        "--model", model,
        "--allow-all",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            env={**os.environ, "NO_COLOR": "1", "FORCE_COLOR": "0"},
        )
        out = proc.stdout or ""
        err = proc.stderr or ""
    except subprocess.TimeoutExpired:
        return "", -1

    raw = out + ("\n[STDERR]\n" + err if err.strip() else "")
    # Save raw trace
    with open(os.path.join(TRACE_DIR, f"{req_id}.txt"), "w") as f:
        f.write(f"MODEL: {model}\nELAPSED: {time.time()-t0:.2f}s\nEXIT: {proc.returncode}\n\n--- PROMPT ---\n{prompt}\n\n--- RAW OUTPUT ---\n{raw}\n")

    text = out
    for pat in HEADER_PATTERNS:
        text = pat.sub("", text)
    text = FOOTER_RE.sub("", text)
    text = text.strip()
    return text, proc.returncode


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[proxy] " + (fmt % args) + "\n")

    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # health
        if self.path in ("/", "/health"):
            self._send_json(200, {"ok": True, "stats": _counter})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            self._send_json(404, {"error": "unknown endpoint"})
            return
        n = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(n))
        except Exception as e:
            self._send_json(400, {"error": f"bad json: {e}"})
            return

        model = body.get("model") or "claude-haiku-4.5"
        messages = body.get("messages") or []
        prompt = flatten_messages(messages)
        req_id = f"{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"

        with _counter_lock:
            _counter["req"] += 1
            req_no = _counter["req"]
        sys.stderr.write(f"[proxy] #{req_no} model={model} prompt_chars={len(prompt)}\n")

        with _sema:
            text, rc = call_copilot(model, prompt, req_id)

        if rc != 0 or not text:
            with _counter_lock:
                _counter["err"] += 1
            self._send_json(502, {"error": f"copilot exit {rc}", "raw": text[:500]})
            return

        with _counter_lock:
            _counter["ok"] += 1

        resp = {
            "id": f"chatcmpl-{req_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self._send_json(200, resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    sys.stderr.write(f"[proxy] listening on http://127.0.0.1:{args.port} max_concurrent={MAX_CONCURRENT}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()

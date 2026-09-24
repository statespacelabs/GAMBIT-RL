#!/usr/bin/env python3
"""Package-local console/browser HUD for the Phase 6 research demo."""
from __future__ import annotations

import json
import os
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


LIMITATION = (
    "This mode demonstrates selected reproducible scenarios and is not a "
    "broad autonomous reliability certification."
)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def format_hud(payload: dict[str, Any]) -> str:
    if payload.get("screen") == "playlist_complete":
        summary = payload.get("summary", {})
        return "\n".join(
            [
                "EXPERIMENTAL AUTONOMOUS AI",
                "PLAYLIST COMPLETE",
                f"Scenarios completed: {summary.get('scenarios_completed', 0)}",
                f"Enemies contacted: {summary.get('enemies_contacted', 0)}",
                "Damage-positive engagements: "
                f"{summary.get('damage_positive_engagements', 0)}",
                f"Enemies defeated: {summary.get('enemies_defeated', 0)}",
                f"Timeouts: {summary.get('timeouts', 0)}",
                f"Known limitation: {LIMITATION}",
            ]
        )
    scenario = payload.get("scenario", {})
    metrics = payload.get("metrics", {})
    total = scenario.get("total", "?")
    order = scenario.get("order", "?")
    health = metrics.get("opponent_health_percent")
    health_text = "unknown" if health is None else f"{float(health):.0f}%"
    contact = metrics.get("time_to_contact_seconds")
    contact_text = "pending" if contact is None else f"{float(contact):.2f}s"
    lines = [
        "EXPERIMENTAL AUTONOMOUS AI",
        f"Scenario {order}/{total}: {scenario.get('scenario_id', '')}",
        f"State: {payload.get('state', 'SEARCH')}",
        f"Time to contact: {contact_text}",
        f"Damage dealt: {float(metrics.get('damage_dealt', 0.0)):.0f}",
        f"Shots / hits: {int(metrics.get('shots', 0))} / "
        f"{int(metrics.get('hits', 0))}",
        f"Opponent health: {health_text}",
        f"Result: {payload.get('result', 'in progress')}",
    ]
    if payload.get("message"):
        lines.append(str(payload["message"]))
    return "\n".join(lines)


class ConsoleHud:
    """Poll a JSON state file without touching policy inputs or actions."""

    def __init__(self, state_path: Path, *, interval: float = 0.25) -> None:
        self.state_path = state_path
        self.interval = float(interval)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def _run(self) -> None:
        previous = ""
        while not self.stop_event.wait(self.interval):
            try:
                raw = self.state_path.read_text(encoding="utf-8")
            except (FileNotFoundError, OSError):
                continue
            if raw == previous:
                continue
            previous = raw
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            print("\n" + format_hud(payload) + "\n", flush=True)


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *_args: Any) -> None:
        return


class BrowserHud:
    """Serve the package HUD on localhost for a browser or OBS overlay."""

    def __init__(self, run_dir: Path, *, port: int = 17831) -> None:
        self.run_dir = run_dir
        self.port = int(port)
        handler = lambda *args, **kwargs: _QuietHandler(  # noqa: E731
            *args, directory=str(run_dir), **kwargs
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )

    def start(self, *, open_browser: bool = True) -> str:
        self.thread.start()
        url = f"http://127.0.0.1:{self.port}/runtime_observer.html"
        if open_browser:
            webbrowser.open(url)
        return url

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


def wait_for_state(path: Path, *, timeout: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            time.sleep(0.05)
    return {}

"""In-process pub/sub feeding the Server-Sent Events stream.

This is what lets the UI drop its polling loop. Every state change publishes
once; each connected browser holds one queue and one HTTP connection.

Subscriber queues are bounded and drop their *oldest* entry when full. A browser
that has been backgrounded for ten minutes must not be able to pin log lines in
memory, and for a progress feed the newest message is the only one that matters.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Iterator

MAX_QUEUE = 512


class Broker:
    def __init__(self) -> None:
        self._subs: set[queue.Queue] = set()
        self._lock = threading.Lock()
        self._seq = 0

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=MAX_QUEUE)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    def publish(self, kind: str, payload: Any) -> None:
        with self._lock:
            self._seq += 1
            message = {"seq": self._seq, "kind": kind, "ts": time.time(), "data": payload}
            targets = list(self._subs)
        for q in targets:
            _offer(q, message)

    def close(self) -> None:
        with self._lock:
            targets = list(self._subs)
            self._subs.clear()
        for q in targets:
            _offer(q, None)


def _offer(q: queue.Queue, message: Any) -> None:
    try:
        q.put_nowait(message)
    except queue.Full:
        try:
            q.get_nowait()          # drop oldest
            q.put_nowait(message)
        except (queue.Empty, queue.Full):
            pass


BROKER = Broker()


def publish(kind: str, payload: Any) -> None:
    BROKER.publish(kind, payload)


def log(line: str, level: str = "info") -> None:
    publish("log", {"line": line, "level": level})


def sse_stream(q: queue.Queue, keepalive: float = 15.0) -> Iterator[bytes]:
    """Yield SSE frames until the subscriber is closed or the client leaves."""
    yield b": connected\n\n"
    while True:
        try:
            message = q.get(timeout=keepalive)
        except queue.Empty:
            yield b": keepalive\n\n"      # keeps proxies and the browser honest
            continue
        if message is None:
            yield b"event: close\ndata: {}\n\n"
            return
        body = json.dumps(message["data"], ensure_ascii=False, default=str)
        frame = f"id: {message['seq']}\nevent: {message['kind']}\ndata: {body}\n\n"
        yield frame.encode("utf-8")

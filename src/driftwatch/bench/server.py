from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI


@dataclass
class ServerHandle:
    url: str
    server: uvicorn.Server
    thread: threading.Thread

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def start_server(app: FastAPI, host: str = "127.0.0.1", startup_timeout: float = 15.0) -> ServerHandle:
    config = uvicorn.Config(app, host=host, port=0, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="driftwatch-bench-server", daemon=True)
    thread.start()
    deadline = time.monotonic() + startup_timeout
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("server did not start in time")
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    return ServerHandle(f"http://{host}:{port}", server, thread)

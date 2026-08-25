from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lowram_ai.api import create_app
from lowram_ai.tests.test_llama import make_llama_fixture


with tempfile.TemporaryDirectory() as directory:
    model = Path(directory) / "tiny.gguf"
    make_llama_fixture(model)
    os.environ["LOWRAM_RATE_LIMIT"] = "0"
    app = create_app(str(model), max_context=4, max_ram_mb=1024)
    config = uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("local API server did not start")
    print("LOCAL_SERVER_READY http://127.0.0.1:8765")
    print("Press Ctrl-C to stop")
    try:
        while thread.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        server.should_exit = True
        thread.join(timeout=5)

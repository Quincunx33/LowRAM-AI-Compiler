"""Optional HTTP serving layer for the low-RAM runtime."""
from contextlib import asynccontextmanager
from pathlib import Path
import json
import os
import threading
import time
from typing import Any, Iterator

from .llama import LlamaRuntime
from .workspace import Workspace, WorkspaceError


def create_app(model_path: str, *, max_context: int | None = None, max_ram_mb: int | None = None):
    try:
        from fastapi import Body, FastAPI, File, HTTPException, UploadFile
        from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("FastAPI server dependencies are not installed") from exc

    context_limit = max_context or 512
    runtime: LlamaRuntime | None = None
    generation_lock = threading.Lock()
    configured_api_key = os.environ.get("LOWRAM_API_KEY")
    rate_limit = max(0, int(os.environ.get("LOWRAM_RATE_LIMIT", "0")))
    request_times: list[float] = []
    request_lock = threading.Lock()
    sessions: dict[str, dict[str, Any]] = {}
    sessions_lock = threading.Lock()
    max_sessions = max(1, int(os.environ.get("LOWRAM_MAX_SESSIONS", "32")))
    workspace = Workspace()

    @asynccontextmanager
    async def lifespan(app: Any):
        nonlocal runtime
        runtime = LlamaRuntime.open(model_path, max_context_tokens=context_limit, max_ram_mb=max_ram_mb)
        try:
            yield
        finally:
            if runtime is not None:
                runtime.close()
                runtime = None

    app = FastAPI(title="LowRAM AI", version="0.3.0", lifespan=lifespan)

    @app.middleware("http")
    async def security_middleware(request: Any, call_next: Any):
        from fastapi.responses import JSONResponse
        if configured_api_key and request.headers.get("x-api-key") != configured_api_key:
            return JSONResponse({"detail": "invalid API key"}, status_code=401)
        if rate_limit:
            now = time.monotonic()
            with request_lock:
                request_times[:] = [stamp for stamp in request_times if now - stamp < 60.0]
                if len(request_times) >= rate_limit:
                    return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
                request_times.append(now)
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def web_ui() -> str:
        return (Path(__file__).with_name("web") / "index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok" if runtime is not None else "starting"}

    @app.get("/v1/model")
    def model_info() -> dict[str, Any]:
        if runtime is None:
            raise HTTPException(status_code=503, detail="model is not ready")
        return {"architecture": runtime.config.architecture, "context_length": runtime.max_context_tokens, "memory": {"weights_bytes": runtime.weight_memory_bytes, "kv_cache_bytes": runtime.cache_memory_bytes, "estimated_bytes": runtime.estimated_model_bytes}}

    def new_session() -> str:
        import secrets
        session_id = secrets.token_urlsafe(12)
        with sessions_lock:
            if len(sessions) >= max_sessions:
                oldest = next(iter(sessions))
                sessions.pop(oldest, None)
            sessions[session_id] = {"messages": [], "summary": ""}
        return session_id

    def get_session(session_id: str | None) -> tuple[str, dict[str, Any]]:
        if not session_id:
            session_id = new_session()
        with sessions_lock:
            session = sessions.setdefault(session_id, {"messages": [], "summary": ""})
        return session_id, session

    def build_prompt(session: dict[str, Any], system_prompt: str, prompt: str) -> tuple[str, int]:
        assert runtime is not None
        prefix = f"System: {system_prompt.strip()}\n\n" if system_prompt.strip() else ""
        messages = list(session["messages"])
        summary = session.get("summary", "")
        while messages:
            history = "\n\n".join(f"{item['role'].title()}: {item['content']}" for item in messages)
            candidate = prefix + (f"Conversation summary: {summary}\n\n" if summary else "") + history + f"\n\nUser: {prompt}\n\nAssistant:"
            if len(runtime.tokenizer.encode(candidate, add_bos=False)) <= max(8, context_limit - 8):
                return candidate, len(messages)
            removed = messages.pop(0)
            snippet = removed["content"].replace("\n", " ")[:240]
            summary = (summary + " " + removed["role"] + ": " + snippet).strip()[-1000:]
        candidate = prefix + (f"Conversation summary: {summary}\n\n" if summary else "") + f"User: {prompt}\n\nAssistant:"
        limit = max(8, context_limit - 8)
        if len(runtime.tokenizer.encode(candidate, add_bos=False)) > limit:
            candidate = f"User: {prompt}\n\nAssistant:"
            if len(runtime.tokenizer.encode(candidate, add_bos=False)) > limit:
                candidate = f"User: {prompt[:256]}\n\nAssistant:"
        session["summary"] = summary[-500:]
        return candidate, 0

    def validate(request: dict[str, Any]) -> dict[str, Any]:
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not 1 <= len(prompt) <= 32768:
            raise HTTPException(status_code=422, detail="prompt must be 1-32768 characters")
        values = {"system_prompt": request.get("system_prompt", ""), "max_new_tokens": request.get("max_new_tokens", 32), "temperature": request.get("temperature", 0.0), "top_k": request.get("top_k", 40), "top_p": request.get("top_p", 0.9), "repetition_penalty": request.get("repetition_penalty", 1.05), "seed": request.get("seed", 0), "conversation_id": request.get("conversation_id")}
        if not isinstance(values["system_prompt"], str) or len(values["system_prompt"]) > 8192:
            raise HTTPException(status_code=422, detail="system_prompt must be at most 8192 characters")
        if not isinstance(values["max_new_tokens"], int) or not 1 <= values["max_new_tokens"] <= 4096:
            raise HTTPException(status_code=422, detail="max_new_tokens must be between 1 and 4096")
        if not isinstance(values["temperature"], (int, float)) or not 0 <= float(values["temperature"]) <= 2:
            raise HTTPException(status_code=422, detail="temperature must be between 0 and 2")
        if not isinstance(values["top_k"], int) or not 0 <= values["top_k"] <= 1000:
            raise HTTPException(status_code=422, detail="top_k must be between 0 and 1000")
        if not isinstance(values["top_p"], (int, float)) or not 0 < float(values["top_p"]) <= 1:
            raise HTTPException(status_code=422, detail="top_p must be in (0, 1]")
        if not isinstance(values["repetition_penalty"], (int, float)) or not 1 <= float(values["repetition_penalty"]) <= 2:
            raise HTTPException(status_code=422, detail="repetition_penalty must be between 1 and 2")
        if not isinstance(values["seed"], int):
            raise HTTPException(status_code=422, detail="seed must be an integer")
        return {"prompt": prompt, **values}

    def generate_result(request: dict[str, Any]) -> dict[str, Any]:
        if runtime is None:
            raise HTTPException(status_code=503, detail="model is not ready")
        values = validate(request)
        session_id, session = get_session(values["conversation_id"])
        full_prompt, history_count = build_prompt(session, values["system_prompt"], values["prompt"])
        prompt_tokens = len(runtime.tokenizer.encode(full_prompt, add_bos=False))
        remaining_tokens = runtime.max_context_tokens - prompt_tokens
        if remaining_tokens <= 0:
            raise HTTPException(status_code=400, detail="prompt is too long for the configured context; start a new chat or reduce history")
        # Never ask the runtime to decode beyond the available KV-cache slots.
        values["max_new_tokens"] = min(values["max_new_tokens"], remaining_tokens)
        try:
            # The runtime owns one reusable KV cache. Rebuild it per request because
            # conversation history is reconstructed in the bounded session prompt;
            # this prevents one request from consuming the cache for later requests.
            with generation_lock:
                runtime.reset()
                text = runtime.generate(full_prompt, max_new_tokens=values["max_new_tokens"], temperature=float(values["temperature"]), top_k=values["top_k"], top_p=float(values["top_p"]), repetition_penalty=float(values["repetition_penalty"]), seed=values["seed"])
        except (ValueError, MemoryError, NotImplementedError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        session["messages"].extend([{"role": "user", "content": values["prompt"]}, {"role": "assistant", "content": text}])
        total_tokens = len(runtime.tokenizer.encode(text, add_bos=False))
        return {"text": text, "conversation_id": session_id, "history_messages": history_count, "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": max(0, total_tokens - prompt_tokens), "total_tokens": total_tokens}, "settings": {k: values[k] for k in ("max_new_tokens", "temperature", "top_k", "top_p", "repetition_penalty", "seed")}}

    @app.get("/v1/workspace")
    def workspace_list() -> dict[str, Any]:
        return {"files": workspace.list(), "root": str(workspace.root), "limits": {"max_file_bytes": workspace.MAX_FILE_BYTES if hasattr(workspace, "MAX_FILE_BYTES") else int(os.environ.get("LOWRAM_WORKSPACE_MAX_FILE_MB", "16")) * 1024 * 1024}}

    @app.post("/v1/workspace/file")
    def workspace_write(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return workspace.write(str(request.get("path", "")), str(request.get("content", "")))
        except WorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/workspace/file")
    def workspace_read(path: str) -> dict[str, str]:
        try:
            return {"path": path, "content": workspace.read(path)}
        except WorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/workspace/upload")
    async def workspace_upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        uploaded = []
        try:
            for item in files:
                data = await item.read()
                uploaded.append(workspace.write_bytes(item.filename or "upload.bin", data))
            return {"files": uploaded}
        except WorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/workspace/download")
    def workspace_download(path: str) -> FileResponse:
        try:
            target = workspace.path(path)
            if not target.is_file():
                raise WorkspaceError("file not found")
            return FileResponse(target, filename=target.name)
        except WorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/workspace/zip")
    def workspace_zip(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return workspace.zip(str(request.get("archive", "archive.zip")), request.get("paths", []))
        except WorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/workspace/unzip")
    def workspace_unzip(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return workspace.unzip(str(request.get("archive", "")), str(request.get("destination", ".")))
        except WorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/workspace/python")
    def workspace_python(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return workspace.run_python(str(request.get("code", "")), str(request.get("filename", "main.py")), request.get("timeout"))
        except WorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/conversations")
    def conversations() -> dict[str, str]:
        return {"conversation_id": new_session()}

    @app.delete("/v1/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str) -> dict[str, bool]:
        with sessions_lock:
            sessions.pop(conversation_id, None)
        return {"deleted": True}

    @app.post("/v1/generate")
    def generate(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return generate_result(request)

    @app.post("/v1/generate/stream")
    def generate_stream(request: dict[str, Any] = Body(...)) -> StreamingResponse:
        result = generate_result(request)
        def events() -> Iterator[str]:
            text = result["text"]
            for start in range(0, len(text), 32):
                yield "data: " + json.dumps({"delta": text[start:start + 32]}, ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps({"done": True, "conversation_id": result["conversation_id"], "usage": result["usage"]}) + "\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main() -> None:
    import uvicorn
    model_path = os.environ.get("LOWRAM_MODEL")
    if not model_path:
        raise SystemExit("LOWRAM_MODEL must point to a GGUF model")
    app = create_app(model_path, max_context=int(os.environ.get("LOWRAM_MAX_CONTEXT", "512")), max_ram_mb=int(os.environ["LOWRAM_MAX_RAM_MB"]) if os.environ.get("LOWRAM_MAX_RAM_MB") else None)
    uvicorn.run(app, host=os.environ.get("LOWRAM_HOST", "127.0.0.1"), port=int(os.environ.get("LOWRAM_PORT", "8000")))


if __name__ == "__main__":
    main()

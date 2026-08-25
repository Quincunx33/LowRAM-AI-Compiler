from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile
from typing import Iterable

MAX_FILE_BYTES = int(os.environ.get("LOWRAM_WORKSPACE_MAX_FILE_MB", "16")) * 1024 * 1024
MAX_WORKSPACE_BYTES = int(os.environ.get("LOWRAM_WORKSPACE_MAX_TOTAL_MB", "128")) * 1024 * 1024
MAX_OUTPUT_BYTES = int(os.environ.get("LOWRAM_WORKSPACE_MAX_OUTPUT_KB", "256")) * 1024
MAX_TIMEOUT = max(1, min(30, int(os.environ.get("LOWRAM_WORKSPACE_TIMEOUT", "10"))))


class WorkspaceError(ValueError):
    pass


class Workspace:
    def __init__(self, root: str | None = None):
        self.root = Path(root or os.environ.get("LOWRAM_WORKSPACE", tempfile.mkdtemp(prefix="lowram-workspace-"))).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relative: str) -> Path:
        if not relative or "\x00" in relative:
            raise WorkspaceError("path is required")
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise WorkspaceError("path escapes workspace")
        return candidate

    def size(self) -> int:
        return sum(item.stat().st_size for item in self.root.rglob("*") if item.is_file())

    def ensure_capacity(self, extra: int = 0) -> None:
        if self.size() + extra > MAX_WORKSPACE_BYTES:
            raise WorkspaceError("workspace storage limit exceeded")

    def write_bytes(self, relative: str, data: bytes) -> dict[str, object]:
        if len(data) > MAX_FILE_BYTES:
            raise WorkspaceError("file size limit exceeded")
        target = self.path(relative)
        self.ensure_capacity(len(data))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return {"path": str(target.relative_to(self.root)), "bytes": len(data)}

    def write(self, relative: str, content: str) -> dict[str, object]:
        return self.write_bytes(relative, content.encode("utf-8"))

    def list(self) -> list[dict[str, object]]:
        result = []
        for item in sorted(self.root.rglob("*")):
            result.append({"path": str(item.relative_to(self.root)), "directory": item.is_dir(), "bytes": item.stat().st_size if item.is_file() else 0})
        return result

    def read(self, relative: str) -> str:
        target = self.path(relative)
        if not target.is_file():
            raise WorkspaceError("file not found")
        if target.stat().st_size > MAX_FILE_BYTES:
            raise WorkspaceError("file size limit exceeded")
        return target.read_text(encoding="utf-8", errors="replace")

    def zip(self, archive: str, paths: Iterable[str]) -> dict[str, object]:
        target = self.path(archive)
        selected = [self.path(item) for item in paths]
        if not selected:
            raise WorkspaceError("at least one path is required")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as output:
            for item in selected:
                if not item.exists():
                    raise WorkspaceError(f"path not found: {item.name}")
                if item.is_file():
                    output.write(item, item.relative_to(self.root))
                else:
                    for child in item.rglob("*"):
                        if child.is_file():
                            output.write(child, child.relative_to(self.root))
        return {"path": str(target.relative_to(self.root)), "bytes": target.stat().st_size}

    def unzip(self, archive: str, destination: str = ".") -> dict[str, object]:
        source = self.path(archive)
        target = self.path(destination)
        if not source.is_file() or not zipfile.is_zipfile(source):
            raise WorkspaceError("valid ZIP archive required")
        with zipfile.ZipFile(source) as zipped:
            members = zipped.infolist()
            total = sum(member.file_size for member in members)
            if total > MAX_WORKSPACE_BYTES:
                raise WorkspaceError("uncompressed archive exceeds workspace limit")
            for member in members:
                self.path(str(Path(destination) / member.filename))
            target.mkdir(parents=True, exist_ok=True)
            zipped.extractall(target)
        return {"destination": str(target.relative_to(self.root)), "files": len(members)}

    def run_python(self, code: str, filename: str = "main.py", timeout: int | None = None) -> dict[str, object]:
        if not code.strip():
            raise WorkspaceError("code is required")
        self.write(filename, code)
        target = self.path(filename)
        requested = MAX_TIMEOUT if timeout is None else max(1, min(MAX_TIMEOUT, int(timeout)))
        environment = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            completed = subprocess.run([sys.executable, "-I", str(target)], cwd=self.root, env=environment, capture_output=True, text=True, timeout=requested)
            output = (completed.stdout + completed.stderr)[-MAX_OUTPUT_BYTES:]
            return {"returncode": completed.returncode, "output": output, "timed_out": False, "file": str(target.relative_to(self.root))}
        except subprocess.TimeoutExpired as exc:
            output = ((exc.stdout or "") + (exc.stderr or ""))[-MAX_OUTPUT_BYTES:]
            return {"returncode": -1, "output": output + "\n[execution timed out]", "timed_out": True, "file": str(target.relative_to(self.root))}

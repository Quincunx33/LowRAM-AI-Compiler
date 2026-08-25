"""Small cross-platform memory accounting helpers."""

from __future__ import annotations

import os
from typing import Optional

try:
    import resource
except ImportError:  # pragma: no cover - exercised on Windows only
    resource = None  # type: ignore[assignment]


def current_rss_mb() -> Optional[float]:
    """Return current process RSS when the host exposes it, otherwise None."""
    if resource is not None:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.name == "posix" and __import__("sys").platform == "darwin":
            return value / (1024 * 1024)
        return value / 1024
    if os.name == "nt":  # pragma: no cover - exercised on Windows only
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024 * 1024)
    return None


def enforce_rss_budget(max_ram_mb: int | None) -> None:
    """Raise when measurable process RSS exceeds a configured hard ceiling."""
    if max_ram_mb is None:
        return
    if max_ram_mb <= 0:
        raise ValueError("max_ram_mb must be positive")
    rss = current_rss_mb()
    if rss is not None and rss > max_ram_mb:
        raise MemoryError(f"RSS budget exceeded: {rss:.1f} MiB > {max_ram_mb} MiB")

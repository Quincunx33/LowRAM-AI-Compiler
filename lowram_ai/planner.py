"""Memory planning helpers for low-RAM model deployment.

The planner is deliberately conservative: a 1 GB device is not assumed to have
1 GB available to the model because the operating system and application also
need memory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class BudgetPlan:
    """A deterministic deployment plan for a model under a RAM ceiling."""

    device_ram_mb: int
    reserve_for_os_mb: int
    reserve_for_app_mb: int
    runtime_headroom_mb: int
    model_budget_mb: int
    max_peak_ram_mb: int
    quantization_bits: int
    max_context_tokens: int
    batch_size: int
    estimated_weight_mb: float
    estimated_kv_cache_mb: float
    estimated_peak_mb: float
    fits_budget: bool
    recommendations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["recommendations"] = list(self.recommendations)
        return data


def estimate_weight_mb(parameters: int, bits: int) -> float:
    """Estimate packed weight size, excluding small metadata tables."""
    if parameters < 0:
        raise ValueError("parameters must be non-negative")
    if bits <= 0 or bits > 32:
        raise ValueError("bits must be between 1 and 32")
    return parameters * bits / 8 / BYTES_PER_MB


def estimate_kv_cache_mb(
    *,
    layers: int,
    hidden_size: int,
    context_tokens: int,
    bits: int = 16,
) -> float:
    """Estimate a decoder KV cache for batch size one.

    The factor of two accounts for K and V. This is a planning estimate, not a
    model-format-specific allocator measurement.
    """
    if min(layers, hidden_size, context_tokens) < 0:
        raise ValueError("layers, hidden_size, and context_tokens cannot be negative")
    if bits <= 0 or bits > 32:
        raise ValueError("bits must be between 1 and 32")
    return 2 * layers * context_tokens * hidden_size * bits / 8 / BYTES_PER_MB


def build_budget_plan(
    *,
    device_ram_mb: int,
    parameters: int,
    layers: int,
    hidden_size: int,
    requested_context_tokens: int = 512,
    quantization_bits: int = 4,
    os_reserve_ratio: float = 0.35,
    app_reserve_mb: int = 96,
    runtime_headroom_ratio: float = 0.15,
) -> BudgetPlan:
    """Build a conservative memory plan for a batch-one decoder runtime."""
    if device_ram_mb <= 0 or parameters <= 0 or layers <= 0 or hidden_size <= 0:
        raise ValueError("device_ram_mb, parameters, layers, and hidden_size must be positive")
    if requested_context_tokens <= 0:
        raise ValueError("requested_context_tokens must be positive")
    if quantization_bits not in (2, 3, 4, 8, 16, 32):
        raise ValueError("quantization_bits must be one of 2, 3, 4, 8, 16, or 32")
    if not 0 < os_reserve_ratio < 1:
        raise ValueError("os_reserve_ratio must be between 0 and 1")
    if app_reserve_mb < 0 or not 0 <= runtime_headroom_ratio < 1:
        raise ValueError("reserves must be non-negative and headroom ratio must be below 1")

    reserve_for_os_mb = max(64, round(device_ram_mb * os_reserve_ratio))
    available_after_reserves = device_ram_mb - reserve_for_os_mb - app_reserve_mb
    model_budget_mb = max(64, available_after_reserves)
    runtime_headroom_mb = round(model_budget_mb * runtime_headroom_ratio)
    max_peak_ram_mb = model_budget_mb + reserve_for_os_mb + app_reserve_mb

    weight_mb = estimate_weight_mb(parameters, quantization_bits)
    # Reserve a small activation scratch area proportional to hidden size.
    scratch_mb = max(8.0, hidden_size * 4 / BYTES_PER_MB)
    context = requested_context_tokens
    kv_mb = estimate_kv_cache_mb(
        layers=layers,
        hidden_size=hidden_size,
        context_tokens=context,
        bits=16,
    )
    peak_mb = weight_mb + kv_mb + scratch_mb + runtime_headroom_mb
    recommendations: list[str] = []

    if peak_mb > model_budget_mb:
        recommendations.append("Reduce context length or enable KV-cache quantization.")
    if weight_mb > model_budget_mb * 0.8:
        recommendations.append("Use a smaller/distilled model; weights dominate the RAM budget.")
    if quantization_bits > 4:
        recommendations.append("Try 4-bit or 3-bit weight quantization before increasing context.")
    if context > 512:
        recommendations.append("Keep the first mobile profile at 256–512 context tokens.")
    if not recommendations:
        recommendations.append("Profile peak RSS on the target device before release.")

    return BudgetPlan(
        device_ram_mb=device_ram_mb,
        reserve_for_os_mb=reserve_for_os_mb,
        reserve_for_app_mb=app_reserve_mb,
        runtime_headroom_mb=runtime_headroom_mb,
        model_budget_mb=model_budget_mb,
        max_peak_ram_mb=max_peak_ram_mb,
        quantization_bits=quantization_bits,
        max_context_tokens=context,
        batch_size=1,
        estimated_weight_mb=round(weight_mb, 2),
        estimated_kv_cache_mb=round(kv_mb, 2),
        estimated_peak_mb=round(peak_mb, 2),
        fits_budget=peak_mb <= model_budget_mb,
        recommendations=tuple(recommendations),
    )

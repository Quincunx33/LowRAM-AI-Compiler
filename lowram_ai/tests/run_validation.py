from pathlib import Path

import numpy as np

from lowram_ai.planner import build_budget_plan
from lowram_ai.quantized import QuantizedMatrix, quantize_npy_matrix


def main() -> None:
    workdir = Path("/tmp/lowram-validation")
    workdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4)
    source = rng.normal(size=(9, 17)).astype(np.float32)
    source_path = workdir / "source.npy"
    output_path = workdir / "weights.lrq"
    np.save(source_path, source)
    info = quantize_npy_matrix(source_path, output_path, bits=4, group_size=8, chunk_rows=2)
    vector = rng.normal(size=17).astype(np.float32)
    expected = source @ vector
    with QuantizedMatrix(output_path) as matrix:
        actual = matrix.matvec(vector)
    mae = float(np.mean(np.abs(expected - actual)))
    assert mae < 0.5, mae
    assert info["output_bytes"] < info["input_bytes"]

    odd_source = np.arange(15, dtype=np.float32).reshape(3, 5)
    odd_path = workdir / "odd.npy"
    odd_output = workdir / "odd.lrq"
    np.save(odd_path, odd_source)
    quantize_npy_matrix(odd_path, odd_output, bits=4, group_size=4)
    with QuantizedMatrix(odd_output) as matrix:
        odd_result = matrix.matvec(np.ones(5, dtype=np.float32))
    assert odd_result.shape == (3,)

    plan = build_budget_plan(
        device_ram_mb=1024,
        parameters=500_000_000,
        layers=16,
        hidden_size=2048,
        requested_context_tokens=256,
        quantization_bits=4,
    )
    assert plan.fits_budget is True
    print({"quantized_mae": round(mae, 6), "quantized_info": info, "plan": plan.to_dict()})


if __name__ == "__main__":
    main()

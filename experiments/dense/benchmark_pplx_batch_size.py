#!/usr/bin/env python3
"""Small offline PPLX batch-size benchmark on real fixed_1200_0 chunks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.chunk.run_bm25_parameter_grid import CONFIGS  # noqa: E402
from experiments.dense.run_pplx_dense_train_chunk_grid import (  # noqa: E402
    CORPUS,
    iter_model_inputs,
)
from experiments.dense.run_pplx_dense_train_chunk_config import (  # noqa: E402
    DIMENSION,
    load_model,
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments/dense/results/pplx_batch_benchmark_fixed_1200_0.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=480)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 24, 32])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.samples <= 0 or any(value <= 0 for value in args.batch_sizes):
        parser.error("samples and batch sizes must be positive")
    if any(args.samples % value for value in args.batch_sizes):
        parser.error("--samples must be divisible by every batch size")
    return args


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    config = CONFIGS["fixed_1200_0"]
    texts: list[str] = []
    for metadata, model_input in iter_model_inputs(CORPUS, config):
        if metadata["body_length"] == 1200:
            texts.append(model_input)
            if len(texts) == args.samples:
                break
    if len(texts) != args.samples:
        raise ValueError(f"found only {len(texts)} full-length chunks")

    model = load_model()
    with torch.inference_mode():
        model.encode(
            texts[:16],
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    torch.cuda.synchronize()

    results: list[dict[str, object]] = []
    for batch_size in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        completed = 0
        try:
            with torch.inference_mode():
                for start in range(0, len(texts), batch_size):
                    vectors = model.encode(
                        texts[start : start + batch_size],
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )
                    if vectors.shape != (batch_size, DIMENSION):
                        raise ValueError(f"unexpected output shape: {vectors.shape}")
                    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
                        raise ValueError("invalid PPLX output")
                    completed += len(vectors)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            row = {
                "batch_size": batch_size,
                "status": "passed",
                "samples": completed,
                "seconds": seconds,
                "chunks_per_second": completed / seconds,
                "peak_gpu_memory_allocated_gib": torch.cuda.max_memory_allocated()
                / (1024**3),
                "peak_gpu_memory_reserved_gib": torch.cuda.max_memory_reserved()
                / (1024**3),
            }
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        except torch.cuda.OutOfMemoryError as error:
            row = {
                "batch_size": batch_size,
                "status": "oom",
                "samples_completed": completed,
                "error": str(error),
            }
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            torch.cuda.empty_cache()
            break

    output = {
        "status": "complete",
        "chunk_config": "fixed_1200_0",
        "sample_rule": "first full-length 1200-character chunks in corpus order",
        "same_samples_for_all_batch_sizes": True,
        "warmup_samples": 16,
        "dtype": "float32",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

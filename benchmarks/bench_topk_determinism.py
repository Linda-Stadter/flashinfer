import os

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

BATCH_SIZES = [4096, 8192, 16384]
VOCAB_SIZE = 200000
K_VALUES = [128, 129, 256, 257, 512, 513]
ELEM_SIZE = 4  # fp32

ALGOS = ["filtered"]


def bench_one(fn):
    times = bench_gpu_time(fn, enable_cupti=True, dry_run_iters=10, repeat_iters=100)
    return float(np.median(times))


def input_gbps(time_ms, batch_size):
    return (batch_size * VOCAB_SIZE * ELEM_SIZE) / (time_ms * 1e-3) / 1e9


def fmt_overhead(test_ms, base_ms):
    if base_ms == 0:
        return "    N/A"
    pct = (test_ms / base_ms - 1) * 100
    return f"{pct:>+7.1f}%"


def print_row(
    k, nd_ms, nd_gbps, d_ms, d_gbps, oh, nd_s_ms, nd_s_gbps, d_s_ms, d_s_gbps, oh_s
):
    print(
        f"  {k:>4} |"
        f"  {nd_ms:>7.3f}ms {nd_gbps:>7.1f}"
        f"  {d_ms:>7.3f}ms {d_gbps:>7.1f}"
        f"  {oh:>8}"
        f"  |"
        f"  {nd_s_ms:>7.3f}ms {nd_s_gbps:>7.1f}"
        f"  {d_s_ms:>7.3f}ms {d_s_gbps:>7.1f}"
        f"  {oh_s:>8}"
    )


@torch.inference_mode()
def main():
    dtype = torch.float32

    for algo in ALGOS:
        if algo == "auto":
            os.environ.pop("FLASHINFER_TOPK_ALGO", None)
        else:
            os.environ["FLASHINFER_TOPK_ALGO"] = algo

        for batch_size in BATCH_SIZES:
            scores = torch.randn(batch_size, VOCAB_SIZE, device="cuda", dtype=dtype)

            print(f"Algorithm: {algo}")
            print(f"  batch_size={batch_size}  vocab_size={VOCAB_SIZE}  dtype=fp32")
            print()

            half = 48
            print(f"       |{'--- Unsorted ---':^{half}} |{'--- Sorted ---':^{half}}")
            print(
                f"     k |"
                f"  {'NonDet':>9} {'GB/s':>7}"
                f"  {'Det':>9} {'GB/s':>7}"
                f"  {'overhead':>8}"
                f"  |"
                f"  {'NonDet':>9} {'GB/s':>7}"
                f"  {'Det':>9} {'GB/s':>7}"
                f"  {'overhead':>8}"
            )
            print("-" * (7 + 1 + half + 2 + 1 + half))

            for k in K_VALUES:
                os.environ["FLASHINFER_DETERMINISTIC_TOPK"] = "0"
                nd_unsorted = bench_one(
                    lambda k=k, s=scores: flashinfer.top_k(s, k, sorted=False)
                )
                nd_sorted = bench_one(
                    lambda k=k, s=scores: flashinfer.top_k(s, k, sorted=True)
                )

                os.environ["FLASHINFER_DETERMINISTIC_TOPK"] = "1"
                d_unsorted = bench_one(
                    lambda k=k, s=scores: flashinfer.top_k(s, k, sorted=False)
                )
                d_sorted = bench_one(
                    lambda k=k, s=scores: flashinfer.top_k(s, k, sorted=True)
                )

                print_row(
                    k,
                    nd_unsorted,
                    input_gbps(nd_unsorted, batch_size),
                    d_unsorted,
                    input_gbps(d_unsorted, batch_size),
                    fmt_overhead(d_unsorted, nd_unsorted),
                    nd_sorted,
                    input_gbps(nd_sorted, batch_size),
                    d_sorted,
                    input_gbps(d_sorted, batch_size),
                    fmt_overhead(d_sorted, nd_sorted),
                )

            del scores
            print()

    os.environ.pop("FLASHINFER_DETERMINISTIC_TOPK", None)
    os.environ.pop("FLASHINFER_TOPK_ALGO", None)


if __name__ == "__main__":
    main()

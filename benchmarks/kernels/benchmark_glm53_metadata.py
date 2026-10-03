# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure first-use and warm GLM metadata latency across prompt remainders.

Run each source revision in an isolated process with an empty TRITON_CACHE_DIR.
The one-request metadata calculation is checked exactly against its reference.
"""

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import torch

from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    B12xGLM5NextMLASparseMetadataBuilder,
    _glm_device_token_metadata_kernel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="521,547,569,593,617,641,673,701,733,769")
    args = parser.parse_args()
    counts = [1, *map(int, args.lengths.split(","))]
    device = torch.device("cuda", torch.accelerator.current_device_index())
    builder = object.__new__(B12xGLM5NextMLASparseMetadataBuilder)
    builder.dcp_world_size = 1
    builder.dcp_rank = 0
    builder.cp_kv_cache_interleave_size = 1
    builder.req_id_per_token_buffer = torch.empty(
        max(counts), dtype=torch.int32, device=device
    )
    builder.cache_seq_lens_per_token_buffer = torch.empty_like(
        builder.req_id_per_token_buffer
    )
    starts = torch.zeros(2, dtype=torch.int32, device=device)
    lengths = torch.zeros(1, dtype=torch.int32, device=device)
    cache = _glm_device_token_metadata_kernel.device_caches[device.index][0]
    cache.clear()
    rows = []
    for repetition in range(2):
        for count in counts:
            starts[1] = count
            lengths[0] = 8192 + count
            common = SimpleNamespace(
                num_actual_tokens=count,
                num_reqs=1,
                query_start_loc=starts,
                seq_lens=lengths,
            )
            torch.accelerator.synchronize()
            start = time.perf_counter()
            request_ids = builder._build_req_id_per_token(common)
            torch.accelerator.synchronize()
            elapsed = time.perf_counter() - start
            expected = torch.arange(
                8193, 8193 + count, dtype=torch.int32, device=device
            )
            assert torch.equal(request_ids, torch.zeros_like(request_ids))
            assert torch.equal(
                builder.cache_seq_lens_per_token_buffer[:count], expected
            )
            rows.append(
                {
                    "repetition": repetition,
                    "tokens": count,
                    "wall_ms": elapsed * 1000,
                    "compiled_variants": len(cache),
                }
            )
    accelerator = torch.get_device_module(device)
    graph = accelerator.CUDAGraph()
    with accelerator.graph(graph):
        for _ in range(300):
            builder._build_req_id_per_token(common)
    times = []
    for _ in range(5):
        start = accelerator.Event(enable_timing=True)
        end = accelerator.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / 300)
    print(
        json.dumps(
            {
                "rows": rows,
                "graph_kernel_us": times,
                "median_graph_kernel_us": statistics.median(times),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

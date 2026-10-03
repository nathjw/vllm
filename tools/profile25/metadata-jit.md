# GLM metadata: reuse one kernel across prompt remainders

## Change

`_glm_device_token_metadata_kernel` used a compile-time `NUM_TOKENS` even
though it only bounds the final tile. Unseen prompt remainders generated
additional Triton binaries during inference. The count is now a runtime
integer with `do_not_specialize`; request count, search depth, tile size and
DCP geometry retain their existing specialization.

This changes kernel compilation reuse. The request IDs, causal lengths,
attention algorithm, weights, precision and speculative verification policy
remain the same. Full-model serving qualification is in progress; this
document currently reports kernel-level evidence only.

## GPU correctness

Tested in the exact ARM64 R28.8-A image recorded in `baseline.json`, on GB10:

- The regression test fails on the unchanged source: the second nonempty
  prompt length creates a second compiled variant.
- Patched code reuses one variant across 15 lengths, including zero, one,
  alignment boundaries and a partial final tile. Request IDs and causal
  lengths match exactly; output outside the valid range remains untouched.
- All six existing adaptive/mixed-prefill/DCP CUDA graph tests pass.
- Total: **7 passed** (64 unrelated tests deselected).

The test file was mounted as `/tmp/test_indexer.py` in the pinned image,
with isolated pytest dependencies and `B12X_GLM53_GPU_TEST=1`:

```sh
/opt/venv/bin/python -m pytest -q /tmp/test_indexer.py \
  -k 'metadata_reuses_jit or adaptive_sparse_metadata'
```

## Isolated latency

`benchmarks/kernels/benchmark_glm53_metadata.py` checks results against an
exact reference while measuring cold and warm calls. Each revision ran in
a separate pinned container with an empty `TRITON_CACHE_DIR`.

| Measurement | Original | Patched |
| --- | ---: | ---: |
| Compiled variants: warmup + 10 unseen lengths | 11 | 1 |
| Median first-use operation, excluding initial warmup | 30.674 ms | 0.013 ms |
| Median warm host call | 0.0155 ms | 0.0122 ms |
| Median CUDA graph kernel time | 0.819 us | 0.806 us |

The approximately 31 ms saving concerns one metadata operation for a new
shape. It is not a 31 ms saving per token or a measured steady TG speedup.
`metadata-microbenchmark.json` contains the individual observations.

## Static checks

The applicable pre-commit checks ran. Mypy reports five pre-existing errors
in this file (two `seq_lens_cpu` attributes, two prepared-state attributes,
and one optional value indexing error). Running the same hook against the
unmodified `a77e56d` source reproduces all five; the patch adds none. Only
that known failing hook is skipped for the commit. Other hooks are retained.

AI assistance was used for implementation, testing and analysis. No upstream
PR has been opened.

# GLM metadata: reuse one kernel across prompt remainders

## Change

`_glm_device_token_metadata_kernel` used a compile-time `NUM_TOKENS` even
though it only bounds the final tile. Unseen prompt remainders generated
additional Triton binaries during inference. The count is now a runtime
integer with `do_not_specialize`; request count, search depth, tile size and
DCP geometry retain their existing specialization.

This changes kernel compilation reuse. The request IDs, causal lengths,
attention algorithm, weights, precision and speculative verification policy
remain the same. Kernel tests, full-model timing, coding evaluation and
bounded concurrency/long-prompt checks have completed, including a fresh
stock coding control. This remains an opt-in experiment; stock was restored.

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

## Full-model serving

Separate server starts showed larger timing shifts than this small operation,
so a temporary worker extension alternated the two kernels in one process.
The patched kernel's AST matched this commit. The original model, allocations,
other kernels and settings stayed fixed. The temporary control API was bound
to localhost and removed afterwards.

After warming both variants, six 8K requests per variant followed ABBA order.
Twelve unseen lengths (523–859 tokens) each had a first and repeat request per
variant, alternating ABBA and BAAB. Inputs were fixed token IDs, temperature 0,
seed 42, 64 output tokens, distinct cache salts, and zero cached tokens.

| Measurement | Original | Patched |
| --- | ---: | ---: |
| Warm 8K median TTFT | 3.771 s | 3.781 s |
| Warm 8K range | 3.763–3.790 s | 3.764–3.798 s |
| Median first-use minus repeat TTFT | 47.386 ms | 1.767 ms |
| Mean first-use minus repeat TTFT | 49.813 ms | 0.088 ms |

All 62 requests, including warmups, completed. The first-use stall is removed;
warm 8K medians differ by 0.26%, with overlapping ranges. This is a bounded
first-use PP improvement. No steady TG gain is established. Other kernels can
still compile for new shapes. `metadata-serving-ab.json` records the pairs.

In a separate fresh-cache serving sweep, the stock metadata cache gained ten
binaries for ten previously unseen lengths, versus zero with the patch.

## Coding and stability

SWE-Sharp executor-v2: three tasks, two repetitions, brief plans, C# LSP,
requested effort high, one bounded correction and blinded gpt-5.6-sol/xhigh
review. The candidate resolved 6/6 final attempts and scored 19.50/20. All
20 fail-to-pass and both pass-to-pass checks passed. There were no agent API
errors, timeouts, nonzero exits, empty-completion recoveries or judge errors.

All first-pass machine grades passed, but three attempts needed a reviewer
correction: null-description semantics in both task-530 repetitions, a fixture
ID assumption in the second, and XML ordering in task-918 repetition 1. This
is not six uncorrected successes. The historical stock run also resolved 6/6
and scored 19.50/20, with one correction.

The fresh stock control used the same current harness and the original
launcher, with image/command/environment/source checked against the captured
baseline on both nodes. It resolved 6/6 and scored **19.83/20**, using two
corrections, compared with the candidate's **19.50/20** and three corrections.
All first-pass machine grades passed in both runs. Stock needed corrections
for null-description semantics (530/r1) and a missing project dependency
(918/r2). Neither run had agent errors, timeouts or empty recoveries.

These are small samples, with separate normal boots and different generated
trajectories. They do not establish causation or quality non-inferiority. The
shorter candidate agent wall time (759.477 vs 1,232.275 s) is not evidence of
a kernel speedup: it generated 21,917 tokens versus stock's 38,295. Keep
stock as the default; retain this numerically exact metadata change for
opt-in evaluation. `metadata-quality.json` records the scored aggregates.

Two four-request 32K trials and one 128K single-request trial completed with
128 output tokens per request and no cached prompt tokens. The 128K request
had 60.571 s TTFT and 41.17 tok/s delivered generation. Both containers had
zero restarts and no OOM, CUDA error, request failure or collective timeout.
These are stress results, not paired throughput comparisons. This bounded
qualification does not replace a 4–5 hour production soak or establish
statistical quality equivalence.

## Static checks

The applicable pre-commit checks ran. Mypy reports five pre-existing errors
in this file (two `seq_lens_cpu` attributes, two prepared-state attributes,
and one optional value indexing error). Running the same hook against the
unmodified `a77e56d` source reproduces all five; the patch adds none. Only
that known failing hook is skipped for the commit. Other hooks are retained.

AI assistance was used for implementation, testing and analysis. No upstream
PR has been opened.

# Striped L2 prefetch experiment: rejected for profile 25

2026-10-03, two GB10 nodes, R28.8-A image and NVFP4-Spark model pinned by
`baseline.json`. This is an opt-in research branch, not a recommended setting.

## Idea

TensorFold's [GLM patch 0046](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/blob/4bbf2f3d83f8b5363832368021c5b56b4ced3d92/patches/0046-glm-l2-prefetch.patch)
distributes a partial prefetch across weight chunks. That patch credits
[Jay Leaton's patch 0460](https://github.com/jayleaton/glm53-tensorfold-spark),
Apache-2.0. The implementation here uses generic aligned tensor regions, not
TensorFold's EXL3/Q4 storage geometry. No inference or quantization kernels are
ported.

The new opt-in `VLLM_GLM53_L2_PREFETCH_STRIPES=64` distributes each partial
weight fill across 64 regions. The host splits these into independent 4 KiB
pieces and a CuTe kernel processes them in parallel. Later windows receive
disjoint remaining ranges. The default of 1 retains the original planner and
kernel path. Prefetch remains disabled by default on SM121.

## Full-model result

Three cold trials per prompt, concurrency 1, 256 generated tokens, identical
saved prompt IDs, temperature 0. The target verifier and all weights/precision
remain unchanged. MTP-normalized speed is draft rounds / server decode time.

| Prompt | Baseline steps/s | Striped steps/s | Change |
| --- | ---: | ---: | ---: |
| 2,048 | 13.926 | 13.487 | -3.1% |
| 8,192 | 14.092 | 13.445 | -4.6% |
| 32,768 | 13.968 | 13.300 | -4.8% |

Medians are shown. The existing contiguous prefetch implementation with the
same budgets was approximately neutral. The striped variant was rolled back.
Raw delivered token rates sometimes increased because more draft tokens were
accepted; that does not establish a faster implementation. Cold greedy output
already varies in the stock baseline, so byte-identical cold model responses
cannot serve as a quality gate.

Settings used on both ranks:

```sh
VLLM_GLM53_L2_PREFETCH=1
VLLM_GLM53_L2_PREFETCH_STRIPES=64
VLLM_GLM53_L2_PREFETCH_BUDGET_A_MB=8
VLLM_GLM53_L2_PREFETCH_BUDGET_B_MB=12
VLLM_GLM53_L2_PREFETCH_BUDGET_C_MB=6
VLLM_GLM53_L2_PREFETCH_BUDGET_A_MLA_MB=12
```

## Checks

The isolated pinned GPU image ran the updated
`tests/models/test_glm5next_l2_prefetch_persist.py`: **41 passed, 1 skipped**.
Coverage includes byte-range bounds, budget limits, disjoint later windows,
tail alignment, successful native compilation, side-stream CUDA graph replay,
unchanged weight bytes, and exact projection output equality. The skipped
existing smoke test requires SM120 hardware; these nodes are SM121. All applicable
pre-commit checks passed. AI assistance was used for implementation and analysis.

These checks establish the tested cache-hint behavior. This rejected candidate
has not undergone a long agentic coding qualification and is not deployed in
the normal profile 25 launcher.

# Profile 25: pinned Technigma baseline

This branch starts from local-inference-lab/vllm at
`22476af54c637cbb7c7d8193addd160da83a5ce3` and preserves the eight Python files
that differ in the installed Technigma R28.8-A image. `baseline.json` records
the image digest and file hashes. No upstream fixes are dropped.

The image supplies its exact ARM64 native extensions, B12X, FlashInfer, NCCL,
and separately installed Display-KV allocator. This checkout alone is not a
replacement for that complete runtime. The baseline image retains its original
notices and dependency licenses.

Performance work is developed on a separate branch and evaluated using the
unchanged NVFP4-Spark checkpoint, MTP3 target verification and FP8 KV format.

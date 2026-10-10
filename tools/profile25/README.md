# Profile 25: pinned Technigma baseline

The maintained optional bundle is on `profile25/proven-fixes`. Its
[included fixes, evidence, deployment contract and maintenance rules](proven-fixes.md)
and [runtime manifest](proven-fixes.json) combine the qualified metadata
fix with exact GLM history preservation. Launchers pin an immutable commit;
experimental performance settings are excluded.

The October 10 refresh starts from the supported upstream branch
`integration/karmic-kraken-beta` at
`19f2c20ed4d64ae22bd009ceacadd85e2a81e354`, carrying the eight vendor
compatibility adaptations and the runtime-token metadata fix. The GLM parser
history repair is now inherited from upstream. `baseline.json` retains the
original vendor image evidence; the schema-2 release manifest identifies the
complete refreshed image and hashes every installed source/native file.

The derived image preserves the original ARM64 native extensions, CUDA,
PyTorch, NCCL and Display-KV allocator, and updates FlashInfer/B12X and CuTe
dependencies to the supported integration versions. This checkout alone is
not a replacement for that complete runtime. The image retains the original
notices and dependency licenses.

Performance work is developed on a separate branch and evaluated using the
unchanged NVFP4-Spark checkpoint, MTP3 target verification and FP8 KV format.

On `profile25/gb10-perf`, the only additional inference change is the metadata
kernel specialization fix at `88ea13c`. See [the measurements and qualification
record](metadata-jit.md). It removes first-use compilation stalls without
changing model arithmetic. It does not establish a sustained TG speedup.

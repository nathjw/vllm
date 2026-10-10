# Profile 25 maintained fixes

`profile25/proven-fixes` is the maintained branch for the optional profile 25
launcher. It combines the qualified metadata compilation fix and GLM exact
history preservation, using the existing model and runtime. Launchers select
an immutable commit from this branch and verify the accompanying
[overlay manifest](proven-fixes.json).

The October 10 refresh re-stacks the vendor compatibility changes and metadata
fix onto `integration/karmic-kraken-beta` at
`19f2c20ed4d64ae22bd009ceacadd85e2a81e354`. That upstream already contains the
exact parser/history repair and regression tests. The verbatim assistant
template is preserved. The original release
`6328df0da1ba99f8b84d334217ef061e491aa186` remains an ancestor and is preserved
under an archive ref; historical experiment branches remain available.

The fork's literal `main` at `47ccf6c57d92f03630ebcbad3809450545825488` is
dated August 30 and lacks GLM5Next/B12X model support. The supported integration
branch above is dated October 8 and is the current source base for this runtime.
It is the compatible upstream target; literal `main` cannot run this model.

## Included fixes and evidence

| Fix | Recorded evidence | Practical limit |
| --- | --- | --- |
| Runtime token count for GLM sparse attention metadata | Seven GPU regression/graph tests passed; exact IDs and causal lengths across boundaries; same-process first-use minus repeat TTFT median 47.386 ms to 1.767 ms | Removes a first-use compilation stall; warm 8K timings overlap and no sustained TG gain is established |
| GLM parser plus verbatim assistant-content template | Candidate 265 CPU tests passed; stock 219 passed and 46 selected whitespace cases failed; actual-template streaming/non-streaming API round trips covered | Repairs whitespace loss that can invalidate response checkpoints; natural benefit depends on outputs and client history handling |

The metadata candidate and fresh stock coding control each resolved 6/6 final
SWE-Sharp attempts. Candidate score was 19.50/20 with three reviewer corrections;
stock scored 19.83/20 with two. These small separate-start samples establish
neither statistical quality equivalence nor a causal quality difference.
Bounded concurrency/128K checks completed; a four-to-five-hour production soak
has not been established by this qualification.

Natural history checks preserved all generated history in **6/6 pairs on both
stock and candidate**, with 12/12 completed turns per variant. Both processed
exactly **370 new follow-up prompt tokens** in aggregate. No natural cache-speed
gain was demonstrated. The history candidate used B12X draft experts while
stock used Marlin, so their latency cannot isolate a parser effect. This
maintained launcher retains stock Marlin drafting.

Detailed receipts:

- [Metadata implementation, performance, coding and stability](metadata-jit.md).
- [Metadata same-process serving measurements](metadata-serving-ab.json).
- [Metadata coding aggregates](metadata-quality.json).
- [History implementation, CPU/live checks, attribution and mount instructions](history-preservation.md).
- [Sanitized natural history records](history-live-summary.json).

The parser preserves generated whitespace intentionally. Clients need to
retain assistant content and reasoning. Whitespace-only content before a tool
call may be returned as a string instead of null. Tool-argument JSON
reserialization and whitespace between/after tool calls remain separate
representation limits. The companion template is required for assistant
content; parser changes alone cannot repair stripping performed at rendering.

## Immutable runtime contract

Use this registry reference:

```text
technigmaai/glm-5.3-flash-nvfp4-2x-dgx-sparks@sha256:1169f797539454e3c286557d49fddd488488957d9a3f10638b01052998370622
```

The original qualified ARM64 local image ID is:

```text
sha256:a4dc62d34e6f28dbbeef6f0f233e93198cb6f7978694c09fa2648e19c2b9b48b
```

The model is `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` at revision
`a608241037e4c2565356bff7ca293f2133888f88`. The image supplies its existing
native extensions, CUDA 13.4, NVIDIA PyTorch 2.14, communication stack and
Display-KV. The new schema-2 manifest selects a separate immutable runtime
image with the complete refreshed vLLM Python source. Its native source,
CMake/build inputs and common dependency specification are unchanged from
`22476af54c637cbb7c7d8193addd160da83a5ce3`; this preserves the vendor native
ABI instead of resolving vanilla PyPI PyTorch. [baseline.json](baseline.json)
records the original Technigma compatibility adaptations.

FlashInfer/B12X comes from the supported integration fork at
`b4c35e6ec1d712f6b832679c08ce5719fae57cd6`, using its hash-verified
`0.7.1+lil.cu134.sm120.gb4c35e6ec1d7` platform-independent Python wheel.
The ARM64 runtime lock specifies the same CUDA 13.4/NVIDIA PyTorch 2.14 ABI and
CuTe DSL 4.7.1. SM120/x86 precompiled JIT-cache wheels are deliberately not used
on SM121/ARM64. Runtime kernel compilation uses the retained SM121 settings.
The matching CuTe 4.7.1 library packages, Quack 0.6.5 and required dependency
updates are recorded in the companion image-build receipt.

The refreshed runtime image ID is
`sha256:c81dc93312d7d68fa46233eb052869881c9822b9df386078cdf5242bed354db3`.
Its source is `760d2d817ed43591cf546921e247f2f1d4156828`, with vLLM tree
`65b587fe071bebca51f858d10b4746e300acb457`. Package version
`0.1.dev1+g760d2d817ed4.profile25` preserves the actual publisher's
`0.1.dev1+ga7b41c45a.d20260926.cu134` version prefix and replaces only stale
build metadata; it does not claim a vanilla vLLM release number. The complete
source hashes, not the version string alone, identify the deployed code.

The [refresh receipt](upstream-refresh-20261010.json) records 265 passing
CPU parser/serving tests and successful parsing of both stock rank commands.
These checks expose no GPUs and load no model weights. Fresh two-node startup,
live history and hard tool-eval evidence are recorded separately; the earlier
performance/quality numbers above are not results of this upstream refresh.

Schema 2 installs the full source tree in its immutable image and mounts only
the exact verbatim template read-only on both nodes. The manifest records
the source revision/tree, all committed source SHA256 hashes, preserved native
extension hashes and immutable image ID. Preflight reads and verifies the
actual installed files in CPU-only containers, as well as source labels.

| Repository source | Container target |
| --- | --- |
| `tools/profile25/chat_template_verbatim.jinja` | `/opt/glm53/chat_template.jinja` |

Retain `--chat-template /opt/glm53/chat_template.jinja` and the stock recipe's
model, precision, standard verification, MTP3 policy, scheduler, graph ladder,
cache allocation and communication settings. There are no experimental
performance settings in this manifest. The launcher reads committed bytes,
verifies their hashes, and mounts the template into the pinned full-source
image. It uses caches scoped by source-tree ID to retain the old cache state.
The helper still supports the original schema-1 seven-file overlay manifest
when rolling back to the previous source pin and image.

The maintained companion launcher and its combined runtime validation are
owned by the
[dgx-spark-agent-coders repository](https://github.com/nathjw/dgx-spark-agent-coders).
That repository keeps the original stock launcher alongside the optional
maintained launcher. Its release receipt records the selected fork commit,
both nodes' resolved configuration and combined deployment smoke checks.

## Maintenance and promotion rules

1. Keep the public branch history intact. Every launcher release pins a full
   commit ID; advancing the branch alone does not change an installed launcher.
2. Validate the immutable image, checkpoint revision, exact source/target
   allowlist, source hashes and both nodes' mounted hashes before serving.
   Update the manifest whenever included file bytes change.
3. Preserve upstream authorship, implementation tests and evidence. A new
   source-base/image/dependency version receives its own compatibility and
   numerical qualification rather than inheriting the old runtime claim.
4. Promote fixes after focused regression tests, applicable lint/type checks,
   combined API startup/history checks and representative workload evaluation.
   Performance claims need controlled end-to-end measurements with cache,
   sampling, concurrency and startup confounds recorded.
5. Retain a known immutable launcher/source pair for rollback. Record adverse
   results and unresolved limitations with the change. Re-run longer coding
   and stability checks when changing inference behavior or dependencies.
6. Keep experimental tuning on separate branches or explicit experiment
   launchers until its performance, quality and stability gates are met.
   Preserve target quantization and stock standard verification for this
   maintained bundle.

## Research roadmap

The October 4 investigation found no additional setting ready for promotion.
Its compact reports are maintained in the companion repository under
`benchmark/results/2026-10-04-profile-25-deep-perf.md` and the adjacent result
directory. The most useful next steps are:

| Priority | Investigation | Evidence needed before promotion |
| --- | --- | --- |
| 1 | Qualify probabilistic MTP proposals | Initial C1 temperature-1 medians were +15.5% at 8K and +12.5% at 32K, with trajectory and separate-boot graph-allocation confounds. Repeat balanced controls, real client sampling, C4/temperature-zero checks, coding evaluation and extended soak. Keep standard rejection. |
| 2 | Remove further dynamic-count metadata specializations | Inspect installed functions against [upstream PR 914](https://github.com/local-inference-lab/vllm/pull/914); prove exact boundary behavior and measure new-shape stalls separately from warm throughput. |
| 3 | Revisit batching with cache geometry fixed | Earlier 8192-token batching also changed resolved recurrent/target block geometry. Hold both blocks fixed and measure mixed PP/decode latency as well as isolated PP. |
| 4 | Evaluate target NVFP4 weights with BF16 activations separately | Existing path was audited and focused tests prepared, but no GPU result was obtained. It changes activation arithmetic and needs its own numerical, model-quality, memory and full-model qualification. |
| 5 | Cost-aware MTP depth and exact small concurrency graph sizes | Existing adaptive depth disables fused multi-step draft graphs. Price proposal/verification cost and acceptance together; qualify all cache/rollback/finish paths before changing scheduling. |

Completed screens guide where to spend effort: larger prefill graphs did not
improve 8K/32K PP and slowed 1K; larger KDA windows regressed full chunks;
B12X draft experts did not improve verification rate; two faster dense
microkernels offered a combined optimistic TG bound below 0.7%. These settings
remain outside this bundle. The runtime is already extensively tuned.

Original history implementation/tests are attributed to Martin Vit with
Claude assistance upstream. TechnigmaAI's template provenance is preserved in
the history document. Consolidation and qualification were performed by
Nathan W with OpenAI Codex assistance. This maintained fork branch does not
open an upstream PR or change the repository's default branch.

# Profile 25 maintained fixes

`profile25/proven-fixes` is the maintained branch for the optional profile 25
launcher. It combines the qualified metadata compilation fix and GLM exact
history preservation, using the existing model and runtime. Launchers select
an immutable commit from this branch and verify the accompanying
[overlay manifest](proven-fixes.json).

The branch starts from `77af8000d2541b161d5d31365ea8af889f0a65ba`, which
already contains metadata commit `88ea13c933df1894db6cd03fb080cdace668be1d`.
The consolidation introduces a manifest and documentation only. Its six
runtime Python files and exact candidate template retain their previously
qualified bytes. Historical experiment branches remain available as evidence.

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

The qualified ARM64 local image ID is:

```text
sha256:a4dc62d34e6f28dbbeef6f0f233e93198cb6f7978694c09fa2648e19c2b9b48b
```

The model is `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` at revision
`a608241037e4c2565356bff7ca293f2133888f88`. The image supplies its existing
native extensions, B12X, FlashInfer, communication stack, installed Python
overlays and Display-KV. The source lineage is
`22476af54c637cbb7c7d8193addd160da83a5ce3`; [baseline.json](baseline.json)
records the preserved Technigma image overlays.

Apply exactly the following seven read-only file bindings on both nodes.
Here, `SITE` means `/opt/venv/lib/python3.12/site-packages`; every source is
relative to the selected fork commit. The JSON manifest records full absolute
container targets and SHA256 hashes of the source bytes.

| Repository source | Container target |
| --- | --- |
| `vllm/v1/attention/backends/mla/b12x_mla_sparse.py` | `SITE/vllm/v1/attention/backends/mla/b12x_mla_sparse.py` |
| `vllm/parser/abstract_parser.py` | `SITE/vllm/parser/abstract_parser.py` |
| `vllm/parser/engine/adapters.py` | `SITE/vllm/parser/engine/adapters.py` |
| `vllm/parser/engine/parser_engine.py` | `SITE/vllm/parser/engine/parser_engine.py` |
| `vllm/parser/glm47_moe.py` | `SITE/vllm/parser/glm47_moe.py` |
| `vllm/tool_parsers/abstract_tool_parser.py` | `SITE/vllm/tool_parsers/abstract_tool_parser.py` |
| `tools/profile25/chat_template_verbatim.jinja` | `/opt/glm53/chat_template.jinja` |

Retain `--chat-template /opt/glm53/chat_template.jinja` and the stock recipe's
model, precision, standard verification, MTP3 policy, scheduler, graph ladder,
cache allocation and communication settings. There are no experimental
performance settings in this manifest. The launcher reads committed bytes,
verifies their hashes, and mounts individual files into the pinned image.
It preserves unrelated installed packages and runtime files.

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

# Profile 25 exact history preservation

This branch ports [local-inference-lab/vllm PR 932](https://github.com/local-inference-lab/vllm/pull/932)
onto the pinned profile 25 source. Runtime changes and upstream tests are
unchanged from that PR. Upstream commits are `eec23823da4e578a45d6cf18789505d2eeada3a5`,
`b5dedaf16b387fffd86cd3bd3891edb903159bb3` and
`7b2fc4e1db400a9d93e5194b56f47480f804a8c1`.

The GLM parser retains generated reasoning/content whitespace so a client can
return the same history. Content rendering must also retain that whitespace:
the profile 25 candidate template changes both assistant `content.strip()`
expressions to `content`, preserving every other template rule and Jinja
whitespace control. No target weights, quantization or inference arithmetic
are changed. The effect is workload-dependent prefix-cache reuse, not higher
raw kernel throughput.

## CPU gate

The five changed runtime files were mounted into the immutable profile 25 image
`sha256:a4dc62d34e6f28dbbeef6f0f233e93198cb6f7978694c09fa2648e19c2b9b48b`,
without GPU devices or network. Existing test-only pytest dependencies were
mounted separately; inference packages were not changed.

The following upstream suites were executed with Spark's real tokenizer from
`a608241037e4c2565356bff7ca293f2133888f88`. The fake-engine serving tests use the
actual full profile 25 candidate template rather than their simplified fixture:

- `tests/parser/test_glm47_exact_history.py`
- `tests/entrypoints/openai/chat_completion/test_glm_exact_history_serving.py`
- `tests/tool_parsers/test_glm47_moe_tool_parser.py`
- `tests/tool_parsers/test_glm4_moe_tool_parser.py`
- `tests/parser/engine/test_parser_engine.py`
- `tests/parser/engine/test_glm47_moe.py`

Candidate: **265 passed**. Stock parser and stock template: **46 failed,
219 passed**. All eight deliberately chosen serving round trips fail with
stock and pass with the candidate. Generic parser behavior, streaming chunk
boundaries, tool calls and unaffected Qwen whitespace behavior are covered.

These are adversarial regression fixtures, not a measured natural-workload
speedup. They establish the repair for selected whitespace cases. The companion
workspace experiment provides the CPU reproduction script and natural two-turn
API benchmark under `experiments/profile25-vllm-perf/history/`.

## Natural live history checks

On 2026-10-03/04 UTC, actual model-generated plain explanation, C# cache repair
and `read_file` tool conversations were each checked in streamed and
non-streamed modes. Each pair used a unique cache salt, returned the generated
assistant content/reasoning/tool calls, and requested prompt/output token IDs.
Temperature was zero, top-p 0.95, seed 42, reasoning effort low, and the output
budget 1536 tokens. There was one pair per case/mode and no extra project
context or forced model output.

| Check | Stock parser/template | Candidate parser/template |
| --- | --- | --- |
| Exact generated-history prefixes | **6/6** | **6/6** |
| Completed turns, without truncation | 12/12 | 12/12 |
| Follow-up prompt tokens | 3008 | 3075 |
| Follow-up cached tokens | 2638 | 2705 |
| Newly processed follow-up tokens | **370** | **370** |
| Recomputed representable history tokens | 0 | 0 |

**No natural-workload speedup was demonstrated.** Every reusable generated
history token was cached in both variants. Different response lengths explain
the different total prompt/cache counts. Stock ran with Marlin draft MoE on
port 8000; the candidate ran on a separate boot with B12X draft MoE, five
parser overlays and the candidate template on port 8015. These are history-correctness
checks, not an isolated parser performance comparison. Their elapsed times
must not be attributed to this patch. Non-streamed TTFT was not observed.

[history-live-summary.json](history-live-summary.json) retains the per-case
counts, token-sequence hashes, completion status, latency, source-artifact
hashes and deployment limitations. Generated text and token ID values are
omitted. All six first-prompt hashes match between variants. Known terminal
stop tokens were excluded from the history-prefix comparison. These short
checks do not establish quality equivalence or long-run stability.

## Exact template and deployment overlay

[chat_template_verbatim.jinja](chat_template_verbatim.jinja) is the exact
template used for the CPU and live candidate checks. Its only differences from
the stock recipe are the two `content.strip()` to `content` replacements.
No attribution header was inserted into the template, preserving its qualified
bytes; source attribution is recorded below.

```text
stock SHA256:
7a5a0dda1331a7c40d930961cc1cb3b57c3b52625250c13372fe006ba2e9dfdb
candidate SHA256:
688487bcf8d08c2c0945242c93988bca1a5657d79c9e23664259fb0858976770
```

Use the pinned profile 25 image and the intended existing experiment settings.
On each node, check out this history branch at the same commit and set
`PROFILE25_HISTORY_ROOT` to that checkout's absolute local path. Add the
following as a separate Compose override to the recipe's `glm53` service on
both nodes. Compose merges volumes by container destination, replacing the
stock template bind while retaining unrelated recipe mounts.

```yaml
services:
  glm53:
    volumes:
      - ${PROFILE25_HISTORY_ROOT}/tools/profile25/chat_template_verbatim.jinja:/opt/glm53/chat_template.jinja:ro
      - ${PROFILE25_HISTORY_ROOT}/vllm/parser/abstract_parser.py:/opt/venv/lib/python3.12/site-packages/vllm/parser/abstract_parser.py:ro
      - ${PROFILE25_HISTORY_ROOT}/vllm/parser/engine/adapters.py:/opt/venv/lib/python3.12/site-packages/vllm/parser/engine/adapters.py:ro
      - ${PROFILE25_HISTORY_ROOT}/vllm/parser/engine/parser_engine.py:/opt/venv/lib/python3.12/site-packages/vllm/parser/engine/parser_engine.py:ro
      - ${PROFILE25_HISTORY_ROOT}/vllm/parser/glm47_moe.py:/opt/venv/lib/python3.12/site-packages/vllm/parser/glm47_moe.py:ro
      - ${PROFILE25_HISTORY_ROOT}/vllm/tool_parsers/abstract_tool_parser.py:/opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/abstract_tool_parser.py:ro
```

The image-specific Python destination is `/opt/venv/lib/python3.12/site-packages`.
Keep the recipe's `--chat-template /opt/glm53/chat_template.jinja` argument.
Parser changes alone cannot prevent the stock template stripping content.
Mount the five individual Python files rather than substituting the entire
checkout for the image's installed vLLM package. Check resolved mounts and
the template hash before a planned restart. For `docker run`/`create`, use
the same six source/destination bindings as read-only mounts. This documented
overlay does not edit or enable changes in the normal recipe.

## Scope and attribution

The runtime overlay comprises `parser/abstract_parser.py`,
`parser/engine/adapters.py`, `parser/engine/parser_engine.py`,
`parser/glm47_moe.py` and `tool_parsers/abstract_tool_parser.py`, all under
`vllm/`. API-visible GLM whitespace preservation is intentional. Other parsers
retain their defaults. Whitespace after tool calls and JSON argument formatting
remain representational limits.

Original implementation and tests: Martin Vit, with Claude assistance as
recorded upstream. Profile 25 backport and qualification: Nathan W with OpenAI
Codex assistance. No upstream pull request has been opened.

Template source: TechnigmaAI's deployed recipe,
[`files/chat_template.jinja` at `fd20fa9efd62`](https://github.com/technigmaai/glm-5.3-flash-nvfp4-2x-dgx-sparks/blob/fd20fa9efd62af49fe6c8fa009d7ec1ec58fac18/files/chat_template.jinja).
Its stock hash is recorded above. The verbatim assistant-content change follows
the companion runtime fix cited by
[upstream vLLM PR 932](https://github.com/local-inference-lab/vllm/pull/932).
The template's existing content and notices are retained; this experiment does
not claim authorship of the original GLM template or the TechnigmaAI recipe.

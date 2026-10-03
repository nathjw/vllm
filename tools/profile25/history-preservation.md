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
speedup. No GPU performance, quality-equivalence or long-run stability claim
is made for this candidate. The companion workspace experiment provides the
CPU reproduction script, natural two-turn API benchmark and candidate template
under `experiments/profile25-vllm-perf/history/`.

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

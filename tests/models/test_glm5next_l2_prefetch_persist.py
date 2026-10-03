# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persisting-L2 set-aside sizing for the GLM-5.3 L2 weight prefetcher."""

import pytest
import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.models.glm5next.nvidia import l2_prefetch as l2pf

MAX = 84_000_000


@pytest.mark.parametrize(
    ("raw", "max_bytes", "expected"),
    [
        ("max", MAX, MAX),
        (" MAX ", MAX, MAX),
        ("all", MAX, MAX),
        ("40", MAX, 40_000_000),
        ("40.5", MAX, 40_500_000),
        ("200", MAX, MAX),  # clamped to the device maximum
        ("0", MAX, 0),
        ("off", MAX, 0),
        ("", MAX, 0),
        (None, MAX, 0),
        ("-5", MAX, 0),
        ("bogus", MAX, 0),
        ("max", 0, 0),  # device without a persisting L2 set-aside
    ],
)
def test_persisting_l2_request(raw, max_bytes, expected):
    assert l2pf.persisting_l2_request(raw, max_bytes) == expected


def test_default_request_leaves_the_driver_limit_unchanged(monkeypatch):
    monkeypatch.delenv("VLLM_GLM53_L2_PREFETCH_PERSIST_MB", raising=False)
    assert l2pf._persisting_l2_env_request() == "0"
    assert l2pf.persisting_l2_request("0", MAX) == 0


def test_invalid_numeric_environment_values_use_defaults(monkeypatch):
    monkeypatch.setenv("VLLM_GLM53_L2_PREFETCH_MAX_TOKENS", "invalid")
    monkeypatch.setenv("VLLM_GLM53_L2_PREFETCH_BUDGET_A_MB", "invalid")
    assert l2pf._int_env("VLLM_GLM53_L2_PREFETCH_MAX_TOKENS", 256) == 256
    assert l2pf._mb("VLLM_GLM53_L2_PREFETCH_BUDGET_A_MB", "20") == 20_000_000


@pytest.mark.parametrize("size", [48, 4096, 4096 * 17 + 48, 50_000_000])
@pytest.mark.parametrize("budget", [0, 15, 4096, 8_000_000])
def test_striped_prefetch_partitions_storage_within_budget(size, budget):
    """A partial cache hint must not overlap or cross the weight allocation."""
    pointer = 0x10000
    taken, rest = l2pf.stripe_budget(("weight", pointer, size), budget, 64)
    assert sum(length for _, _, length in taken) <= budget
    cursor = pointer
    for _, start, length in sorted(taken + rest, key=lambda item: item[1]):
        assert start == cursor
        assert start % 16 == 0
        assert length > 0
        cursor += length
    assert cursor == pointer + size
    if size == 50_000_000 and budget == 8_000_000:
        assert len(taken) == 64
        assert taken[-1][1] > pointer + size * 0.9


def test_flat_prefetch_pieces_never_read_past_a_tail():
    pieces = l2pf.prefetch_pieces([("weight", 0x10000, 8192 + 35)])
    assert pieces == [
        ("weight", 0x10000, 4096),
        ("weight", 0x11000, 4096),
        ("weight", 0x12000, 32),
    ]


def test_later_prefetch_window_does_not_repeat_striped_reads(monkeypatch):
    monkeypatch.setattr(l2pf, "_STRIPES", 64)
    source = [("projection", 0x10000, 24_000_000)]
    first, rest = l2pf.take_budget(source, 8_000_000)
    second, rest = l2pf.take_budget(rest, 6_000_000)
    assert sum(size for _, _, size in first) <= 8_000_000
    assert sum(size for _, _, size in second) <= 6_000_000
    cursor = source[0][1]
    for _, start, length in sorted(first + second + rest, key=lambda x: x[1]):
        assert start == cursor
        cursor += length
    assert cursor == source[0][1] + source[0][2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_striped_prefetch_graph_preserves_projection_output(monkeypatch):
    """Exercise the compiled flat-piece kernel and its side-stream graph join."""
    monkeypatch.setattr(l2pf, "_STRIPES", 64)
    monkeypatch.setattr(l2pf, "_compiled", None)
    monkeypatch.setattr(l2pf, "_compile_failed", False)
    weight = torch.randn((1024, 2048), dtype=torch.bfloat16, device="cuda")
    inputs = torch.randn((4, 1024), dtype=weight.dtype, device=weight.device)
    reference = inputs @ weight
    original = weight.clone()
    segment = l2pf.tensor_segment("weight", weight)
    assert segment is not None
    plan, _ = l2pf.make_plan([segment], 1_000_000, weight.device)
    assert plan is not None
    assert l2pf._get_launcher() is not None
    prefetcher = l2pf.L2Prefetcher.get(weight.device)
    prefetcher.issue(plan, 4)
    prefetcher.join()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prefetcher.issue(plan, 4)
        prefetcher.join()
        output = inputs @ weight
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(weight, original)
    assert torch.equal(output, reference)


def test_prefetch_is_disabled_for_the_entire_breakable_capture(monkeypatch):
    monkeypatch.setattr(
        BreakableCUDAGraphCapture,
        "current",
        classmethod(lambda cls: object()),
    )
    assert not l2pf._prefetch_allowed()


def test_prefetch_is_disabled_for_uncaptured_graph_warmup(monkeypatch):
    from vllm.compilation import monitor

    monkeypatch.setattr(
        BreakableCUDAGraphCapture,
        "current",
        classmethod(lambda cls: None),
    )
    monkeypatch.setattr(monitor, "is_cudagraph_capturing_enabled", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert not l2pf._prefetch_allowed()


def test_prefetch_is_enabled_for_regular_full_capture(monkeypatch):
    from vllm.compilation import monitor

    monkeypatch.setattr(
        BreakableCUDAGraphCapture,
        "current",
        classmethod(lambda cls: None),
    )
    monkeypatch.setattr(monitor, "is_cudagraph_capturing_enabled", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert l2pf._prefetch_allowed()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_segments_include_backend_tensors_and_skip_runtime_caches():
    class PackedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(64, device="cuda"))
            self.register_buffer("weight_scale", torch.empty(32, device="cuda"))
            self.register_buffer(
                "runtime_scratch", torch.empty(48, device="cuda"), persistent=False
            )
            self.packed_weight = torch.empty(48, device="cuda")
            self.topk_indices_buffer = torch.empty(48, device="cuda")
            self.kv_cache = torch.empty(48, device="cuda")

    module = PackedLinear()
    segments = l2pf.segments_of(module, min_bytes=0)
    names = [name for name, _, _ in segments]
    pointers = [pointer for _, pointer, _ in segments]

    assert names == ["weight", "weight_scale", "packed_weight"]
    assert len(pointers) == len(set(pointers))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_tensor_segment_limits_noncontiguous_views_to_their_storage():
    storage = torch.empty(8, dtype=torch.float32, device="cuda")
    view = storage.expand(16, 8)

    segment = l2pf.tensor_segment("expanded", view)

    assert segment == ("expanded", view.data_ptr(), storage.numel() * 4)


def _driver():
    from cuda.bindings import driver as cu

    return cu


def _read_limit(cu, device: torch.device) -> int:
    with torch.accelerator.device_index(device.index):
        torch.empty(1, device=device)  # make the primary context current
        err, value = cu.cuCtxGetLimit(cu.CUlimit.CU_LIMIT_PERSISTING_L2_CACHE_SIZE)
    assert err == cu.CUresult.CUDA_SUCCESS
    return int(value)


def _set_limit(cu, device: torch.device, value: int) -> None:
    with torch.accelerator.device_index(device.index):
        torch.empty(1, device=device)
        (err,) = cu.cuCtxSetLimit(cu.CUlimit.CU_LIMIT_PERSISTING_L2_CACHE_SIZE, value)
    assert err == cu.CUresult.CUDA_SUCCESS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_configure_persisting_l2_applies_to_the_primary_context():
    cu = _driver()
    device = torch.device("cuda", 0)
    err, dev = cu.cuDeviceGet(0)
    assert err == cu.CUresult.CUDA_SUCCESS
    err, max_bytes = cu.cuDeviceGetAttribute(
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE, dev
    )
    assert err == cu.CUresult.CUDA_SUCCESS
    if max_bytes <= 0:
        pytest.skip("device has no persisting L2 set-aside")
    before = _read_limit(cu, device)
    try:
        assert l2pf.configure_persisting_l2(device, request="max") == max_bytes
        assert _read_limit(cu, device) == max_bytes

        # A zero request never touches the driver state.
        assert l2pf.configure_persisting_l2(device, request="0") == 0
        assert _read_limit(cu, device) == max_bytes

        # Over-sized requests are clamped to the device maximum.
        assert l2pf.configure_persisting_l2(device, request="100000") == max_bytes
    finally:
        _set_limit(cu, device, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_prefetcher_applies_the_env_request_once(monkeypatch):
    cu = _driver()
    device = torch.device("cuda", 0)
    before = _read_limit(cu, device)
    err, dev = cu.cuDeviceGet(0)
    assert err == cu.CUresult.CUDA_SUCCESS
    err, max_bytes = cu.cuDeviceGetAttribute(
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
        dev,
    )
    assert err == cu.CUresult.CUDA_SUCCESS
    if max_bytes <= 0:
        pytest.skip("device has no persisting L2 set-aside")
    calls: list[str | None] = []
    real = l2pf.configure_persisting_l2

    def spy(dev, request=None):
        calls.append(request)
        return real(dev, request=request)

    monkeypatch.setattr(l2pf, "configure_persisting_l2", spy)
    monkeypatch.setattr(l2pf.L2Prefetcher, "_instances", {})
    monkeypatch.setattr(l2pf, "_persisting_l2_applied", {})
    monkeypatch.setattr(l2pf, "PERSIST_L2", "1")
    try:
        first = l2pf.L2Prefetcher.get(device)
        second = l2pf.L2Prefetcher.get(device)
        assert first is second
        assert calls == [None]
        applied = _read_limit(cu, device)
        assert first.persisting_l2_bytes == applied
        # CUDA rounds the request up to the device's allocation granularity.
        assert 1_000_000 <= applied < int(max_bytes)
    finally:
        _set_limit(cu, device, before)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 CUDA device required",
)
def test_prefetch_side_stream_captures_and_replays(monkeypatch):
    device = torch.device("cuda", 0)
    data = torch.empty(1 << 20, dtype=torch.uint8, device=device)
    segment = l2pf.tensor_segment("weight", data)
    assert segment is not None
    plan = l2pf.L2PrefetchPlan([segment], device)

    monkeypatch.setattr(l2pf, "ENABLED", True)
    monkeypatch.setattr(l2pf, "PERSIST_L2", "0")
    monkeypatch.setattr(l2pf, "_prefetch_allowed", lambda: True)
    monkeypatch.setattr(l2pf.L2Prefetcher, "_instances", {})
    monkeypatch.setattr(l2pf, "_persisting_l2_applied", {})
    assert l2pf._get_launcher() is not None
    l2pf.L2Prefetcher.get(device)

    l2pf.issue(plan, 1)
    l2pf.join_all()
    torch.accelerator.synchronize()

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            l2pf.issue(plan, 1)
            l2pf.join_all()
    torch.cuda.current_stream(device).wait_stream(stream)
    graph.replay()
    graph.replay()
    torch.accelerator.synchronize()
    assert plan.segs is not None and plan.segs.data_ptr() != 0

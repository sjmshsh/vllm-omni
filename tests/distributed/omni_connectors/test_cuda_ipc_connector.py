# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the CudaIPC connector and its IpcRing control plane.

Two layers, gated independently:

1. ``IpcRing`` (the lock-free SPSC keyed-mailbox control plane defined in
   ``cuda_ipc_connector.py``): pure-Python, NO CUDA — these tests run anywhere a CPU
   torch import works, incl. CPU-only CI.

2. ``CudaIPCConnector`` functional put/get: requires a real GPU. CUDA IPC handles cannot be
   opened in the same process that created them, so these spawn sender + receiver processes.
   The GPU gate is on ``TestCudaIPCFunctional`` (class level), NOT module level, so it does
   not skip the CPU ring tests above.
"""

from __future__ import annotations

import multiprocessing as mp
import uuid
from typing import Any

import pytest
import torch

# ════════════════════════════════════════════════════════════════════
# Layer 1 — IpcRing control plane (CPU-only, runs in CI without a GPU)
# ════════════════════════════════════════════════════════════════════
#
# Single-mapping publish/poll protocol tests. Cross-process / cross-mapping integrity is
# separately exercised by tests/dfx/perf/ipc_ring_soak.py (the real deployment shape).
# NOTE: same-process re-open() of a SharedMemory segment is not coherent on macOS, so these
# poll from the producing ring object directly.
from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import (
    IpcRing,
    RingFullError,
)


def _kh(s: str) -> bytes:
    """16-byte key hash, as the connector derives via sha1(key)[:16]."""
    import hashlib

    return hashlib.sha1(s.encode()).digest()[:16]


@pytest.fixture()
def ring():
    """A small sender-owned ring; unlinks on close."""
    name = f"test_ipc_ring_{uuid.uuid4().hex[:12]}"
    r = IpcRing.create(name, n_slots=8, body_max=64, header_bytes=32)
    yield r
    r.close()


def test_ring_header_round_trip(ring):
    blob = b"edge-constant-handles\x00\x01\x02"
    ring.write_header(blob)
    assert ring.read_header(len(blob)) == blob


def test_ring_header_overflow_rejected(ring):
    with pytest.raises(ValueError):
        ring.write_header(b"x" * 33)  # header_bytes=32


def test_ring_publish_then_poll(ring):
    kh = _kh("req-A_0_1")
    ring.publish(kh, pclass=0, body=b"hello")
    got = ring.poll(kh)
    assert got is not None
    pclass, body = got
    assert pclass == 0 and body == b"hello"


def test_ring_poll_miss_returns_none(ring):
    assert ring.poll(_kh("never-published")) is None


def test_ring_pclass_is_carried(ring):
    ring.publish(_kh("k-inline"), pclass=0, body=b"a")
    ring.publish(_kh("k-pool"), pclass=1, body=b"bb")
    assert ring.poll(_kh("k-inline"))[0] == 0
    assert ring.poll(_kh("k-pool"))[0] == 1


def test_ring_poll_marks_consumed_once(ring):
    kh = _kh("once")
    ring.publish(kh, 0, b"x")
    assert ring.poll(kh) is not None  # first poll consumes
    assert ring.poll(kh) is None  # second poll: already consumed


def test_ring_consumed_slot_is_reused(ring):
    """Producer must reuse a slot the consumer has taken — else the ring wedges after
    n_slots publishes. Round-trips far more entries than slots."""
    for i in range(8 * 20):  # 160 entries through 8 slots
        kh = _kh(f"seq-{i}")
        ring.publish(kh, 0, b"%d" % i)
        got = ring.poll(kh)
        assert got is not None and got[1] == b"%d" % i


def test_ring_open_addressed_collision(ring):
    """Distinct keys that may land on the same home slot must each be retrievable."""
    keys = [_kh(f"collide-{i}") for i in range(6)]  # < n_slots, all live at once
    for i, k in enumerate(keys):
        ring.publish(k, 0, b"v%d" % i)
    for i, k in enumerate(keys):
        got = ring.poll(k)
        assert got is not None and got[1] == b"v%d" % i


def test_ring_full_raises(ring):
    for i in range(8):  # fill all 8 slots without consuming
        ring.publish(_kh(f"fill-{i}"), 0, b"z")
    with pytest.raises(RingFullError):
        ring.publish(_kh("one-too-many"), 0, b"z")


def test_ring_body_too_big_raises(ring):
    with pytest.raises(ValueError):
        ring.publish(_kh("big"), 0, b"x" * 65)  # body_max=64


def test_ring_ttl_reclaims_stale_entry(ring):
    """C4: an occupied-but-unconsumed slot older than ttl_sec is reclaimed in place so an
    aborted/never-polled request cannot wedge the ring. Fresh entries are NOT reclaimed."""
    for i in range(8):  # fill at t=100, never consumed
        ring.publish(_kh(f"stale-{i}"), 0, b"old", ts=100, ttl_sec=30)
    with pytest.raises(RingFullError):  # t=110 within ttl -> still full
        ring.publish(_kh("fresh"), 0, b"new", ts=110, ttl_sec=30)
    # t=200: the t=100 entries are stale (>30s) -> reclaimed in place, publish succeeds
    ring.publish(_kh("after-ttl"), 0, b"new", ts=200, ttl_sec=30)


def test_ring_ttl_zero_never_reclaims(ring):
    """ttl_sec=0 disables reclaim — full ring stays full regardless of ts."""
    for i in range(8):
        ring.publish(_kh(f"x-{i}"), 0, b"o", ts=100, ttl_sec=0)
    with pytest.raises(RingFullError):
        ring.publish(_kh("y"), 0, b"n", ts=999999, ttl_sec=0)


# ════════════════════════════════════════════════════════════════════
# Layer 2 — CudaIPCConnector functional put/get (requires a GPU)
# ════════════════════════════════════════════════════════════════════


def _sender_proc(cmd_q: mp.Queue, res_q: mp.Queue, cfg: dict):
    import torch

    torch.cuda.set_device(cfg["device"])
    from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import CudaIPCConnector

    sender = CudaIPCConnector(cfg)
    try:
        res_q.put(("ready",))
        while True:
            msg = cmd_q.get()
            if msg[0] == "put":
                _, fs, ts, key, spec = msg
                data = _materialize(spec, "cuda")
                ok, size, meta = sender.put(fs, ts, key, data)
                res_q.put(("put_done", ok, size, meta))
            elif msg[0] == "health":
                res_q.put(("health", sender.health()))
            elif msg[0] == "metrics":
                res_q.put(("metrics", dict(sender._metrics)))
            elif msg[0] == "quit":
                break
    finally:
        sender.close()


def _receiver_proc(cmd_q: mp.Queue, res_q: mp.Queue, cfg: dict):
    import torch

    torch.cuda.set_device(cfg["device"])
    from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import CudaIPCConnector

    receiver = CudaIPCConnector(cfg)
    try:
        res_q.put(("ready",))
        while True:
            msg = cmd_q.get()
            if msg[0] == "get":
                _, fs, ts, key, meta = msg
                result = receiver.get(fs, ts, key, metadata=meta)
                if result is None:
                    res_q.put(("get_done", None))
                else:
                    obj, rsize = result
                    res_q.put(("get_done", _summarize(obj), rsize))
            elif msg[0] == "quit":
                break
    finally:
        receiver.close()


def _materialize(spec: dict, device: str) -> dict:
    import torch

    out: dict[str, Any] = {}
    for k, v in spec.items():
        if isinstance(v, dict) and v.get("__t"):
            out[k] = torch.randn(*v["shape"], device=device, dtype=getattr(torch, v["dtype"]))
        else:
            out[k] = v
    return out


def _tspec(shape: tuple, dtype: str = "bfloat16") -> dict:
    return {"__t": True, "shape": list(shape), "dtype": dtype}


def _summarize(obj: dict) -> dict:
    summary: dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, torch.Tensor):
            summary[k] = {"shape": list(v.shape), "device": str(v.device)}
        else:
            summary[k] = v
    return summary


class _Harness:
    def __init__(self, pool_size_mb: int = 32, pool_credits: int = 16):
        ctx = mp.get_context("spawn")
        self.s_cmd: mp.Queue = ctx.Queue()
        self.s_res: mp.Queue = ctx.Queue()
        self.r_cmd: mp.Queue = ctx.Queue()
        self.r_res: mp.Queue = ctx.Queue()
        dev = torch.accelerator.current_device_index()

        s_cfg = {
            "stage_id": 0,
            "role": "sender",
            "local_device": dev,
            "pool_size_mb": pool_size_mb,
            "pool_credits": pool_credits,
            "tensor_lifetime_sec": 10.0,
            "device": dev,
        }
        r_cfg = {"stage_id": 1, "role": "receiver", "local_device": dev, "device": dev}

        self.sender = ctx.Process(target=_sender_proc, args=(self.s_cmd, self.s_res, s_cfg), daemon=True)
        self.receiver = ctx.Process(target=_receiver_proc, args=(self.r_cmd, self.r_res, r_cfg), daemon=True)
        self.sender.start()
        self.receiver.start()
        assert self.s_res.get(timeout=30)[0] == "ready"
        assert self.r_res.get(timeout=30)[0] == "ready"

    def put(self, fs, ts, key, spec, timeout=10):
        self.s_cmd.put(("put", fs, ts, key, spec))
        r = self.s_res.get(timeout=timeout)
        return r[1], r[2], r[3]

    def get(self, fs, ts, key, meta=None, timeout=10):
        self.r_cmd.put(("get", fs, ts, key, meta))
        r = self.r_res.get(timeout=timeout)
        if r[1] is None:
            return None
        return r[1], r[2]

    def health(self, timeout=5):
        self.s_cmd.put(("health",))
        return self.s_res.get(timeout=timeout)[1]

    def metrics(self, timeout=5):
        self.s_cmd.put(("metrics",))
        return self.s_res.get(timeout=timeout)[1]

    def close(self):
        for q in (self.s_cmd, self.r_cmd):
            try:
                q.put(("quit",))
            except Exception:
                pass
        self.sender.join(timeout=5)
        self.receiver.join(timeout=5)
        for p in (self.sender, self.receiver):
            if p.is_alive():
                p.kill()


@pytest.fixture()
def harness():
    h = _Harness()
    yield h
    h.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCudaIPCFunctional:
    def test_put_then_get(self, harness):
        spec = {"hidden": _tspec((128, 256)), "meta": {"req_id": "r1"}}
        ok, size, meta = harness.put("s0", "s1", "req_1", spec)
        assert ok and size > 0

        result = harness.get("s0", "s1", "req_1", meta=meta)
        assert result is not None
        summary, _ = result
        assert summary["hidden"]["shape"] == [128, 256]
        assert "cuda" in summary["hidden"]["device"]
        assert summary["meta"]["req_id"] == "r1"

    def test_multiple_keys(self, harness):
        for i in range(8):
            ok, _, _ = harness.put("s0", "s1", f"req_{i}", {"h": _tspec((64, 128)), "i": i})
            assert ok

        for i in range(8):
            result = harness.get("s0", "s1", f"req_{i}")
            assert result is not None
            summary, _ = result
            assert summary["i"] == i

    def test_cpu_fallback_on_overflow(self, harness):
        spec = {"big": _tspec((8 * 1024 * 1024,), "float32")}
        ok, _, meta = harness.put("s0", "s1", "big_req", spec)
        assert ok
        assert meta.get("cpu_fallback", False)

    def test_health(self, harness):
        h = harness.health()
        assert h["status"] == "healthy"
        assert h["role"] == "sender"
        assert h["pool_credits"] == 16

    def test_supports_gpu_tensor_flag(self):
        from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import CudaIPCConnector

        assert CudaIPCConnector.supports_gpu_tensor is True

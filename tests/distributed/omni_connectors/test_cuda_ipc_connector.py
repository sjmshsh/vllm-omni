# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CudaIPCConnector basic functionality.

CUDA IPC handles require cross-process usage (cudaIpcOpenMemHandle cannot be
called in the same process that called cudaIpcGetMemHandle).  All put/get
tests spawn sender + receiver in separate processes, coordinated via queues.

Requires at least one CUDA GPU.  Skipped automatically when CUDA is unavailable.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.gpu,
]


# ── Multi-process helpers ───────────────────────────────────────────


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


# ── Functional tests ────────────────────────────────────────────────


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

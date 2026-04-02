# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for CudaIPCConnector.

Includes both:
  - CPU-only mock tests (no GPU required) that verify payload-processor
    integration logic (.cpu() call gating).
  - GPU tests that verify actual CUDA IPC D2D transfer (skipped without GPU).
"""

import multiprocessing as mp
import time
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
    thinker2talker_async_chunk,
)


# ---------------------------------------------------------------------------
# CPU-only mock tests (no GPU needed)
# ---------------------------------------------------------------------------

pytestmark_cpu = [pytest.mark.core_model, pytest.mark.cpu]


def _build_transfer_manager(*, supports_gpu_tensor: bool):
    return SimpleNamespace(
        put_req_chunk=defaultdict(int, {"req-1": 1}),
        request_payload={},
        connector=SimpleNamespace(supports_gpu_tensor=supports_gpu_tensor),
    )


def _build_request():
    return SimpleNamespace(
        external_req_id="req-1",
        output_token_ids=[1, 2],
    )


def _mock_pooling_output():
    tensor = MagicMock()
    tensor.detach.return_value = tensor
    tensor.cpu.return_value = tensor
    return {"0": tensor}


def test_factory_has_cuda_ipc_connector():
    """CudaIPCConnector should be registered in the factory."""
    assert "CudaIPCConnector" in OmniConnectorFactory.list_registered_connectors()


@pytest.mark.parametrize(
    "supports_gpu_tensor,expected_cpu_calls",
    [(True, 0), (False, 1)],
)
def test_thinker2talker_respects_gpu_tensor_capability(supports_gpu_tensor, expected_cpu_calls):
    """Verify that .cpu() is only called when connector does not support GPU tensors."""
    tm = _build_transfer_manager(supports_gpu_tensor=supports_gpu_tensor)
    req = _build_request()
    pool = _mock_pooling_output()

    out = thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output=pool,
        request=req,
        is_finished=False,
    )

    assert out is not None
    assert out["thinker_output_token_ids"] == [1, 2]
    assert pool["0"].cpu.call_count == expected_cpu_calls


# ---------------------------------------------------------------------------
# GPU tests (require CUDA)
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@requires_cuda
class TestCudaIPCConnectorSingleGPU:
    """Tests that run on a single GPU (control-plane + same-device D2D)."""

    def _make_connector(self, **overrides):
        from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import (
            CudaIPCConnector,
        )
        config = {"local_device": "cuda:0", "role": "sender", **overrides}
        return CudaIPCConnector(config)

    def test_create_via_factory(self):
        from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import (
            CudaIPCConnector,
        )
        from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

        spec = ConnectorSpec(
            name="CudaIPCConnector",
            extra={"local_device": "auto", "role": "sender"},
        )
        connector = OmniConnectorFactory.create_connector(spec)
        assert isinstance(connector, CudaIPCConnector)
        assert connector.supports_gpu_tensor is True
        connector.close()

    def test_invalid_role_raises(self):
        from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import (
            CudaIPCConnector,
        )
        with pytest.raises(ValueError, match="Invalid role"):
            CudaIPCConnector({"role": "invalid"})

    def test_put_get_cpu_only_data(self):
        """put/get with CPU-only data (no GPU tensors)."""
        sender = self._make_connector(role="sender")
        receiver = self._make_connector(role="receiver")
        try:
            data = {"tokens": [1, 2, 3], "finished": torch.tensor(False, dtype=torch.bool)}
            success, size, meta = sender.put("0", "1", "test_cpu_only", data)
            assert success

            result = receiver.get("0", "1", "test_cpu_only")
            assert result is not None
            payload, _ = result
            assert payload["tokens"] == [1, 2, 3]
        finally:
            sender.close()
            receiver.close()

    def test_put_get_gpu_tensor_same_device(self):
        """put/get with GPU tensor on the same device (D2D copy to self)."""
        sender = self._make_connector(role="sender")
        receiver = self._make_connector(role="receiver")
        try:
            original = torch.randn(4, 8, device="cuda:0")
            data = {"embedding": original, "flag": True}

            success, size, meta = sender.put("0", "1", "test_same_dev", data)
            assert success

            result = receiver.get("0", "1", "test_same_dev")
            assert result is not None
            payload, _ = result

            got = payload["embedding"]
            assert got.is_cuda
            assert torch.equal(got, original)
            assert payload["flag"] is True
        finally:
            sender.close()
            receiver.close()

    def test_mixed_dict_payload(self):
        """put/get with a mixed dict mimicking real Qwen3-Omni payload."""
        sender = self._make_connector(role="sender")
        receiver = self._make_connector(role="receiver")
        try:
            data = {
                "thinker_prefill_embeddings": torch.randn(50, 2048, device="cuda:0"),
                "thinker_hidden_states": torch.randn(50, 2048, device="cuda:0"),
                "tts_bos_embed": torch.randn(1, 4096, device="cuda:0"),
                "thinker_sequences": list(range(100)),
                "thinker_input_ids": list(range(50)),
                "finished": torch.tensor(False, dtype=torch.bool),
            }

            success, size, meta = sender.put("0", "1", "test_mixed", data)
            assert success

            result = receiver.get("0", "1", "test_mixed")
            assert result is not None
            payload, _ = result

            for key in ["thinker_prefill_embeddings", "thinker_hidden_states", "tts_bos_embed"]:
                assert payload[key].is_cuda, f"{key} should be on GPU"
                assert torch.equal(payload[key], data[key])

            assert payload["thinker_sequences"] == list(range(100))
            assert payload["thinker_input_ids"] == list(range(50))
        finally:
            sender.close()
            receiver.close()

    def test_ack_releases_held_tensors(self):
        """ACK from receiver should release sender's held tensor refs."""
        sender = self._make_connector(role="sender")
        receiver = self._make_connector(role="receiver")
        try:
            t = torch.randn(2, 3, device="cuda:0")
            success, _, _ = sender.put("0", "1", "test_ack", {"t": t})
            assert success
            assert "test_ack" in sender._held_tensors

            # Receiver gets (which sends ACK)
            result = receiver.get("0", "1", "test_ack")
            assert result is not None

            # Wait for ACK drain loop (~20ms interval)
            time.sleep(0.5)

            assert "test_ack" not in sender._held_tensors
        finally:
            sender.close()
            receiver.close()

    def test_ttl_expiry_releases_tensors(self):
        """TTL expiry should release held tensor refs."""
        sender = self._make_connector(role="sender", tensor_lifetime_sec=0.5)
        try:
            t = torch.randn(2, 3, device="cuda:0")
            success, _, _ = sender.put("0", "1", "test_ttl", {"t": t})
            assert success
            assert "test_ttl" in sender._held_tensors

            # Wait for TTL to expire
            time.sleep(1.5)

            assert "test_ttl" not in sender._held_tensors
            assert sender._metrics["ack_timeouts"] > 0
        finally:
            sender.close()

    def test_health(self):
        """health() should return expected fields."""
        connector = self._make_connector(role="sender")
        try:
            h = connector.health()
            assert h["status"] == "healthy"
            assert h["role"] == "sender"
            assert "held_tensors" in h
            assert "puts" in h
        finally:
            connector.close()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires at least 2 GPUs for cross-device D2D test",
)
class TestCudaIPCConnectorMultiGPU:
    """Tests that require 2+ GPUs for actual cross-device D2D transfer."""

    def test_cross_device_d2d_transfer(self):
        """D2D transfer between GPU0 and GPU1 in the same process."""
        from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import (
            CudaIPCConnector,
        )

        sender = CudaIPCConnector({"role": "sender", "local_device": "cuda:0"})
        receiver = CudaIPCConnector({"role": "receiver", "local_device": "cuda:1"})
        try:
            original = torch.randn(32, 2048, device="cuda:0")
            data = {"hidden_states": original, "tokens": [1, 2, 3]}

            success, _, meta = sender.put("0", "1", "test_d2d", data)
            assert success

            result = receiver.get("0", "1", "test_d2d")
            assert result is not None
            payload, _ = result

            got = payload["hidden_states"]
            assert got.device == torch.device("cuda:1")
            assert torch.equal(got.cpu(), original.cpu()), "Data should be bit-exact"
        finally:
            sender.close()
            receiver.close()

    def test_cross_process_d2d_transfer(self):
        """D2D transfer between two separate processes (GPU0 -> GPU1)."""
        mp.set_start_method("spawn", force=True)
        barrier = mp.Barrier(2)

        key = "test_cross_proc"
        tensor_data = list(range(16))
        shape = (4, 4)
        dtype_str = "float32"

        sender_proc = mp.Process(
            target=_sender_process,
            args=(0, key, tensor_data, shape, dtype_str, barrier),
        )
        receiver_proc = mp.Process(
            target=_receiver_process,
            args=(1, key, tensor_data, shape, dtype_str, barrier),
        )

        sender_proc.start()
        receiver_proc.start()

        sender_proc.join(timeout=15)
        receiver_proc.join(timeout=15)

        assert sender_proc.exitcode == 0, f"Sender failed with exit code {sender_proc.exitcode}"
        assert receiver_proc.exitcode == 0, f"Receiver failed with exit code {receiver_proc.exitcode}"


# ---------------------------------------------------------------------------
# Cross-process test helpers (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _sender_process(device_id, key, tensor_data, shape, dtype_str, barrier):
    from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import (
        CudaIPCConnector,
    )

    torch.cuda.set_device(device_id)
    connector = CudaIPCConnector({
        "local_device": f"cuda:{device_id}",
        "role": "sender",
        "tensor_lifetime_sec": 30,
    })
    try:
        tensor = torch.tensor(
            tensor_data,
            dtype=getattr(torch, dtype_str),
            device=f"cuda:{device_id}",
        ).reshape(shape)
        data = {"gpu_tensor": tensor, "cpu_list": [1, 2, 3], "cpu_flag": True}
        success, size, metadata = connector.put("0", "1", key, data)
        assert success

        barrier.wait(timeout=10)
        time.sleep(0.5)
    finally:
        connector.close()


def _receiver_process(device_id, key, tensor_data, shape, dtype_str, barrier):
    from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import (
        CudaIPCConnector,
    )

    torch.cuda.set_device(device_id)
    connector = CudaIPCConnector({
        "local_device": f"cuda:{device_id}",
        "role": "receiver",
        "tensor_lifetime_sec": 30,
    })
    try:
        result = None
        for _ in range(50):
            result = connector.get("0", "1", key)
            if result is not None:
                break
            time.sleep(0.1)

        assert result is not None
        payload, size = result

        got = payload["gpu_tensor"]
        assert got.is_cuda
        assert got.device == torch.device(f"cuda:{device_id}")

        expected = torch.tensor(
            tensor_data, dtype=getattr(torch, dtype_str)
        ).reshape(shape)
        assert torch.equal(got.cpu(), expected)
        assert payload["cpu_list"] == [1, 2, 3]
        assert payload["cpu_flag"] is True

        barrier.wait(timeout=10)
    finally:
        connector.close()

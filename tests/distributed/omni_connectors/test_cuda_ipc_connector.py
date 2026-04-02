# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for CudaIPCConnector.

These tests exercise the CUDA IPC D2D transfer path.  Tests that require
two GPUs are skipped when fewer are available.  Single-GPU tests verify
the control-plane logic (SHM serialization, ACK mechanism) without
actually performing cross-device D2D copies.
"""

import multiprocessing as mp
import time

import pytest
import torch

from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import (
    CudaIPCConnector,
    _shm_name,
)
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

# Skip all GPU tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sender_process(device_id: int, key: str, tensor_data: list, shape: tuple, dtype_str: str, barrier):
    """Sender subprocess: creates a GPU tensor and puts it via CudaIPCConnector."""
    torch.cuda.set_device(device_id)
    connector = CudaIPCConnector({
        "local_device": f"cuda:{device_id}",
        "role": "sender",
        "tensor_lifetime_sec": 30,
    })
    try:
        tensor = torch.tensor(tensor_data, dtype=getattr(torch, dtype_str), device=f"cuda:{device_id}").reshape(shape)
        data = {
            "gpu_tensor": tensor,
            "cpu_list": [1, 2, 3],
            "cpu_flag": True,
        }
        success, size, metadata = connector.put("0", "1", key, data)
        assert success, "put() should succeed"
        assert metadata is not None
        assert metadata["has_gpu_tensors"] is True

        # Wait for receiver to complete
        barrier.wait(timeout=10)
        # Give cleanup thread time to process ACK
        time.sleep(1.0)
    finally:
        connector.close()


def _receiver_process(device_id: int, key: str, expected_data: list, shape: tuple, dtype_str: str, barrier):
    """Receiver subprocess: gets data via CudaIPCConnector and verifies."""
    torch.cuda.set_device(device_id)
    connector = CudaIPCConnector({
        "local_device": f"cuda:{device_id}",
        "role": "receiver",
        "tensor_lifetime_sec": 30,
    })
    try:
        # Poll until data is available
        result = None
        for _ in range(50):  # up to 5 seconds
            result = connector.get("0", "1", key)
            if result is not None:
                break
            time.sleep(0.1)

        assert result is not None, "get() should return data"
        payload, size = result

        # Verify GPU tensor
        assert "gpu_tensor" in payload
        got = payload["gpu_tensor"]
        assert got.is_cuda, "Tensor should be on GPU"
        assert got.device == torch.device(f"cuda:{device_id}"), f"Tensor should be on cuda:{device_id}"

        expected = torch.tensor(expected_data, dtype=getattr(torch, dtype_str)).reshape(shape)
        assert torch.equal(got.cpu(), expected), "Tensor data should match bit-exactly"

        # Verify CPU data
        assert payload["cpu_list"] == [1, 2, 3]
        assert payload["cpu_flag"] is True

        # Signal sender
        barrier.wait(timeout=10)
    finally:
        connector.close()


# ---------------------------------------------------------------------------
# Factory test
# ---------------------------------------------------------------------------

class TestCudaIPCConnectorFactory:
    def test_create_via_factory(self):
        """Test creating CudaIPCConnector via OmniConnectorFactory."""
        spec = ConnectorSpec(
            name="CudaIPCConnector",
            extra={"local_device": "auto", "role": "sender"},
        )
        connector = OmniConnectorFactory.create_connector(spec)
        assert isinstance(connector, CudaIPCConnector)
        assert connector.supports_gpu_tensor is True
        connector.close()

    def test_invalid_role_raises(self):
        """Test that an invalid role raises ValueError."""
        with pytest.raises(ValueError, match="Invalid role"):
            CudaIPCConnector({"role": "invalid"})


# ---------------------------------------------------------------------------
# Single-GPU tests (control plane only)
# ---------------------------------------------------------------------------

class TestCudaIPCConnectorSingleGPU:
    def test_put_get_cpu_only_data(self):
        """Test put/get with CPU-only data (no GPU tensors)."""
        sender = CudaIPCConnector({"role": "sender", "local_device": "cuda:0"})
        receiver = CudaIPCConnector({"role": "receiver", "local_device": "cuda:0"})
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
        """Test put/get with GPU tensor on the same device (D2D copy to self)."""
        sender = CudaIPCConnector({"role": "sender", "local_device": "cuda:0"})
        receiver = CudaIPCConnector({"role": "receiver", "local_device": "cuda:0"})
        try:
            original = torch.randn(4, 8, device="cuda:0")
            data = {"embedding": original, "flag": True}

            success, size, meta = sender.put("0", "1", "test_same_dev", data)
            assert success
            assert meta["has_gpu_tensors"]

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

    def test_ack_releases_held_tensors(self):
        """Test that ACK from receiver releases sender's held tensor refs."""
        sender = CudaIPCConnector({"role": "sender", "local_device": "cuda:0"})
        receiver = CudaIPCConnector({"role": "receiver", "local_device": "cuda:0"})
        try:
            t = torch.randn(2, 3, device="cuda:0")
            data = {"t": t}

            success, _, _ = sender.put("0", "1", "test_ack", data)
            assert success

            internal_key = sender._make_key("test_ack", "0", "1")
            assert internal_key in sender._held_tensors, "Tensor should be held after put"

            # Receiver gets (which sends ACK)
            result = receiver.get("0", "1", "test_ack")
            assert result is not None

            # Wait for cleanup thread to process ACK
            time.sleep(2.0)

            assert internal_key not in sender._held_tensors, "Tensor should be released after ACK"
        finally:
            sender.close()
            receiver.close()

    def test_ttl_expiry_releases_tensors(self):
        """Test that TTL expiry releases held tensor refs."""
        sender = CudaIPCConnector({
            "role": "sender",
            "local_device": "cuda:0",
            "tensor_lifetime_sec": 1,  # Very short TTL for testing
        })
        try:
            t = torch.randn(2, 3, device="cuda:0")
            success, _, _ = sender.put("0", "1", "test_ttl", {"t": t})
            assert success

            internal_key = sender._make_key("test_ttl", "0", "1")
            assert internal_key in sender._held_tensors

            # Wait for TTL to expire
            time.sleep(3.0)

            assert internal_key not in sender._held_tensors, "Tensor should be released after TTL"
            assert sender._metrics["ttl_expired"] > 0
        finally:
            sender.close()

    def test_health(self):
        """Test health() returns expected fields."""
        connector = CudaIPCConnector({"role": "sender", "local_device": "cuda:0"})
        try:
            h = connector.health()
            assert h["status"] == "healthy"
            assert h["role"] == "sender"
            assert "held_tensor_groups" in h
            assert "puts" in h
        finally:
            connector.close()

    def test_mixed_dict_payload(self):
        """Test put/get with a mixed dict of GPU tensors and CPU data."""
        sender = CudaIPCConnector({"role": "sender", "local_device": "cuda:0"})
        receiver = CudaIPCConnector({"role": "receiver", "local_device": "cuda:0"})
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
            assert meta["has_gpu_tensors"]

            result = receiver.get("0", "1", "test_mixed")
            assert result is not None
            payload, _ = result

            # GPU tensors should be on GPU
            for key in ["thinker_prefill_embeddings", "thinker_hidden_states", "tts_bos_embed"]:
                assert payload[key].is_cuda, f"{key} should be on GPU"
                assert torch.equal(payload[key], data[key])

            # CPU data should be preserved
            assert payload["thinker_sequences"] == list(range(100))
            assert payload["thinker_input_ids"] == list(range(50))
        finally:
            sender.close()
            receiver.close()


# ---------------------------------------------------------------------------
# Multi-GPU tests (actual D2D transfer)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Requires at least 2 GPUs for cross-device D2D test",
)
class TestCudaIPCConnectorMultiGPU:
    def test_cross_device_d2d_transfer(self):
        """Test D2D transfer between GPU0 and GPU1 on the same device."""
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
            assert got.device == torch.device("cuda:1"), "Tensor should be on cuda:1"
            assert torch.equal(got.cpu(), original.cpu()), "Data should be bit-exact"
        finally:
            sender.close()
            receiver.close()

    def test_cross_process_d2d_transfer(self):
        """Test D2D transfer between two separate processes (GPU0 -> GPU1)."""
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

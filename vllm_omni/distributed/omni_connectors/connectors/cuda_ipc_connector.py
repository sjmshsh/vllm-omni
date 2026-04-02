# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA IPC Connector for single-node GPU-to-GPU (D2D) tensor transfer.

Split-plane architecture:
  - Data Plane: GPU tensors stay on sender GPU. CUDA IPC handles (64 bytes
    each) are passed to the receiver, which opens them and performs a D2D
    copy via NVLink / PCIe P2P.
  - Control Plane: IPC handles + small CPU data serialized via /dev/shm
    (< 1 KB per chunk).
"""

import fcntl
import struct
import threading
import time as _time_mod
from multiprocessing import shared_memory as shm_pkg
from typing import Any

import torch

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

# Magic bytes to identify GPU tensor entries in the serialized control payload
_GPU_TENSOR_MARKER = b"__cuda_ipc__"

# ACK value written by receiver to signal D2D copy completion
_ACK_DONE = b"\x01"


def _shm_name(key: str, suffix: str = "") -> str:
    """Generate a deterministic SHM segment name from a connector key."""
    # SHM names must be <= 255 chars and start with /
    name = f"cipc_{key}{suffix}"
    # Replace characters illegal in POSIX shm names
    return name.replace("/", "_").replace("@", "_")[:200]


class CudaIPCConnector(OmniConnectorBase):
    """Connector for direct GPU-to-GPU transfer via CUDA IPC + NVLink.

    GPU tensors are transferred using CUDA IPC memory handles, which allow
    cross-process access to GPU memory.  The receiver opens the handle and
    performs a D2D copy (leveraging NVLink when available).  Non-tensor data
    (token IDs, flags, etc.) is serialized and passed via shared memory.

    An ACK-based lifecycle ensures the sender holds tensor references until
    the receiver completes the D2D copy.
    """

    supports_gpu_tensor: bool = True

    def __init__(self, config: dict[str, Any]):
        self._closed = False
        self.config = config
        self.stage_id = config.get("stage_id", -1)

        # Device configuration
        local_device = config.get("local_device", "auto")
        if local_device == "auto":
            self._local_device = None  # resolved lazily
        else:
            self._local_device = torch.device(local_device)

        # Tensor lifetime: fallback timeout for held tensor refs
        self._tensor_lifetime_sec = float(config.get("tensor_lifetime_sec", 30))

        # Role: sender or receiver (injected by orchestration layer)
        role = str(config.get("role", "sender")).lower()
        if role not in {"sender", "receiver"}:
            raise ValueError(f"Invalid role={role!r}. Expected 'sender' or 'receiver'.")
        self._role = role

        # Sender state: held tensor references keyed by put_key
        # Each entry: (tensor_refs: list[torch.Tensor], created_at: float)
        self._held_tensors: dict[str, tuple[list[torch.Tensor], float]] = {}
        self._held_lock = threading.Lock()

        # Start background cleanup thread for stale tensor refs
        self._stop_event = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="cuda-ipc-cleanup"
        )
        self._cleanup_thread.start()

        # Metrics
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "gpu_tensors_transferred": 0,
            "bytes_via_shm": 0,
            "ack_received": 0,
            "ttl_expired": 0,
        }

        # Validate P2P access (best-effort; may not have CUDA context yet)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self._validate_p2p_access()

        logger.info(
            f"CudaIPCConnector initialized: role={self._role}, "
            f"local_device={local_device}, tensor_lifetime={self._tensor_lifetime_sec}s"
        )

    def _validate_p2p_access(self) -> None:
        """Check that CUDA P2P access is available between GPUs."""
        n_devices = torch.cuda.device_count()
        no_p2p_pairs = []
        for i in range(n_devices):
            for j in range(i + 1, n_devices):
                if not torch.cuda.can_device_access_peer(i, j):
                    no_p2p_pairs.append((i, j))
        if no_p2p_pairs:
            logger.warning(
                f"No P2P access between GPU pairs: {no_p2p_pairs}. "
                f"D2D copy will fall back to PCIe staging (slower than NVLink)."
            )

    @property
    def local_device(self) -> torch.device:
        if self._local_device is None:
            self._local_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        return self._local_device

    # ------------------------------------------------------------------
    # put() — Sender side
    # ------------------------------------------------------------------

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """Store data for D2D transfer.

        GPU tensors in *data* (expected to be a dict) are extracted: for each
        tensor we obtain a CUDA IPC handle and record the tensor shape/dtype.
        The IPC metadata plus all non-tensor values are serialized and written
        to a shared memory segment.  The original GPU tensors are held alive
        in ``_held_tensors`` until the receiver ACKs.
        """
        if self._closed:
            raise RuntimeError("CudaIPCConnector is closed")

        internal_key = self._make_key(put_key, from_stage, to_stage)

        try:
            gpu_meta, cpu_data, tensor_refs = self._split_data(data)

            # Serialize the control payload: gpu metadata + cpu-only data
            control_payload = self.serialize_obj({
                "gpu_meta": gpu_meta,
                "cpu_data": cpu_data,
            })
            size = len(control_payload)

            # Write control payload to SHM
            shm_name = _shm_name(internal_key)
            self._shm_write(shm_name, control_payload)

            # Hold tensor references until receiver ACKs
            if tensor_refs:
                with self._held_lock:
                    # Release old refs if key already exists
                    old = self._held_tensors.pop(internal_key, None)
                    if old:
                        logger.warning(f"Overwriting held tensors for key={internal_key}")
                    self._held_tensors[internal_key] = (tensor_refs, _time_mod.monotonic())

            self._metrics["puts"] += 1
            self._metrics["bytes_via_shm"] += size

            metadata = {
                "shm_name": shm_name,
                "control_size": size,
                "has_gpu_tensors": len(gpu_meta) > 0,
            }
            return True, size, metadata

        except Exception as e:
            logger.error(f"CudaIPCConnector put failed for {internal_key}: {e}", exc_info=True)
            return False, 0, None

    def _split_data(self, data: Any) -> tuple[list[dict], dict, list[torch.Tensor]]:
        """Separate GPU tensors from CPU data in a dict payload.

        Returns:
            gpu_meta: list of {key, handle, shape, dtype, device} for each GPU tensor
            cpu_data: dict of non-tensor items + CPU tensors (moved to CPU for serialization)
            tensor_refs: list of original GPU tensors (to be held alive)
        """
        if not isinstance(data, dict):
            return [], data, []

        gpu_meta: list[dict] = []
        cpu_data: dict[str, Any] = {}
        tensor_refs: list[torch.Tensor] = []

        for key, value in data.items():
            if isinstance(value, torch.Tensor) and value.is_cuda:
                # Ensure tensor is contiguous for IPC
                t = value.contiguous()
                # Synchronize the stream to make sure tensor data is ready
                with torch.cuda.device(t.device):
                    torch.cuda.current_stream().synchronize()

                # Get IPC handle
                handle = t.storage()._share_cuda_()
                # handle is a tuple: (device, handle_bytes, storage_size_bytes, storage_offset_bytes)
                gpu_meta.append({
                    "key": key,
                    "ipc_handle": handle,
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "device": str(t.device),
                })
                tensor_refs.append(t)
            else:
                cpu_data[key] = value

        return gpu_meta, cpu_data, tensor_refs

    # ------------------------------------------------------------------
    # get() — Receiver side
    # ------------------------------------------------------------------

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        """Retrieve data via D2D copy.

        Reads the control payload from SHM, then for each GPU tensor:
        opens the CUDA IPC handle and performs a D2D copy to the local GPU.
        After all copies complete, sends an ACK via SHM.
        """
        if self._closed:
            raise RuntimeError("CudaIPCConnector is closed")

        internal_key = self._make_key(get_key, from_stage, to_stage)
        shm_name = _shm_name(internal_key)

        try:
            control_bytes = self._shm_read(shm_name)
            if control_bytes is None:
                return None

            control = self.deserialize_obj(control_bytes)
            gpu_meta = control.get("gpu_meta", [])
            cpu_data = control.get("cpu_data", {})
            size = len(control_bytes)

            # Reconstruct GPU tensors via IPC
            result = dict(cpu_data) if isinstance(cpu_data, dict) else cpu_data
            if gpu_meta and isinstance(result, dict):
                target_device = self.local_device
                for entry in gpu_meta:
                    key = entry["key"]
                    ipc_handle = entry["ipc_handle"]
                    shape = entry["shape"]
                    dtype = getattr(torch, entry["dtype"].replace("torch.", ""))

                    # Reconstruct storage from IPC handle
                    # ipc_handle is (device, handle_bytes, storage_size_bytes, storage_offset_bytes)
                    storage = torch.UntypedStorage._new_shared_cuda(
                        ipc_handle[0],  # device
                        ipc_handle[1],  # handle bytes
                        ipc_handle[2],  # storage size bytes
                        ipc_handle[3],  # storage offset bytes
                    )

                    # Create tensor from the shared storage (on sender's GPU)
                    src_tensor = torch.tensor([], dtype=dtype).set_(storage).reshape(shape)

                    # D2D copy to local GPU
                    dst_tensor = torch.empty(shape, dtype=dtype, device=target_device)
                    dst_tensor.copy_(src_tensor, non_blocking=True)

                    # Synchronize to ensure copy is complete before releasing
                    with torch.cuda.device(target_device):
                        torch.cuda.current_stream().synchronize()

                    result[key] = dst_tensor
                    self._metrics["gpu_tensors_transferred"] += 1

            # Send ACK to sender
            self._send_ack(internal_key)

            self._metrics["gets"] += 1
            self._metrics["bytes_via_shm"] += size
            return result, size

        except FileNotFoundError:
            # SHM segment not yet created by sender
            return None
        except Exception as e:
            logger.error(f"CudaIPCConnector get failed for {internal_key}: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # ACK mechanism
    # ------------------------------------------------------------------

    def _send_ack(self, internal_key: str) -> None:
        """Write an ACK to SHM to signal the sender that D2D copy is done."""
        ack_name = _shm_name(internal_key, suffix="_ack")
        try:
            self._shm_write(ack_name, _ACK_DONE)
        except Exception as e:
            logger.warning(f"Failed to send ACK for {internal_key}: {e}")

    def _check_ack(self, internal_key: str) -> bool:
        """Check whether the receiver has sent an ACK for this key."""
        ack_name = _shm_name(internal_key, suffix="_ack")
        try:
            data = self._shm_read(ack_name)
            return data == _ACK_DONE
        except Exception:
            return False

    def _release_held_tensors(self, internal_key: str) -> None:
        """Release held tensor references for a given key."""
        with self._held_lock:
            item = self._held_tensors.pop(internal_key, None)
        if item:
            self._metrics["ack_received"] += 1
            logger.debug(f"Released {len(item[0])} held tensor refs for {internal_key}")

    # ------------------------------------------------------------------
    # Background cleanup
    # ------------------------------------------------------------------

    def _cleanup_loop(self) -> None:
        """Periodically check for ACKs and release tensor refs."""
        while not self._stop_event.wait(timeout=1.0):
            self._process_acks_and_ttl()

    def _process_acks_and_ttl(self) -> None:
        """Check ACKs and enforce TTL on held tensors."""
        now = _time_mod.monotonic()
        keys_to_release: list[str] = []
        ttl_expired: list[str] = []

        with self._held_lock:
            for key, (refs, created_at) in list(self._held_tensors.items()):
                if self._check_ack(key):
                    keys_to_release.append(key)
                elif now - created_at > self._tensor_lifetime_sec:
                    ttl_expired.append(key)

        for key in keys_to_release:
            self._release_held_tensors(key)
            # Clean up ACK SHM segment
            self._shm_cleanup(_shm_name(key, suffix="_ack"))

        for key in ttl_expired:
            with self._held_lock:
                item = self._held_tensors.pop(key, None)
            if item:
                self._metrics["ttl_expired"] += 1
                logger.warning(
                    f"TTL expired ({self._tensor_lifetime_sec}s): force-released "
                    f"{len(item[0])} tensor refs for {key}"
                )

    # ------------------------------------------------------------------
    # SHM helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shm_write(name: str, data: bytes) -> None:
        """Write bytes to a named POSIX shared memory segment."""
        # Prepend 4-byte length header for reliable reading
        payload = struct.pack("<I", len(data)) + data
        try:
            shm = shm_pkg.SharedMemory(name=name, create=True, size=len(payload))
        except FileExistsError:
            # Segment already exists; unlink and recreate
            try:
                old_shm = shm_pkg.SharedMemory(name=name, create=False)
                old_shm.close()
                old_shm.unlink()
            except Exception:
                pass
            shm = shm_pkg.SharedMemory(name=name, create=True, size=len(payload))
        try:
            shm.buf[: len(payload)] = payload
        finally:
            shm.close()

    @staticmethod
    def _shm_read(name: str) -> bytes | None:
        """Read bytes from a named POSIX shared memory segment."""
        try:
            shm = shm_pkg.SharedMemory(name=name, create=False)
        except FileNotFoundError:
            return None
        try:
            length = struct.unpack("<I", bytes(shm.buf[:4]))[0]
            data = bytes(shm.buf[4 : 4 + length])
            return data
        finally:
            shm.close()
            try:
                shm.unlink()
            except Exception:
                pass

    @staticmethod
    def _shm_cleanup(name: str) -> None:
        """Best-effort removal of a SHM segment."""
        try:
            shm = shm_pkg.SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self, request_id: str) -> None:
        """Release resources for a specific request."""
        # Try to release any held tensors matching this request_id prefix
        keys_to_remove = []
        with self._held_lock:
            for key in self._held_tensors:
                if request_id in key:
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            self._release_held_tensors(key)
            self._shm_cleanup(_shm_name(key))
            self._shm_cleanup(_shm_name(key, suffix="_ack"))

    def health(self) -> dict[str, Any]:
        held_count = len(self._held_tensors)
        return {
            "status": "healthy" if not self._closed else "closed",
            "role": self._role,
            "held_tensor_groups": held_count,
            **self._metrics,
        }

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        logger.info("Closing CudaIPCConnector...")

        # Stop cleanup thread
        self._stop_event.set()
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=2.0)

        # Release all held tensor refs
        with self._held_lock:
            for key, (refs, _) in self._held_tensors.items():
                logger.debug(f"Close: releasing {len(refs)} refs for {key}")
            self._held_tensors.clear()

        # Collect CUDA IPC resources
        if torch.cuda.is_available():
            try:
                torch.cuda.ipc_collect()
            except Exception as e:
                logger.warning(f"torch.cuda.ipc_collect() failed: {e}")

        logger.info("CudaIPCConnector closed.")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA IPC Connector for single-node GPU-to-GPU (D2D) tensor transfer.

Split-plane architecture:
  - Data Plane: GPU tensors stay on sender GPU. CUDA IPC handles (64 bytes
    each) are passed to the receiver, which opens them and performs a D2D
    copy via NVLink / PCIe P2P using raw cudart APIs.
  - Control Plane: IPC handles + small CPU data serialized via /dev/shm
    (< 1 KB per chunk), reusing the existing shm_write_bytes/shm_read_bytes
    utilities for consistency with SharedMemoryConnector.

_CudaIpcMemHandle is a ctypes wrapper for the 64-byte opaque structure
``cudaIpcMemHandle_t`` defined in ``cuda_runtime_api.h``.  It is the
cross-process "ticket" that lets a receiver map the sender's GPU memory:

  Sender:  cudaIpcGetMemHandle(ptr)  ->  64-byte handle  --(via SHM)-->
  Receiver: cudaIpcOpenMemHandle(handle) -> mapped dev_ptr -> cudaMemcpy(D2D)

We wrap it in ctypes so Python can pass it to/from the CUDA runtime C API.
"""

import ctypes
import fcntl
import hashlib
import os
import threading
import time as _time_mod
from multiprocessing import shared_memory as shm_pkg
from typing import Any

import torch

from vllm_omni.entrypoints.stage_utils import shm_read_bytes, shm_write_bytes

from ..utils.logging import get_connector_logger
from ..utils.serialization import OmniSerializer
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

# Marker key embedded in serialized dicts to identify IPC tensor entries
_GPU_TENSOR_MARKER = "__cuda_ipc_tensor__"

# Default GPU memory budget for held tensors (bytes).
# When exceeded, put() falls back to CPU serialization to avoid OOM.
_DEFAULT_MAX_HELD_BYTES = 2 * 1024**3  # 2 GB


class _CudaIpcMemHandle(ctypes.Structure):
    """ctypes wrapper for ``cudaIpcMemHandle_t`` (64-byte opaque struct).

    This is the cross-process "ticket" that lets a receiver map the sender's
    GPU memory.  We need the ctypes struct so Python can correctly pass it
    by reference to cudaIpcGetMemHandle / cudaIpcOpenMemHandle.
    """

    _fields_ = [("reserved", ctypes.c_char * 64)]


class CudaIPCConnector(OmniConnectorBase):
    """Single-node CUDA IPC connector for stage-to-stage payload transfer.

    GPU tensors are transferred using raw cudaIpcGetMemHandle /
    cudaIpcOpenMemHandle calls (via ctypes) for maximum stability across
    PyTorch versions.  Non-tensor data is serialized via OmniSerializer and
    passed through /dev/shm.

    An ACK-based lifecycle ensures the sender holds tensor references until
    the receiver completes the D2D copy.

    Memory backpressure:
        ``max_held_bytes`` caps the total GPU memory pinned by un-ACK'd
        tensors.  When the limit is reached, ``put()`` automatically falls
        back to the CPU serialization path (same as SharedMemoryConnector),
        trading latency for safety against OOM.
    """

    supports_gpu_tensor: bool = True

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = int(config.get("stage_id", -1))
        self.role = str(config.get("role", "sender")).lower()
        if self.role not in {"sender", "receiver"}:
            raise ValueError(f"Invalid role={self.role!r}. Expected 'sender' or 'receiver'.")
        self.tensor_lifetime_sec = float(config.get("tensor_lifetime_sec", 30.0))
        self.local_device = self._resolve_local_device(config.get("local_device", "auto"))
        self._closed = False
        self._cudart = None

        # --- Memory backpressure (Q7) ---
        self._max_held_bytes = int(config.get("max_held_bytes", _DEFAULT_MAX_HELD_BYTES))
        self._held_bytes = 0  # current total GPU bytes pinned by _held_tensors

        # Sender state: held tensor references keyed by *composite key*
        # (includes stage info to avoid SHM collisions — Q2).
        # Each entry: (created_at, holders, total_nbytes)
        self._held_tensors: dict[str, tuple[float, list[torch.Tensor], int]] = {}
        self._held_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ack_thread: threading.Thread | None = None

        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "gpu_tensors_transferred": 0,
            "acks": 0,
            "ack_timeouts": 0,
            "errors": 0,
            "cpu_fallbacks": 0,
        }

        if not torch.cuda.is_available():
            raise RuntimeError("CudaIPCConnector requires CUDA runtime.")
        self._cudart = torch.cuda.cudart()

        # Validate P2P access (best-effort; may not have CUDA context yet)
        if torch.cuda.device_count() > 1:
            self._validate_p2p_access()

        # Sender starts background ACK drain thread
        if self.role == "sender":
            self._ack_thread = threading.Thread(
                target=self._ack_loop, daemon=True, name="cuda-ipc-ack-loop"
            )
            self._ack_thread.start()

        logger.info(
            f"CudaIPCConnector initialized: role={self.role}, "
            f"local_device={self.local_device}, tensor_lifetime={self.tensor_lifetime_sec}s, "
            f"max_held_bytes={self._max_held_bytes / 1024**2:.0f}MB"
        )

    # ------------------------------------------------------------------
    # Device & naming helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_local_device(local_device_cfg: str | int) -> torch.device:
        if local_device_cfg == "auto":
            return torch.device("cuda", torch.cuda.current_device())
        if isinstance(local_device_cfg, int):
            return torch.device("cuda", local_device_cfg)
        if isinstance(local_device_cfg, str):
            if local_device_cfg.startswith("cuda"):
                return torch.device(local_device_cfg)
            return torch.device("cuda", int(local_device_cfg))
        return torch.device("cuda", torch.cuda.current_device())

    @staticmethod
    def _safe_name(prefix: str, key: str) -> str:
        """Generate a short, collision-free SHM name via SHA1 hash."""
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
        return f"{prefix}_{digest}"

    def _payload_name(self, composite_key: str) -> str:
        """SHM name for the payload.  *composite_key* includes stage info (Q2)."""
        return self._safe_name("cudaipc", composite_key)

    def _ack_name(self, composite_key: str) -> str:
        """SHM name for the ACK signal."""
        return self._safe_name("cudaipc_ack", composite_key)

    def _lock_file(self, name: str) -> str:
        return f"/dev/shm/{name}.lock"

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

    # ------------------------------------------------------------------
    # Low-level CUDA IPC via ctypes (stable across PyTorch versions)
    # ------------------------------------------------------------------

    def _get_ipc_handle(self, ptr: int) -> bytes:
        """Obtain a 64-byte CUDA IPC memory handle for a device pointer."""
        handle = _CudaIpcMemHandle()
        ret = self._cudart.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(ptr))
        if ret != 0:
            raise RuntimeError(f"cudaIpcGetMemHandle failed with code {ret}")
        return bytes(handle.reserved)

    def _open_ipc_ptr(self, handle_bytes: bytes) -> ctypes.c_void_p:
        """Open a CUDA IPC handle and return the mapped device pointer."""
        handle = _CudaIpcMemHandle()
        handle.reserved = handle_bytes
        dev_ptr = ctypes.c_void_p()
        # 1 = cudaIpcMemLazyEnablePeerAccess
        ret = self._cudart.cudaIpcOpenMemHandle(
            ctypes.byref(dev_ptr), handle, ctypes.c_uint(1)
        )
        if ret != 0:
            raise RuntimeError(f"cudaIpcOpenMemHandle failed with code {ret}")
        return dev_ptr

    def _close_ipc_ptr(self, dev_ptr: ctypes.c_void_p) -> None:
        """Release a mapped CUDA IPC device pointer."""
        ret = self._cudart.cudaIpcCloseMemHandle(dev_ptr)
        if ret != 0:
            logger.warning("cudaIpcCloseMemHandle failed with code %s", ret)

    def _d2d_copy(self, dst_ptr: int, src_ptr: ctypes.c_void_p, nbytes: int) -> None:
        """Perform a D2D memcpy via cudart."""
        # cudaMemcpyDeviceToDevice = 3
        ret = self._cudart.cudaMemcpy(
            ctypes.c_void_p(dst_ptr), src_ptr, ctypes.c_size_t(nbytes), ctypes.c_int(3)
        )
        if ret != 0:
            raise RuntimeError(f"cudaMemcpy(D2D) failed with code {ret}")

    # ------------------------------------------------------------------
    # Recursive encode / decode (handles nested dict/list/tuple)
    # ------------------------------------------------------------------

    def _encode_gpu_tensor(self, tensor: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor]:
        """Encode a single CUDA tensor into an IPC metadata dict.

        ``.contiguous()`` is required because IPC handles point to a single
        contiguous allocation.  If the input is already contiguous the call
        is a no-op (returns the same storage, no GPU copy).  The returned
        *holder* tensor is stored in ``_held_tensors`` to keep the underlying
        CUDA memory alive until the receiver ACKs.
        """
        t = tensor.detach().contiguous()
        handle = self._get_ipc_handle(t.data_ptr())
        return {
            _GPU_TENSOR_MARKER: True,
            "shape": list(t.shape),
            "dtype": str(t.dtype).removeprefix("torch."),
            "nbytes": int(t.nbytes),
            "handle": handle,
        }, t

    def _walk_encode(self, obj: Any, holders: list[torch.Tensor]) -> Any:
        """Recursively walk a data structure, replacing CUDA tensors with IPC metadata."""
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            encoded, holder = self._encode_gpu_tensor(obj)
            holders.append(holder)
            return encoded
        if isinstance(obj, dict):
            return {k: self._walk_encode(v, holders) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._walk_encode(v, holders) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._walk_encode(v, holders) for v in obj)
        return obj

    def _decode_gpu_tensor(self, meta: dict[str, Any]) -> torch.Tensor:
        """Decode an IPC metadata dict into a local GPU tensor via D2D copy."""
        shape = tuple(meta["shape"])
        dtype = getattr(torch, meta["dtype"])
        nbytes = int(meta["nbytes"])
        dst = torch.empty(shape, dtype=dtype, device=self.local_device)
        src_ptr = self._open_ipc_ptr(meta["handle"])
        try:
            self._d2d_copy(dst.data_ptr(), src_ptr, nbytes)
        finally:
            self._close_ipc_ptr(src_ptr)
        self._metrics["gpu_tensors_transferred"] += 1
        return dst

    def _walk_decode(self, obj: Any) -> Any:
        """Recursively walk a data structure, restoring CUDA tensors from IPC metadata."""
        if isinstance(obj, dict) and obj.get(_GPU_TENSOR_MARKER):
            return self._decode_gpu_tensor(obj)
        if isinstance(obj, dict):
            return {k: self._walk_decode(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._walk_decode(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._walk_decode(v) for v in obj)
        return obj

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

        GPU tensors in *data* are recursively replaced with CUDA IPC handle
        metadata.  The transformed structure is serialized via OmniSerializer
        and written to a shared memory segment.  Original GPU tensors are held
        alive in ``_held_tensors`` until the receiver ACKs.

        If held GPU memory exceeds ``max_held_bytes``, the method
        automatically falls back to full CPU serialization (same path as
        SharedMemoryConnector) to avoid OOM.
        """
        if self._closed:
            return False, 0, None

        # Build composite key that includes stage routing info (Q2:
        # prevents SHM name collision when different stage pairs share
        # the same put_key).
        composite_key = f"{put_key}@{from_stage}_{to_stage}"

        try:
            # Opportunistically drain ACKs to release memory early
            self._drain_acks()

            # --- Memory backpressure check (Q7) ---
            if self._held_bytes >= self._max_held_bytes:
                return self._put_cpu_fallback(from_stage, to_stage, put_key, composite_key, data)

            holders: list[torch.Tensor] = []
            encoded_obj = self._walk_encode(data, holders)
            payload = OmniSerializer.serialize(encoded_obj)
            size = len(payload)

            # Compute total GPU bytes for this batch of holders
            batch_nbytes = sum(t.nbytes for t in holders)

            # Re-check after encoding (encoding itself is fast, but another
            # thread may have added entries between our check and here).
            if holders and self._held_bytes + batch_nbytes > self._max_held_bytes:
                return self._put_cpu_fallback(from_stage, to_stage, put_key, composite_key, data)

            # --- Q1 fix: store holders BEFORE writing SHM ---
            # This eliminates the window where the handle is visible to the
            # receiver but the underlying tensor could theoretically be GC'd.
            if holders:
                with self._held_lock:
                    self._held_tensors[composite_key] = (_time_mod.time(), holders, batch_nbytes)
                    self._held_bytes += batch_nbytes

            payload_name = self._payload_name(composite_key)
            lock_file = self._lock_file(payload_name)
            try:
                with open(lock_file, "wb+") as lockf:
                    fcntl.flock(lockf, fcntl.LOCK_EX)
                    meta = shm_write_bytes(payload, name=payload_name)
                    fcntl.flock(lockf, fcntl.LOCK_UN)
            except Exception:
                # SHM write failed — roll back the held tensors
                if holders:
                    with self._held_lock:
                        popped = self._held_tensors.pop(composite_key, None)
                        if popped:
                            self._held_bytes -= popped[2]
                raise

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size
            return True, size, {"shm": meta, "size": size}

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("CudaIPCConnector put failed for %s: %s", put_key, e, exc_info=True)
            return False, 0, None

    def _put_cpu_fallback(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        composite_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """Fallback: serialize everything to CPU and write to SHM.

        Used when held GPU memory exceeds ``max_held_bytes`` to prevent OOM.
        The receiver's ``_walk_decode`` will see no IPC markers and just
        return the deserialized CPU tensors — no D2D copy needed.
        """
        logger.warning(
            "CudaIPCConnector: held GPU memory (%.1f MB) >= limit (%.1f MB), "
            "falling back to CPU serialization for %s",
            self._held_bytes / 1024**2,
            self._max_held_bytes / 1024**2,
            put_key,
        )
        self._metrics["cpu_fallbacks"] += 1
        # Use the base class serializer (same as SharedMemoryConnector):
        # all tensors get .detach().cpu() inside OmniSerializer.
        payload = self.serialize_obj(data)
        size = len(payload)

        payload_name = self._payload_name(composite_key)
        lock_file = self._lock_file(payload_name)
        with open(lock_file, "wb+") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            meta = shm_write_bytes(payload, name=payload_name)
            fcntl.flock(lockf, fcntl.LOCK_UN)

        self._metrics["puts"] += 1
        self._metrics["bytes_transferred"] += size
        return True, size, {"shm": meta, "size": size, "cpu_fallback": True}

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

        Reads the control payload from SHM, then recursively restores CUDA
        tensors by opening IPC handles and performing D2D copies to the local
        GPU.  After all copies complete, sends an ACK via SHM so the sender
        can release its tensor references.

        Transparently handles CPU-fallback payloads: if the data was serialized
        by ``_put_cpu_fallback`` (no IPC markers), ``_walk_decode`` returns
        the deserialized objects unchanged.
        """
        if self._closed:
            return None

        # Must match the composite_key format used in put() (Q2)
        composite_key = f"{get_key}@{from_stage}_{to_stage}"
        payload_name = self._payload_name(composite_key)
        lock_file = self._lock_file(payload_name)
        try:
            with open(lock_file, "rb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                seg = shm_pkg.SharedMemory(name=payload_name)
                try:
                    shm_handle = {"name": payload_name, "size": seg.size}
                finally:
                    seg.close()
                data_bytes = shm_read_bytes(shm_handle)
                fcntl.flock(lockf, fcntl.LOCK_UN)

            if os.path.exists(lock_file):
                os.remove(lock_file)

            raw_obj = OmniSerializer.deserialize(data_bytes)
            obj = self._walk_decode(raw_obj)
            torch.cuda.synchronize(self.local_device)
            self._send_ack(composite_key)

            size = len(data_bytes)
            self._metrics["gets"] += 1
            self._metrics["bytes_transferred"] += size
            return obj, size

        except FileNotFoundError:
            return None
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("CudaIPCConnector get failed for %s: %s", get_key, e, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # ACK mechanism
    # ------------------------------------------------------------------

    def _send_ack(self, composite_key: str) -> None:
        """Write an ACK to SHM to signal the sender that D2D copy is done."""
        ack_name = self._ack_name(composite_key)
        try:
            shm_write_bytes(b"1", name=ack_name)
        except Exception as e:
            logger.debug("Failed to write ACK for %s: %s", composite_key, e)

    def _has_ack(self, composite_key: str) -> bool:
        """Check whether the receiver has sent an ACK for this key."""
        ack_name = self._ack_name(composite_key)
        try:
            seg = shm_pkg.SharedMemory(name=ack_name)
            try:
                handle = {"name": ack_name, "size": seg.size}
            finally:
                seg.close()
            shm_read_bytes(handle)
            return True
        except Exception:
            return False

    def _drain_acks(self) -> None:
        """Scan held tensors: release on ACK or TTL expiry."""
        now = _time_mod.time()
        to_release: list[str] = []
        with self._held_lock:
            for key, (ts, _holders, _nbytes) in self._held_tensors.items():
                if self._has_ack(key):
                    to_release.append(key)
                    self._metrics["acks"] += 1
                elif now - ts > self.tensor_lifetime_sec:
                    to_release.append(key)
                    self._metrics["ack_timeouts"] += 1
            for key in to_release:
                popped = self._held_tensors.pop(key, None)
                if popped:
                    self._held_bytes -= popped[2]

    def _ack_loop(self) -> None:
        """Background loop that periodically drains ACKs.

        Interval: 100ms — aligned with the transfer adapter's recv_loop
        back-off (Q3).  Combined with the synchronous ``_drain_acks()``
        call at the top of every ``put()``, this gives ~10ms effective
        latency for releasing held tensors under steady load.
        """
        while not self._stop_event.is_set():
            try:
                self._drain_acks()
            except Exception as e:
                logger.debug("ACK loop error: %s", e)
            self._stop_event.wait(timeout=0.1)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self, request_id: str) -> None:
        """Release resources for a specific request."""
        with self._held_lock:
            keys_to_remove = [
                k for k in self._held_tensors
                if k.startswith(f"{request_id}_")
            ]
            for k in keys_to_remove:
                popped = self._held_tensors.pop(k, None)
                if popped:
                    self._held_bytes -= popped[2]

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy" if not self._closed else "closed",
            "role": self.role,
            "local_device": str(self.local_device),
            "held_tensors": len(self._held_tensors),
            "held_bytes_mb": round(self._held_bytes / 1024**2, 1),
            "max_held_bytes_mb": round(self._max_held_bytes / 1024**2, 1),
            **self._metrics,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        logger.info("Closing CudaIPCConnector...")

        self._stop_event.set()
        if self._ack_thread is not None and self._ack_thread.is_alive():
            self._ack_thread.join(timeout=1.0)

        with self._held_lock:
            self._held_tensors.clear()
            self._held_bytes = 0

        if torch.cuda.is_available():
            try:
                torch.cuda.ipc_collect()
            except Exception as e:
                logger.warning(f"torch.cuda.ipc_collect() failed: {e}")

        logger.info("CudaIPCConnector closed.")

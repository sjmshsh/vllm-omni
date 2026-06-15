# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA IPC Connector with pre-allocated memory pool for GPU-to-GPU transfer.

Architecture:
  - A fixed GPU memory pool is allocated once at init on the sender side.
    Its IPC handle is registered once — no per-tensor cudaIpcGetMemHandle.
  - Credit-based flow control: the pool is divided into N slots. Each put()
    acquires a slot, copies tensors into it, and releases it on ACK.
  - The receiver opens the pool IPC handle once (cached), reads from offsets.
    No per-tensor cudaIpcOpenMemHandle / cudaIpcCloseMemHandle.
  - Control plane: tensor layout metadata serialized via /dev/shm, same as
    before (< 1 KB per chunk).

CUDA-graph compatibility:
  - Both sender and receiver use dedicated non-blocking copy streams
    for D2D transfers, keeping each side's compute/default stream free
    for forward-pass kernels and CUDA-graph replay.
  - Sender: records compute-stream event → copy_stream waits on it →
    .copy_() on copy_stream → records IPC event (no CPU synchronize).
  - Receiver: opens sender's IPC event (cached) → cudaStreamWaitEvent
    on current_stream → cudaMemcpyAsync from pool → stream synchronize.
  - Credit release uses a shared-memory "release board" (1 byte per slot):
    the receiver flips the slot byte after its copies complete; the sender
    reclaims credits by reading the board — plain memory reads, no per-key
    /dev/shm ACK file syscalls and no 100 ms polling latency. A TTL sweep
    reclaims slots whose receiver died before marking the board.

Flow control: when all credits are in flight, put() falls back to plain
CPU serialization via /dev/shm (the SharedMemoryConnector wire format),
which the receiver reads through the same key-based lookup. Transfers
never block on pool capacity.
"""

import collections
import ctypes
import hashlib
import os
import queue as _queue_mod
import threading
import time as _time_mod
import uuid
from multiprocessing import shared_memory as shm_pkg
from typing import Any

import torch

from vllm_omni.entrypoints.stage_utils import shm_write_bytes

from ..utils.logging import get_connector_logger
from ..utils.serialization import OmniSerializer
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

_GPU_TENSOR_MARKER = "__cuda_ipc_tensor__"
_POOL_MARKER = "__cuda_ipc_pool__"

_POOL_ALIGNMENT = 16  # bytes, for GPU copy efficiency

# Pool defaults: auto-sized when user omits pool_size_mb / pool_credits.
# Auto formula: credits = max(64, max_num_seqs * 4), size = credits * 2 MB.
# Explicit extra config values override auto-sizing.
_DEFAULT_POOL_SIZE_MB = 128
_DEFAULT_POOL_CREDITS = 64

# CUDA runtime API constants (fixed by CUDA spec, not configurable).
_CUDA_MEMCPY_D2D = 3  # cudaMemcpyDeviceToDevice

# Timing constants — overridable via extra config keys of the same name
# (without leading underscore), e.g. ``"credit_wait_sec": 0.01``.
_CREDIT_WAIT_SEC = 0.05  # put() inline reclaim window before CPU fallback
_CREDIT_POLL_SEC = 0.0005  # poll interval within the reclaim window
_RELEASE_FAST_INTERVAL_SEC = 0.001  # board-reclaim thread fast tick
_RELEASE_TTL_EVERY_N_TICKS = 20  # TTL sweep runs every N fast ticks


class _CudaIpcMemHandle(ctypes.Structure):
    """ctypes wrapper for ``cudaIpcMemHandle_t`` (64-byte opaque struct)."""

    _fields_ = [("reserved", ctypes.c_char * 64)]


class _CudaIpcEventHandle(ctypes.Structure):
    """ctypes wrapper for ``cudaIpcEventHandle_t`` (64-byte opaque struct)."""

    _fields_ = [("reserved", ctypes.c_char * 64)]


_CUDA_EVENT_INTERPROCESS = 0x04
_CUDA_EVENT_DISABLE_TIMING = 0x02


class _SlotOverflowError(Exception):
    """Raised when tensors exceed a pool slot's capacity."""


class _PoolSlot:
    """Tracks packing state for tensors within a single pool credit slot."""

    __slots__ = ("_pool", "_base", "_size", "_cursor")

    def __init__(self, pool: torch.Tensor, slot_offset: int, slot_size: int):
        self._pool = pool
        self._base = slot_offset
        self._size = slot_size
        self._cursor = 0

    def pack(self, tensor: torch.Tensor) -> int:
        """Copy tensor into the pool slot via PyTorch .copy_(), return byte offset."""
        nbytes = tensor.nbytes
        padding = (-self._cursor) % _POOL_ALIGNMENT
        aligned = self._cursor + padding
        if aligned + nbytes > self._size:
            raise _SlotOverflowError()
        src_bytes = tensor.view(torch.uint8).reshape(-1)
        self._pool[self._base + aligned : self._base + aligned + nbytes].copy_(src_bytes)
        self._cursor = aligned + nbytes
        return aligned


class CudaIPCConnector(OmniConnectorBase):
    """CUDA IPC connector with pre-allocated memory pool.

    Sender pre-allocates a GPU memory pool, registers its IPC handle once,
    and divides it into credit-managed slots. Each put() copies tensors into
    a slot and sends the offset via SHM. The receiver opens the pool handle
    once (cached) and copies from the offset — no per-tensor IPC overhead.
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

        self._held_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._release_thread: threading.Thread | None = None
        self._shm_compat_decode_failures: dict[str, int] = {}

        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "gpu_tensors_transferred": 0,
            "board_releases": 0,
            "ttl_releases": 0,
            "errors": 0,
            "cpu_fallbacks": 0,
        }

        if not torch.cuda.is_available():
            raise RuntimeError("CudaIPCConnector requires CUDA runtime.")
        self._cudart = self._load_cudart()

        # --- Memory pool (sender side) ---
        # Auto-size from max_num_seqs when user omits explicit values.
        max_num_seqs = int(config.get("max_num_seqs", 0))
        if max_num_seqs > 0 and "pool_credits" not in config:
            auto_credits = max(64, max_num_seqs * 4)
            auto_size_mb = auto_credits * 2
        else:
            auto_credits = _DEFAULT_POOL_CREDITS
            auto_size_mb = _DEFAULT_POOL_SIZE_MB
        pool_size_mb = int(config.get("pool_size_mb", auto_size_mb))
        pool_credits = int(config.get("pool_credits", auto_credits))

        # Timing overrides via extra config.
        self._credit_wait_sec = float(config.get("credit_wait_sec", _CREDIT_WAIT_SEC))
        self._credit_poll_sec = float(config.get("credit_poll_sec", _CREDIT_POLL_SEC))
        self._release_interval_sec = float(config.get("release_fast_interval_sec", _RELEASE_FAST_INTERVAL_SEC))
        self._release_ttl_every = int(config.get("release_ttl_every_n_ticks", _RELEASE_TTL_EVERY_N_TICKS))
        self._pool_size = pool_size_mb * 1024 * 1024
        self._slot_size = self._pool_size // pool_credits
        self._pool_credits = pool_credits

        if self.role == "sender":
            with torch.cuda.device(self.local_device):
                self._pool = torch.zeros(self._pool_size, dtype=torch.uint8, device=self.local_device)
                self._copy_stream = torch.cuda.Stream(device=self.local_device)
                self._compute_event = torch.cuda.Event()
                self._copy_done_event = torch.cuda.Event()
            self._pool_handle = self._get_ipc_handle(self._pool.data_ptr())
            self._ipc_event = ctypes.c_void_p()
            flags = ctypes.c_uint(_CUDA_EVENT_INTERPROCESS | _CUDA_EVENT_DISABLE_TIMING)
            ret = self._cudart.cudaEventCreateWithFlags(ctypes.byref(self._ipc_event), flags)
            if ret != 0:
                raise RuntimeError(f"cudaEventCreateWithFlags failed: {ret}")
            ipc_evt_handle = _CudaIpcEventHandle()
            ret = self._cudart.cudaIpcGetEventHandle(ctypes.byref(ipc_evt_handle), self._ipc_event)
            if ret != 0:
                raise RuntimeError(f"cudaIpcGetEventHandle failed: {ret}")
            self._ipc_event_handle_bytes = bytes(ipc_evt_handle)
            self._credit_queue: _queue_mod.Queue[int] = _queue_mod.Queue(maxsize=pool_credits)
            for i in range(pool_credits):
                self._credit_queue.put_nowait(i * self._slot_size)
            self._held_credits: dict[str, tuple[float, int]] = {}
            self._board_name = f"cudaipc_board_{uuid.uuid4().hex[:16]}"
            self._board = shm_pkg.SharedMemory(create=True, size=pool_credits, name=self._board_name)
            self._board.buf[:pool_credits] = bytes(pool_credits)
        else:
            self._pool = None
            self._pool_handle = None
            self._credit_queue = None
            self._held_credits = {}
            self._board_name = None
            self._board = None
            _NUM_RECV_STREAMS = 8
            with torch.cuda.device(self.local_device):
                self._recv_copy_streams = [
                    torch.cuda.Stream(device=self.local_device) for _ in range(_NUM_RECV_STREAMS)
                ]
                self._recv_copy_events = [torch.cuda.Event() for _ in range(_NUM_RECV_STREAMS)]
            self._recv_stream_idx = 0
            self._deferred_credit_releases: collections.deque = collections.deque()

        # Receiver: cache opened pool IPC mappings / sender release boards / IPC events
        self._opened_pools: dict[bytes, ctypes.c_void_p] = {}
        self._opened_boards: dict[str, shm_pkg.SharedMemory] = {}
        self._opened_events: dict[bytes, ctypes.c_void_p] = {}

        if torch.accelerator.device_count() > 1:
            self._validate_p2p_access()

        if self.role == "sender":
            self._release_thread = threading.Thread(
                target=self._release_loop, daemon=True, name="cuda-ipc-release-loop"
            )
            self._release_thread.start()

        logger.info(
            f"CudaIPCConnector initialized: role={self.role}, "
            f"local_device={self.local_device}, "
            f"pool={pool_size_mb}MB ({pool_credits} credits × {self._slot_size // 1024 // 1024}MB slots), "
            f"config_keys={sorted(config.keys())}"
        )

    # ------------------------------------------------------------------
    # Device & naming helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_local_device(local_device_cfg: str | int) -> torch.device:
        if local_device_cfg == "auto":
            return torch.device("cuda", torch.accelerator.current_device_index())
        if isinstance(local_device_cfg, int):
            return torch.device("cuda", local_device_cfg)
        if isinstance(local_device_cfg, str):
            if local_device_cfg.startswith("cuda"):
                return torch.device(local_device_cfg)
            return torch.device("cuda", int(local_device_cfg))
        return torch.device("cuda", torch.accelerator.current_device_index())

    @staticmethod
    def _safe_name(prefix: str, key: str) -> str:
        """Generate a short, collision-free SHM name via SHA1 hash."""
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
        return f"{prefix}_{digest}"

    def _payload_name(self, composite_key: str) -> str:
        return self._safe_name("cudaipc", composite_key)

    @staticmethod
    def _atomic_shm_write(payload: bytes, name: str) -> dict[str, Any]:
        """Write to SHM atomically: write to temp name, then rename."""
        from multiprocessing.resource_tracker import unregister

        tmp_name = f"{name}__tmp"
        meta = shm_write_bytes(payload, name=tmp_name)
        os.rename(f"/dev/shm/{tmp_name}", f"/dev/shm/{name}")
        try:
            unregister(f"/{tmp_name}", "shared_memory")
        except KeyError:
            pass
        meta["name"] = name
        return meta

    def _validate_p2p_access(self) -> None:
        n_devices = torch.accelerator.device_count()
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
    # Low-level CUDA IPC via ctypes
    # ------------------------------------------------------------------

    @staticmethod
    def _load_cudart():
        """Load libcudart with IPC symbols and signatures."""
        import ctypes.util
        import glob

        lib = None
        name = ctypes.util.find_library("cudart")
        if name:
            try:
                lib = ctypes.CDLL(name)
                if not hasattr(lib, "cudaIpcGetMemHandle"):
                    lib = None
            except OSError:
                lib = None

        if lib is None:
            candidates = sorted(
                glob.glob("/usr/local/cuda*/lib*/libcudart.so*") + glob.glob("/opt/conda/lib/libcudart.so*"),
                reverse=True,
            )
            for path in candidates:
                try:
                    lib = ctypes.CDLL(path)
                    if hasattr(lib, "cudaIpcGetMemHandle"):
                        break
                    lib = None
                except OSError:
                    continue

        if lib is None:
            raise RuntimeError(
                "Cannot find libcudart.so with cudaIpcGetMemHandle. "
                "Ensure CUDA toolkit is installed and libcudart.so is on LD_LIBRARY_PATH."
            )

        lib.cudaIpcGetMemHandle.argtypes = [
            ctypes.POINTER(_CudaIpcMemHandle),
            ctypes.c_void_p,
        ]
        lib.cudaIpcGetMemHandle.restype = ctypes.c_int

        lib.cudaIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            _CudaIpcMemHandle,
            ctypes.c_uint,
        ]
        lib.cudaIpcOpenMemHandle.restype = ctypes.c_int

        lib.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        lib.cudaIpcCloseMemHandle.restype = ctypes.c_int

        lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.cudaMemcpyAsync.restype = ctypes.c_int

        lib.cudaEventCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        lib.cudaEventCreateWithFlags.restype = ctypes.c_int

        lib.cudaIpcGetEventHandle.argtypes = [ctypes.POINTER(_CudaIpcEventHandle), ctypes.c_void_p]
        lib.cudaIpcGetEventHandle.restype = ctypes.c_int

        lib.cudaIpcOpenEventHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _CudaIpcEventHandle]
        lib.cudaIpcOpenEventHandle.restype = ctypes.c_int

        lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.cudaEventRecord.restype = ctypes.c_int

        lib.cudaStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        lib.cudaStreamWaitEvent.restype = ctypes.c_int

        lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
        lib.cudaEventDestroy.restype = ctypes.c_int

        return lib

    def _get_ipc_handle(self, ptr: int) -> bytes:
        """Obtain a 64-byte CUDA IPC memory handle for a device pointer."""
        handle = _CudaIpcMemHandle()
        ret = self._cudart.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(ptr))
        if ret != 0:
            raise RuntimeError(f"cudaIpcGetMemHandle failed with code {ret}")
        return bytes(handle)

    def _open_ipc_ptr(self, handle_bytes: bytes) -> ctypes.c_void_p:
        """Open a CUDA IPC handle and return the mapped device pointer."""
        handle = _CudaIpcMemHandle.from_buffer_copy(handle_bytes)
        dev_ptr = ctypes.c_void_p()
        ret = self._cudart.cudaIpcOpenMemHandle(ctypes.byref(dev_ptr), handle, ctypes.c_uint(1))
        if ret != 0:
            raise RuntimeError(f"cudaIpcOpenMemHandle failed with code {ret}")
        return dev_ptr

    def _close_ipc_ptr(self, dev_ptr: ctypes.c_void_p) -> None:
        ret = self._cudart.cudaIpcCloseMemHandle(dev_ptr)
        if ret != 0:
            logger.warning("cudaIpcCloseMemHandle failed with code %s", ret)

    def _open_pool(self, pool_handle: bytes) -> ctypes.c_void_p:
        """Open a pool IPC handle (cached — only opened once per sender)."""
        if pool_handle not in self._opened_pools:
            self._opened_pools[pool_handle] = self._open_ipc_ptr(pool_handle)
        return self._opened_pools[pool_handle]

    def _open_ipc_event(self, handle_bytes: bytes) -> ctypes.c_void_p:
        """Open a CUDA IPC event handle (cached — only opened once per sender)."""
        if handle_bytes not in self._opened_events:
            handle = _CudaIpcEventHandle.from_buffer_copy(handle_bytes)
            event = ctypes.c_void_p()
            ret = self._cudart.cudaIpcOpenEventHandle(ctypes.byref(event), handle)
            if ret != 0:
                raise RuntimeError(f"cudaIpcOpenEventHandle failed: {ret}")
            self._opened_events[handle_bytes] = event
        return self._opened_events[handle_bytes]

    # ------------------------------------------------------------------
    # Pool-based encode / decode
    # ------------------------------------------------------------------

    def _walk_encode_pool(self, obj: Any, slot: _PoolSlot) -> Any:
        """Recursively replace CUDA tensors with pool offset metadata."""
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                t = obj.detach().contiguous()
                tensor_offset = slot.pack(t)
                return {
                    _GPU_TENSOR_MARKER: True,
                    "shape": list(t.shape),
                    "dtype": str(t.dtype).removeprefix("torch."),
                    "nbytes": int(t.nbytes),
                    "pool_offset": tensor_offset,
                }
            return obj
        if isinstance(obj, dict):
            return {k: self._walk_encode_pool(v, slot) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._walk_encode_pool(v, slot) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._walk_encode_pool(v, slot) for v in obj)
        if hasattr(obj, "__struct_fields__"):
            return {
                f: self._walk_encode_pool(getattr(obj, f), slot)
                for f in obj.__struct_fields__
                if getattr(obj, f) is not None
            }
        return obj

    def _decode_pool_tensor(
        self,
        meta: dict[str, Any],
        pool_ptr: ctypes.c_void_p,
        slot_offset: int,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Decode a tensor from a cached pool mapping on *stream*.

        If *stream* is None, falls back to the current (compute) stream for
        backward compatibility.
        """
        shape = tuple(meta["shape"])
        dtype = getattr(torch, meta["dtype"])
        nbytes = int(meta["nbytes"])
        tensor_offset = int(meta["pool_offset"])
        dst = torch.empty(shape, dtype=dtype, device=self.local_device)
        src = ctypes.c_void_p(pool_ptr.value + slot_offset + tensor_offset)
        target_stream = stream if stream is not None else torch.cuda.current_stream(self.local_device)
        # dst was allocated on the compute stream but is written on
        # target_stream (recv copy-stream). Tell the caching allocator dst is
        # in use on target_stream so the block is not reaped/reused while the
        # async D2D is still in flight on that stream.
        if stream is not None:
            dst.record_stream(target_stream)
        ret = self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(dst.data_ptr()),
            src,
            ctypes.c_size_t(nbytes),
            ctypes.c_int(_CUDA_MEMCPY_D2D),
            ctypes.c_void_p(target_stream.cuda_stream),
        )
        if ret != 0:
            raise RuntimeError(f"cudaMemcpyAsync (pool decode) failed with code {ret}")
        self._metrics["gpu_tensors_transferred"] += 1
        return dst

    def _walk_decode_pool(
        self,
        obj: Any,
        pool_ptr: ctypes.c_void_p,
        slot_offset: int,
        stream: torch.cuda.Stream | None = None,
    ) -> Any:
        """Recursively restore tensors from pool offset metadata."""
        if isinstance(obj, dict) and obj.get(_GPU_TENSOR_MARKER):
            return self._decode_pool_tensor(obj, pool_ptr, slot_offset, stream=stream)
        if isinstance(obj, dict):
            return {k: self._walk_decode_pool(v, pool_ptr, slot_offset, stream=stream) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._walk_decode_pool(v, pool_ptr, slot_offset, stream=stream) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._walk_decode_pool(v, pool_ptr, slot_offset, stream=stream) for v in obj)
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
        if self._closed:
            return False, 0, None

        composite_key = f"{put_key}@{from_stage}_{to_stage}"

        try:
            _t0 = _time_mod.perf_counter()
            slot_offset = self._acquire_credit()
            _t_credit = _time_mod.perf_counter()
            if slot_offset is None:
                return self._put_cpu_fallback(from_stage, to_stage, put_key, composite_key, data)

            credit_returned = False
            try:
                self._board.buf[slot_offset // self._slot_size] = 0

                slot = _PoolSlot(self._pool, slot_offset, self._slot_size)
                self._compute_event.record()
                self._copy_stream.wait_event(self._compute_event)
                try:
                    with torch.cuda.stream(self._copy_stream):
                        encoded_obj = self._walk_encode_pool(data, slot)
                except _SlotOverflowError:
                    self._copy_done_event.record(self._copy_stream)
                    self._copy_done_event.synchronize()
                    credit_returned = True
                    self._credit_queue.put_nowait(slot_offset)
                    return self._put_cpu_fallback(from_stage, to_stage, put_key, composite_key, data)

                _t_copy = _time_mod.perf_counter()
                ret = self._cudart.cudaEventRecord(
                    self._ipc_event,
                    ctypes.c_void_p(self._copy_stream.cuda_stream),
                )
                if ret != 0:
                    logger.warning("cudaEventRecord (IPC) failed: %d", ret)
                _t_sync = _time_mod.perf_counter()

                wrapped = {
                    _POOL_MARKER: True,
                    "pool_handle": self._pool_handle,
                    "slot_offset": slot_offset,
                    "board": self._board_name,
                    "slot_index": slot_offset // self._slot_size,
                    "event_handle": self._ipc_event_handle_bytes,
                    "payload": encoded_obj,
                }
                payload = OmniSerializer.serialize(wrapped)
                size = len(payload)
                _t_set = _time_mod.perf_counter()
            except Exception:
                if not credit_returned:
                    self._credit_queue.put_nowait(slot_offset)
                raise

            with self._held_lock:
                self._held_credits[composite_key] = (_time_mod.time(), slot_offset)

            payload_name = self._payload_name(composite_key)
            try:
                meta = self._atomic_shm_write(payload, name=payload_name)
            except Exception:
                with self._held_lock:
                    self._held_credits.pop(composite_key, None)
                self._credit_queue.put_nowait(slot_offset)
                raise
            _t_shm = _time_mod.perf_counter()

            put_n = self._metrics["puts"]
            if put_n < 20 or put_n % 200 == 0:
                logger.info(
                    "IPC_PUT #%d key=%s credit=%.3fms copy=%.3fms sync=%.3fms "
                    "ser=%.3fms shm=%.3fms total=%.3fms size=%d",
                    put_n,
                    put_key,
                    (_t_credit - _t0) * 1000,
                    (_t_copy - _t_credit) * 1000,
                    (_t_sync - _t_copy) * 1000,
                    (_t_set - _t_sync) * 1000,
                    (_t_shm - _t_set) * 1000,
                    (_t_shm - _t0) * 1000,
                    size,
                )

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
        logger.warning(
            "CudaIPCConnector CPU fallback for %s: pool credits exhausted or slot overflow.",
            put_key,
        )
        self._metrics["cpu_fallbacks"] += 1
        payload = self.serialize_obj(data)
        size = len(payload)

        meta = self._atomic_shm_write(payload, name=put_key)

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
        if self._closed:
            return None

        composite_key = f"{get_key}@{from_stage}_{to_stage}"
        payload_name = self._payload_name(composite_key)
        if os.path.exists(f"/dev/shm/{payload_name}"):
            return self._try_get_ipc(get_key, payload_name)
        return self._try_get_shm_compat(get_key)

    def _flush_deferred_releases(self) -> None:
        """Release credits for completed D2D copies (non-blocking)."""
        still_pending: list[tuple] = []
        while self._deferred_credit_releases:
            event, board, slot_index = self._deferred_credit_releases.popleft()
            if event.query():
                self._mark_board_release(board, slot_index)
            else:
                still_pending.append((event, board, slot_index))
        if still_pending:
            self._deferred_credit_releases.extend(still_pending)

    def _try_get_ipc(
        self,
        get_key: str,
        payload_name: str,
    ) -> tuple[Any, int] | None:
        try:
            _t0 = _time_mod.perf_counter()
            seg = shm_pkg.SharedMemory(name=payload_name)
            try:
                data_bytes = bytes(seg.buf[: seg.size])
            finally:
                seg.close()
                seg.unlink()
            _t_shm = _time_mod.perf_counter()

            raw_obj = OmniSerializer.deserialize(data_bytes)
            _t_deser = _time_mod.perf_counter()
            if not isinstance(raw_obj, dict) or not raw_obj.get(_POOL_MARKER):
                logger.error("CudaIPCConnector get: unexpected payload format for %s (corrupt segment?)", get_key)
                return None

            pool_ptr = self._open_pool(raw_obj["pool_handle"])

            # --- Multi-stream copy pipeline (CUDA-graph safe) ---
            #
            # Round-robin across N copy streams so concurrent D2D transfers
            # overlap on the GPU. Credit release is deferred (event-based,
            # no CPU synchronize on the hot path).
            self._flush_deferred_releases()
            idx = self._recv_stream_idx % len(self._recv_copy_streams)
            self._recv_stream_idx += 1
            copy_stream = self._recv_copy_streams[idx]
            copy_done_event = self._recv_copy_events[idx]

            event_handle = raw_obj.get("event_handle")
            if event_handle:
                ipc_event = self._open_ipc_event(event_handle)
                # copy_stream waits for sender's copy to finish
                self._cudart.cudaStreamWaitEvent(
                    ctypes.c_void_p(copy_stream.cuda_stream),
                    ipc_event,
                    ctypes.c_uint(0),
                )
            # Order copy_stream after the local compute stream before we write
            # into freshly-allocated dst tensors. dst comes from the caching
            # allocator, which may hand back a block whose prior owner still has
            # pending compute-stream work; without this barrier the copy_stream
            # D2D can race that work and corrupt the tensor (intermittent
            # nan/inf downstream).
            copy_stream.wait_stream(torch.cuda.current_stream(self.local_device))
            _t_pool = _time_mod.perf_counter()

            # D2D copies on copy_stream (not compute stream)
            obj = self._walk_decode_pool(
                raw_obj["payload"],
                pool_ptr,
                raw_obj["slot_offset"],
                stream=copy_stream,
            )
            _t_decode = _time_mod.perf_counter()

            # Record event on copy_stream after all copies are issued
            copy_done_event.record(copy_stream)

            # Compute stream waits on copy_done_event (GPU-side dependency,
            # no CPU block -- CUDA-graph capturable)
            torch.cuda.current_stream(self.local_device).wait_event(copy_done_event)

            # Defer credit release: the D2D is in flight on copy_stream.
            # A background flush (called at the top of each get) polls the
            # event and releases the credit once the GPU is done — no CPU
            # synchronize on the hot path.
            self._deferred_credit_releases.append(
                (
                    copy_done_event,
                    raw_obj["board"],
                    int(raw_obj["slot_index"]),
                )
            )
            _t_sync = _time_mod.perf_counter()

            size = len(data_bytes)
            get_n = self._metrics["gets"]
            if get_n < 20 or get_n % 200 == 0:
                logger.info(
                    "IPC_GET #%d key=%s shm=%.3fms deser=%.3fms pool=%.3fms "
                    "decode=%.3fms sync=%.3fms total=%.3fms size=%d",
                    get_n,
                    get_key,
                    (_t_shm - _t0) * 1000,
                    (_t_deser - _t_shm) * 1000,
                    (_t_pool - _t_deser) * 1000,
                    (_t_decode - _t_pool) * 1000,
                    (_t_sync - _t_decode) * 1000,
                    (_t_sync - _t0) * 1000,
                    size,
                )
            self._metrics["gets"] += 1
            self._metrics["bytes_transferred"] += size
            return obj, size
        except FileNotFoundError:
            return None
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("CudaIPCConnector IPC get failed for %s: %s", get_key, e, exc_info=True)
            return None

    def _try_get_shm_compat(self, get_key: str) -> tuple[Any, int] | None:
        try:
            try:
                seg = shm_pkg.SharedMemory(name=get_key)
            except FileNotFoundError:
                return None

            try:
                mv = memoryview(seg.buf)
                data_bytes = bytes(mv[: seg.size])
                del mv
            finally:
                try:
                    seg.close()
                except Exception:
                    pass

            try:
                obj = self.deserialize_obj(data_bytes)
            except Exception as de:
                n = self._shm_compat_decode_failures.get(get_key, 0) + 1
                self._shm_compat_decode_failures[get_key] = n
                logger.warning(
                    "CudaIPCConnector shm_compat decode failed for %s: %s (attempt=%d, bytes=%d)",
                    get_key,
                    de,
                    n,
                    len(data_bytes),
                )
                if n >= 3:
                    try:
                        seg = shm_pkg.SharedMemory(name=get_key)
                        try:
                            seg.unlink()
                        finally:
                            seg.close()
                    except Exception:
                        pass
                return None

            self._shm_compat_decode_failures.pop(get_key, None)
            try:
                seg = shm_pkg.SharedMemory(name=get_key)
                try:
                    seg.unlink()
                finally:
                    seg.close()
            except Exception:
                pass

            obj = self._move_to_device(obj)

            size = len(data_bytes)
            self._metrics["gets"] += 1
            self._metrics["bytes_transferred"] += size
            return obj, size
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning("CudaIPCConnector shm_compat get failed for %s: %s", get_key, e)
            return None

    def _move_to_device(self, obj: Any) -> Any:
        """Move CPU tensors to local GPU so CUDA graph replay is safe."""
        if isinstance(obj, torch.Tensor) and obj.device.type == "cpu":
            return obj.to(self.local_device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._move_to_device(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._move_to_device(v) for v in obj)
        return obj

    # ------------------------------------------------------------------
    # Credit release: shared-memory board (fast path) + TTL sweep
    # ------------------------------------------------------------------

    def _mark_board_release(self, board_name: str, slot_index: int) -> None:
        """Receiver: flip the slot byte on the sender's release board."""
        board = self._opened_boards.get(board_name)
        if board is None:
            try:
                board = shm_pkg.SharedMemory(name=board_name)
            except FileNotFoundError:
                logger.warning("Release board %s not found; sender will rely on TTL.", board_name)
                return
            self._opened_boards[board_name] = board
        if not 0 <= slot_index < board.size:
            logger.warning("Release board %s: slot_index %d out of range.", board_name, slot_index)
            return
        board.buf[slot_index] = 1

    def _reclaim_board_credits(self) -> None:
        """Sender: reclaim credits whose board byte was set by the receiver."""
        if self._board is None:
            return
        buf = self._board.buf
        with self._held_lock:
            released = [
                (key, slot_offset)
                for key, (_ts, slot_offset) in self._held_credits.items()
                if buf[slot_offset // self._slot_size] == 1
            ]
            for key, slot_offset in released:
                self._held_credits.pop(key, None)
                buf[slot_offset // self._slot_size] = 0
                self._credit_queue.put_nowait(slot_offset)
                self._metrics["board_releases"] += 1

    def _acquire_credit(self) -> int | None:
        """Get a free slot offset, reclaiming board credits inline.

        Bounded wait (``credit_wait_sec``) before giving up; returns None to
        trigger the CPU fallback.
        """
        try:
            return self._credit_queue.get_nowait()
        except _queue_mod.Empty:
            pass
        deadline = _time_mod.monotonic() + self._credit_wait_sec
        while _time_mod.monotonic() < deadline:
            self._reclaim_board_credits()
            try:
                return self._credit_queue.get_nowait()
            except _queue_mod.Empty:
                _time_mod.sleep(self._credit_poll_sec)
        return None

    def _release_expired_credits(self) -> None:
        """TTL sweep: reclaim slots whose receiver never marked the board
        (e.g. the request was aborted or the receiver died)."""
        now = _time_mod.time()
        with self._held_lock:
            expired = [
                (key, slot_offset)
                for key, (ts, slot_offset) in self._held_credits.items()
                if now - ts > self.tensor_lifetime_sec
            ]
            for key, slot_offset in expired:
                self._held_credits.pop(key, None)
                self._board.buf[slot_offset // self._slot_size] = 0
                self._credit_queue.put_nowait(slot_offset)
                self._metrics["ttl_releases"] += 1

    def _release_loop(self) -> None:
        tick = 0
        while not self._stop_event.is_set():
            try:
                self._reclaim_board_credits()
                tick += 1
                if tick % self._release_ttl_every == 0:
                    self._release_expired_credits()
            except Exception as e:
                logger.debug("Release loop error: %s", e)
            self._stop_event.wait(timeout=self._release_interval_sec)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self, request_id: str) -> None:
        with self._held_lock:
            keys_to_remove = [k for k in self._held_credits if k.startswith(f"{request_id}_")]
            for k in keys_to_remove:
                entry = self._held_credits.pop(k, None)
                if entry:
                    if self._board is not None:
                        self._board.buf[entry[1] // self._slot_size] = 0
                    self._credit_queue.put_nowait(entry[1])

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy" if not self._closed else "closed",
            "role": self.role,
            "local_device": str(self.local_device),
            "pool_size_mb": self._pool_size // (1024 * 1024),
            "pool_credits": self._pool_credits,
            "held_credits": len(self._held_credits),
            **self._metrics,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        logger.info("Closing CudaIPCConnector...")

        self._stop_event.set()
        if self._release_thread is not None and self._release_thread.is_alive():
            self._release_thread.join(timeout=1.0)

        with self._held_lock:
            self._held_credits.clear()

        # Close cached pool mappings
        for pool_ptr in self._opened_pools.values():
            try:
                self._close_ipc_ptr(pool_ptr)
            except Exception as e:
                logger.warning("Failed to close pool mapping: %s", e)
        self._opened_pools.clear()

        # Release board(s)
        if self._board is not None:
            try:
                self._board.close()
                self._board.unlink()
            except Exception as e:
                logger.warning("Failed to release board: %s", e)
            self._board = None
        for board in self._opened_boards.values():
            try:
                board.close()
            except Exception:
                pass
        self._opened_boards.clear()

        self._pool = None

        if torch.cuda.is_available():
            try:
                torch.cuda.ipc_collect()
            except Exception as e:
                logger.warning(f"torch.cuda.ipc_collect() failed: {e}")

        logger.info("CudaIPCConnector closed.")

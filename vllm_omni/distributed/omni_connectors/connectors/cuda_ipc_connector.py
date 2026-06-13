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

Latency-isolation design (avoids stalling concurrent CUDA-graph replays):
  - All pool copies run as cudaMemcpyAsync on a private non-blocking stream.
    A cudaEvent recorded on the legacy default stream orders the pack copies
    after in-flight producer kernels WITHOUT inserting a barrier that would
    block subsequently launched compute kernels (which a synchronous
    cudaMemcpy on the legacy stream does).
  - The receiver synchronizes only its private copy stream per get() instead
    of a device-wide torch synchronize that waits for unrelated work.
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

import ctypes
import fcntl
import hashlib
import os
import queue as _queue_mod
import threading
import time as _time_mod
import uuid
from multiprocessing import shared_memory as shm_pkg
from typing import Any

import torch

from vllm_omni.entrypoints.stage_utils import shm_read_bytes, shm_write_bytes

from ..utils.logging import get_connector_logger
from ..utils.serialization import OmniSerializer
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

_GPU_TENSOR_MARKER = "__cuda_ipc_tensor__"
_POOL_MARKER = "__cuda_ipc_pool__"

_POOL_ALIGNMENT = 16  # bytes, for GPU copy efficiency

_DEFAULT_POOL_SIZE_MB = 128
_DEFAULT_POOL_CREDITS = 64

_CUDA_STREAM_NON_BLOCKING = 0x01  # cudaStreamNonBlocking
_CUDA_EVENT_DISABLE_TIMING = 0x02  # cudaEventDisableTiming
_CUDA_MEMCPY_D2D = 3  # cudaMemcpyDeviceToDevice

# How long put() waits (reclaiming board credits inline) before CPU fallback.
_CREDIT_WAIT_SEC = 0.008
_CREDIT_POLL_SEC = 0.0002
# Fast board-reclaim cadence; the TTL sweep runs every N fast ticks.
_RELEASE_FAST_INTERVAL_SEC = 0.005
_RELEASE_TTL_EVERY_N_TICKS = 20


class _CudaIpcMemHandle(ctypes.Structure):
    """ctypes wrapper for ``cudaIpcMemHandle_t`` (64-byte opaque struct)."""

    _fields_ = [("reserved", ctypes.c_char * 64)]


class _SlotOverflowError(Exception):
    """Raised when tensors exceed a pool slot's capacity."""


class _PoolSlot:
    """Tracks packing state for tensors within a single pool credit slot."""

    __slots__ = ("_pool_data_ptr", "_base", "_size", "_cursor", "_cudart", "_stream")

    def __init__(self, pool_data_ptr: int, slot_offset: int, slot_size: int, cudart, stream: ctypes.c_void_p):
        self._pool_data_ptr = pool_data_ptr
        self._base = slot_offset
        self._size = slot_size
        self._cursor = 0
        self._cudart = cudart
        self._stream = stream

    def pack(self, tensor: torch.Tensor) -> int:
        """Enqueue an async D2D copy into the pool slot, return byte offset within slot.

        Copies run on the connector's private stream; the caller must
        synchronize that stream before publishing the slot metadata.
        """
        nbytes = tensor.nbytes
        padding = (-self._cursor) % _POOL_ALIGNMENT
        aligned = self._cursor + padding
        if aligned + nbytes > self._size:
            raise _SlotOverflowError()
        dst = ctypes.c_void_p(self._pool_data_ptr + self._base + aligned)
        src = ctypes.c_void_p(tensor.data_ptr())
        ret = self._cudart.cudaMemcpyAsync(
            dst, src, ctypes.c_size_t(nbytes), ctypes.c_int(_CUDA_MEMCPY_D2D), self._stream
        )
        if ret != 0:
            raise RuntimeError(f"cudaMemcpyAsync (pool pack) failed with code {ret}")
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
        pool_size_mb = int(config.get("pool_size_mb", _DEFAULT_POOL_SIZE_MB))
        pool_credits = int(config.get("pool_credits", _DEFAULT_POOL_CREDITS))
        self._pool_size = pool_size_mb * 1024 * 1024
        self._slot_size = self._pool_size // pool_credits
        self._pool_credits = pool_credits

        # Private non-blocking stream for all pool copies: keeps the legacy
        # default stream free so concurrent CUDA-graph replays never stall
        # behind connector memcpys (and vice versa).
        with torch.cuda.device(self.local_device):
            self._copy_stream = self._create_stream()

        if self.role == "sender":
            with torch.cuda.device(self.local_device):
                self._pool = torch.zeros(self._pool_size, dtype=torch.uint8, device=self.local_device)
                # Orders pack copies after in-flight producer kernels.
                self._order_event = self._create_event()
            self._pool_handle = self._get_ipc_handle(self._pool.data_ptr())
            self._credit_queue: _queue_mod.Queue[int] = _queue_mod.Queue(maxsize=pool_credits)
            for i in range(pool_credits):
                self._credit_queue.put_nowait(i * self._slot_size)
            # Track which credit is held for which composite_key: key -> (timestamp, slot_offset)
            self._held_credits: dict[str, tuple[float, int]] = {}
            # Release board: 1 byte per slot, receiver writes 1 when done.
            self._board_name = f"cudaipc_board_{uuid.uuid4().hex[:16]}"
            self._board = shm_pkg.SharedMemory(create=True, size=pool_credits, name=self._board_name)
            self._board.buf[:pool_credits] = bytes(pool_credits)
        else:
            self._pool = None
            self._pool_handle = None
            self._credit_queue = None
            self._held_credits = {}
            self._order_event = None
            self._board_name = None
            self._board = None

        # Receiver: cache opened pool IPC mappings / sender release boards
        self._opened_pools: dict[bytes, ctypes.c_void_p] = {}
        self._opened_boards: dict[str, shm_pkg.SharedMemory] = {}

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
            f"pool={pool_size_mb}MB ({pool_credits} credits × {self._slot_size // 1024 // 1024}MB slots)"
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

    def _lock_file(self, name: str) -> str:
        return f"/dev/shm/{name}.lock"

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

        lib.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        lib.cudaStreamCreateWithFlags.restype = ctypes.c_int
        lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        lib.cudaStreamSynchronize.restype = ctypes.c_int
        lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        lib.cudaStreamDestroy.restype = ctypes.c_int

        lib.cudaEventCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        lib.cudaEventCreateWithFlags.restype = ctypes.c_int
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

    def _create_stream(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        ret = self._cudart.cudaStreamCreateWithFlags(ctypes.byref(stream), ctypes.c_uint(_CUDA_STREAM_NON_BLOCKING))
        if ret != 0:
            raise RuntimeError(f"cudaStreamCreateWithFlags failed with code {ret}")
        return stream

    def _create_event(self) -> ctypes.c_void_p:
        event = ctypes.c_void_p()
        ret = self._cudart.cudaEventCreateWithFlags(ctypes.byref(event), ctypes.c_uint(_CUDA_EVENT_DISABLE_TIMING))
        if ret != 0:
            raise RuntimeError(f"cudaEventCreateWithFlags failed with code {ret}")
        return event

    def _sync_copy_stream(self) -> None:
        ret = self._cudart.cudaStreamSynchronize(self._copy_stream)
        if ret != 0:
            raise RuntimeError(f"cudaStreamSynchronize failed with code {ret}")

    def _wait_producer_kernels(self) -> None:
        """Make the copy stream wait for in-flight kernels on the legacy stream.

        cudaEventRecord on the NULL (legacy) stream completes after all prior
        work submitted to blocking streams; cudaStreamWaitEvent then orders
        our async pack copies after that point without blocking any compute
        stream the way a synchronous legacy-stream cudaMemcpy would.
        """
        ret = self._cudart.cudaEventRecord(self._order_event, None)
        if ret != 0:
            raise RuntimeError(f"cudaEventRecord failed with code {ret}")
        ret = self._cudart.cudaStreamWaitEvent(self._copy_stream, self._order_event, ctypes.c_uint(0))
        if ret != 0:
            raise RuntimeError(f"cudaStreamWaitEvent failed with code {ret}")

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

    def _decode_pool_tensor(self, meta: dict[str, Any], pool_ptr: ctypes.c_void_p, slot_offset: int) -> torch.Tensor:
        """Decode a tensor from a cached pool mapping (async on copy stream)."""
        shape = tuple(meta["shape"])
        dtype = getattr(torch, meta["dtype"])
        nbytes = int(meta["nbytes"])
        tensor_offset = int(meta["pool_offset"])
        dst = torch.empty(shape, dtype=dtype, device=self.local_device)
        src = ctypes.c_void_p(pool_ptr.value + slot_offset + tensor_offset)
        ret = self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(dst.data_ptr()),
            src,
            ctypes.c_size_t(nbytes),
            ctypes.c_int(_CUDA_MEMCPY_D2D),
            self._copy_stream,
        )
        if ret != 0:
            raise RuntimeError(f"cudaMemcpyAsync (pool decode) failed with code {ret}")
        self._metrics["gpu_tensors_transferred"] += 1
        return dst

    def _walk_decode_pool(self, obj: Any, pool_ptr: ctypes.c_void_p, slot_offset: int) -> Any:
        """Recursively restore tensors from pool offset metadata."""
        if isinstance(obj, dict) and obj.get(_GPU_TENSOR_MARKER):
            return self._decode_pool_tensor(obj, pool_ptr, slot_offset)
        if isinstance(obj, dict):
            return {k: self._walk_decode_pool(v, pool_ptr, slot_offset) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._walk_decode_pool(v, pool_ptr, slot_offset) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._walk_decode_pool(v, pool_ptr, slot_offset) for v in obj)
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
            slot_offset = self._acquire_credit()
            if slot_offset is None:
                return self._put_cpu_fallback(from_stage, to_stage, put_key, composite_key, data)

            credit_returned = False
            try:
                # Clear any stale release mark before handing the slot out.
                self._board.buf[slot_offset // self._slot_size] = 0

                # Order async pack copies after in-flight producer kernels.
                self._wait_producer_kernels()

                slot = _PoolSlot(self._pool.data_ptr(), slot_offset, self._slot_size, self._cudart, self._copy_stream)
                try:
                    encoded_obj = self._walk_encode_pool(data, slot)
                except _SlotOverflowError:
                    self._sync_copy_stream()  # drain partial enqueued copies
                    credit_returned = True
                    self._credit_queue.put_nowait(slot_offset)
                    return self._put_cpu_fallback(from_stage, to_stage, put_key, composite_key, data)

                # Data must be resident in the pool before metadata is published.
                self._sync_copy_stream()

                wrapped = {
                    _POOL_MARKER: True,
                    "pool_handle": self._pool_handle,
                    "slot_offset": slot_offset,
                    "board": self._board_name,
                    "slot_index": slot_offset // self._slot_size,
                    "payload": encoded_obj,
                }
                payload = OmniSerializer.serialize(wrapped)
                size = len(payload)
            except Exception:
                # The slot is not yet tracked in _held_credits — without this
                # the credit would leak permanently on an unexpected failure.
                if not credit_returned:
                    self._credit_queue.put_nowait(slot_offset)
                raise

            # Track credit BEFORE writing SHM (same safety pattern as before)
            with self._held_lock:
                self._held_credits[composite_key] = (_time_mod.time(), slot_offset)

            payload_name = self._payload_name(composite_key)
            lock_file = self._lock_file(payload_name)
            try:
                with open(lock_file, "wb+") as lockf:
                    fcntl.flock(lockf, fcntl.LOCK_EX)
                    meta = shm_write_bytes(payload, name=payload_name)
                    fcntl.flock(lockf, fcntl.LOCK_UN)
            except Exception:
                with self._held_lock:
                    self._held_credits.pop(composite_key, None)
                self._credit_queue.put_nowait(slot_offset)
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
        logger.warning(
            "CudaIPCConnector CPU fallback for %s: pool credits exhausted or slot overflow.",
            put_key,
        )
        self._metrics["cpu_fallbacks"] += 1
        payload = self.serialize_obj(data)
        size = len(payload)

        lock_file = f"/dev/shm/shm_{put_key}_lockfile.lock"
        with open(lock_file, "wb+") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            meta = shm_write_bytes(payload, name=put_key)
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
        if self._closed:
            return None

        composite_key = f"{get_key}@{from_stage}_{to_stage}"
        payload_name = self._payload_name(composite_key)
        lock_file = self._lock_file(payload_name)
        ipc_exists = False
        try:
            seg = shm_pkg.SharedMemory(name=payload_name)
            seg.close()
            ipc_exists = True
        except FileNotFoundError:
            ipc_exists = False
        except ValueError:
            return None

        if ipc_exists:
            return self._try_get_ipc(get_key, payload_name, lock_file)
        return self._try_get_shm_compat(get_key)

    def _try_get_ipc(
        self,
        get_key: str,
        payload_name: str,
        lock_file: str,
    ) -> tuple[Any, int] | None:
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
            if not isinstance(raw_obj, dict) or not raw_obj.get(_POOL_MARKER):
                logger.error("CudaIPCConnector get: unexpected payload format for %s (corrupt segment?)", get_key)
                return None

            pool_ptr = self._open_pool(raw_obj["pool_handle"])
            obj = self._walk_decode_pool(raw_obj["payload"], pool_ptr, raw_obj["slot_offset"])
            # Wait only for our own copies — never for unrelated device
            # work (e.g. in-flight CUDA-graph replays).
            self._sync_copy_stream()

            # Release the sender's pool slot. Must happen only after the
            # copy-stream sync above — the sender may reuse the slot
            # immediately.
            self._mark_board_release(raw_obj["board"], int(raw_obj["slot_index"]))

            size = len(data_bytes)
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
        lock_file = f"/dev/shm/shm_{get_key}_lockfile.lock"
        try:
            with open(lock_file, "rb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                try:
                    seg = shm_pkg.SharedMemory(name=get_key)
                except FileNotFoundError:
                    fcntl.flock(lockf, fcntl.LOCK_UN)
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
                    fcntl.flock(lockf, fcntl.LOCK_UN)

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
                    try:
                        if os.path.exists(lock_file):
                            os.remove(lock_file)
                    except OSError:
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
            if os.path.exists(lock_file):
                os.remove(lock_file)

            size = len(data_bytes)
            self._metrics["gets"] += 1
            self._metrics["bytes_transferred"] += size
            return obj, size
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning("CudaIPCConnector shm_compat get failed for %s: %s", get_key, e)
            return None

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

        Bounded wait (~_CREDIT_WAIT_SEC) before giving up; returns None to
        trigger the CPU fallback.
        """
        try:
            return self._credit_queue.get_nowait()
        except _queue_mod.Empty:
            pass
        deadline = _time_mod.monotonic() + _CREDIT_WAIT_SEC
        while _time_mod.monotonic() < deadline:
            self._reclaim_board_credits()
            try:
                return self._credit_queue.get_nowait()
            except _queue_mod.Empty:
                _time_mod.sleep(_CREDIT_POLL_SEC)
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
                if tick % _RELEASE_TTL_EVERY_N_TICKS == 0:
                    self._release_expired_credits()
            except Exception as e:
                logger.debug("Release loop error: %s", e)
            self._stop_event.wait(timeout=_RELEASE_FAST_INTERVAL_SEC)

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

        # Destroy CUDA stream / event
        try:
            if self._order_event is not None:
                self._cudart.cudaEventDestroy(self._order_event)
            if self._copy_stream is not None:
                self._cudart.cudaStreamDestroy(self._copy_stream)
        except Exception as e:
            logger.warning("Failed to destroy CUDA stream/event: %s", e)
        self._order_event = None
        self._copy_stream = None

        self._pool = None

        if torch.cuda.is_available():
            try:
                torch.cuda.ipc_collect()
            except Exception as e:
                logger.warning(f"torch.cuda.ipc_collect() failed: {e}")

        logger.info("CudaIPCConnector closed.")

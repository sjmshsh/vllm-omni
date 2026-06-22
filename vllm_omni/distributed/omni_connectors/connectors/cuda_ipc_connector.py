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
  - Control plane: a per-edge SPSC keyed mailbox ring (``CudaIpcControlRing`` below) carries
    small payloads inline and big-payload pool descriptors, opened once per edge —
    replacing the per-transfer /dev/shm round-trip.

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
from .cuda_ipc_control_ring import CudaIpcControlRing, RingFullError
from .cuda_ipc_runtime import (
    _CUDA_EVENT_DISABLE_TIMING,
    _CUDA_EVENT_INTERPROCESS,
    _CUDA_MEMCPY_D2D,
    _CudaIpcEventHandle,
    _CudaIpcMemHandle,
    load_cudart,
)

logger = get_connector_logger(__name__)

_GPU_TENSOR_MARKER = "__cuda_ipc_tensor__"
_POOL_MARKER = "__cuda_ipc_pool__"

_POOL_ALIGNMENT = 16  # bytes, for GPU copy efficiency

# Pool defaults: auto-sized when user omits pool_size_mb / pool_credits.
# Auto formula: credits = max(64, max_num_seqs * 4), size = credits * 2 MB.
# Explicit extra config values override auto-sizing.
_DEFAULT_POOL_SIZE_MB = 128
_DEFAULT_POOL_CREDITS = 64

# Timing constants — overridable via extra config keys of the same name
# (without leading underscore), e.g. ``"credit_wait_sec": 0.01``.
_CREDIT_WAIT_SEC = 0.05  # put() inline reclaim window before CPU fallback
_CREDIT_POLL_SEC = 0.0005  # poll interval within the reclaim window
_RELEASE_FAST_INTERVAL_SEC = 0.001  # board-reclaim thread fast tick
_RELEASE_TTL_EVERY_N_TICKS = 20  # TTL sweep runs every N fast ticks

# Ring header carries the edge-constant pool/event/board handles once:
# pool_handle(64) + event_handle(64) + board_name_len(1) + board_name(<=63).
_RING_HEADER_BYTES = 256
_RING_PCLASS_INLINE = 0
_RING_PCLASS_POOL = 1


class _SlotOverflowError(Exception):
    """Raised when tensors exceed a pool slot's capacity."""

    def __init__(self, nbytes: int = 0, slot_size: int = 0):
        super().__init__(f"tensor {nbytes}B exceeds pool slot {slot_size}B")
        self.nbytes = nbytes
        self.slot_size = slot_size


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
            raise _SlotOverflowError(nbytes, self._size)
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
        # Ring control plane: put()/get() route through a pre-allocated per-edge SPSC
        # mailbox (see ``CudaIpcControlRing`` above); small payloads ride inline, big ones use the
        # pool + D2D. /dev/shm remains only as the overflow/ring-full/abort fallback.
        self._inline_threshold = int(config.get("inline_threshold_bytes", 16384))
        self._ring_entries_cfg = int(config.get("ring_entries", 0))  # 0 => auto from credits
        # Ring slot body must hold the LARGER of: an inline small payload, OR a pool
        # descriptor. The pool descriptor carries CPU-side payload (token-id lists) and
        # scales with sequence length (~40KB at input 4000), so default generously; a
        # descriptor that still overflows degrades to the CPU fallback, never crashes.
        self._ring_body_max = int(config.get("ring_body_max", 524288))
        self._ring: CudaIpcControlRing | None = None  # sender: created at init; receiver: None
        self._opened_rings: dict[tuple[str, str], CudaIpcControlRing] = {}  # receiver cache
        self._ring_edge_handles: dict[tuple[str, str], tuple[bytes, bytes, str]] = {}
        self.local_device = self._resolve_local_device(config.get("local_device", "auto"))
        self._closed = False
        self._cudart = None

        self._held_lock = threading.Lock()
        self._open_lock = threading.Lock()
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
        self._cudart = load_cudart()

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
            # CPU-fallback /dev/shm segments we wrote {name: ts}. The receiver unlinks each
            # on successful read; on ABORT (receiver never reads) the segment would orphan,
            # so the release loop TTL-sweeps these (the adapter does NOT call connector
            # cleanup(), so we cannot rely on it). Bounds the leak to the TTL window.
            self._fallback_segs: dict[str, float] = {}
            self._board_name = f"cudaipc_board_{uuid.uuid4().hex[:16]}"
            self._board = shm_pkg.SharedMemory(create=True, size=pool_credits, name=self._board_name)
            self._board.buf[:pool_credits] = bytes(pool_credits)
            n_slots = self._ring_entries_cfg or max(64, pool_credits * 4)
            body_max = max(self._ring_body_max, self._inline_threshold)
            self._ring = CudaIpcControlRing.create(
                self._ring_name(self.stage_id, self.stage_id + 1),
                n_slots,
                body_max,
                header_bytes=_RING_HEADER_BYTES,
            )
            self._ring.write_header(self._ring_header_blob())
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
    # Low-level CUDA IPC via ctypes (bindings in cuda_ipc_runtime.load_cudart)
    # ------------------------------------------------------------------

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
        """Open a pool IPC handle (cached — only opened once per sender).

        Lock-guarded: the optional pre-warm thread and the recv thread may both
        reach here, and cudaIpcOpenMemHandle on an already-open handle errors,
        so the check-and-open must be atomic.
        """
        with self._open_lock:
            if pool_handle not in self._opened_pools:
                self._opened_pools[pool_handle] = self._open_ipc_ptr(pool_handle)
            return self._opened_pools[pool_handle]

    def _open_ipc_event(self, handle_bytes: bytes) -> ctypes.c_void_p:
        """Open a CUDA IPC event handle (cached — only opened once per sender)."""
        with self._open_lock:
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
        target_stream = stream if stream is not None else torch.cuda.current_stream(self.local_device)
        # Allocate dst ON target_stream (not the compute stream) so dst's owning
        # stream IS the copy stream the D2D runs on — no cross-stream usage, so NO
        # record_stream is needed. The old code called dst.record_stream(copy_stream)
        # BEFORE the cudaMemcpyAsync; that registered a free-protection event ahead of
        # the copy, and at dst's destruction the allocator's insert_events iterated that
        # stream and hit the poisoned context (illegal access in ~TensorImpl). Since the
        # caller (_get_ring) now synchronizes copy_done_event before
        # returning, dst's lifetime is covered by that hard sync — correct by construction.
        with torch.cuda.stream(target_stream):
            dst = torch.empty(shape, dtype=dtype, device=self.local_device)
        src = ctypes.c_void_p(pool_ptr.value + slot_offset + tensor_offset)
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
        return self._put_ring(from_stage, to_stage, put_key, composite_key, data)

    def _put_cpu_fallback(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        composite_key: str,
        data: Any,
        reason: str = "",
    ) -> tuple[bool, int, dict[str, Any] | None]:
        logger.warning(
            "CudaIPCConnector CPU fallback for %s (from_stage=%s to_stage=%s): %s",
            put_key,
            from_stage,
            to_stage,
            reason or "pool credits exhausted or slot overflow",
        )
        self._metrics["cpu_fallbacks"] += 1
        payload = self.serialize_obj(data)
        size = len(payload)

        meta = self._atomic_shm_write(payload, name=put_key)
        # Track for TTL cleanup in case the receiver aborts and never reads/unlinks it.
        if getattr(self, "_fallback_segs", None) is not None:
            self._fallback_segs[put_key] = _time_mod.time()

        self._metrics["puts"] += 1
        self._metrics["bytes_transferred"] += size
        return True, size, {"shm": meta, "size": size, "cpu_fallback": True}

    # ------------------------------------------------------------------
    # Ring control plane — put/get over the per-edge SPSC mailbox.
    # The only transport path; replaces the per-transfer /dev/shm round-trip.
    # ------------------------------------------------------------------

    @staticmethod
    def _ring_name(from_stage, to_stage) -> str:
        return f"cudaipc_ring_s{from_stage}_{to_stage}"

    @staticmethod
    def _key_hash16(composite_key: str) -> bytes:
        return hashlib.sha1(composite_key.encode("utf-8")).digest()[:16]

    def _ring_header_blob(self) -> bytes:
        bn = self._board_name.encode("utf-8")
        if len(bn) > _RING_HEADER_BYTES - 129:
            raise ValueError("board_name too long for ring header")
        return self._pool_handle + self._ipc_event_handle_bytes + bytes([len(bn)]) + bn

    @staticmethod
    def _parse_ring_header(blob: bytes) -> tuple[bytes, bytes, str]:
        pool_handle = bytes(blob[0:64])
        event_handle = bytes(blob[64:128])
        bn_len = blob[128]
        board_name = bytes(blob[129 : 129 + bn_len]).decode("utf-8")
        return pool_handle, event_handle, board_name

    def _estimate_nbytes(self, obj: Any) -> int:
        """Sum GPU-tensor bytes (the part that would go to the pool) WITHOUT a
        serialize/D2H — used to route inline vs pool."""
        if isinstance(obj, torch.Tensor):
            return obj.nbytes if obj.is_cuda else 0
        if isinstance(obj, dict):
            return sum(self._estimate_nbytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(self._estimate_nbytes(v) for v in obj)
        if hasattr(obj, "__struct_fields__"):
            return sum(
                self._estimate_nbytes(getattr(obj, f)) for f in obj.__struct_fields__ if getattr(obj, f) is not None
            )
        return 0

    def _put_ring(self, from_stage, to_stage, put_key, composite_key, data):
        try:
            kh = self._key_hash16(composite_key)
            if self._estimate_nbytes(data) < self._inline_threshold:
                # INLINE: serialize (the cheap D2H for a tiny GPU tensor) -> ring body.
                payload = self.serialize_obj(data)
                if len(payload) > self._ring._body_max:
                    return self._put_cpu_fallback(
                        from_stage,
                        to_stage,
                        put_key,
                        composite_key,
                        data,
                        reason=f"inline_too_big {len(payload)}",
                    )
                try:
                    self._ring.publish(
                        kh,
                        _RING_PCLASS_INLINE,
                        payload,
                        ts=int(_time_mod.time()),
                        ttl_sec=int(self.tensor_lifetime_sec),
                    )
                except RingFullError:
                    return self._put_cpu_fallback(
                        from_stage, to_stage, put_key, composite_key, data, reason="ring_full"
                    )
                self._metrics["puts"] += 1
                self._metrics["bytes_transferred"] += len(payload)
                return True, len(payload), {"ring": True, "size": len(payload)}

            # POOL: existing credit + D2D-into-pool machinery (unchanged), then publish
            # a small descriptor (slot_offset/slot_index + tensor layout) to the ring.
            slot_offset = self._acquire_credit()
            if slot_offset is None:
                return self._put_cpu_fallback(
                    from_stage, to_stage, put_key, composite_key, data, reason="credits_exhausted"
                )
            credit_returned = False
            try:
                self._board.buf[slot_offset // self._slot_size] = 0
                slot = _PoolSlot(self._pool, slot_offset, self._slot_size)
                self._compute_event.record()
                self._copy_stream.wait_event(self._compute_event)
                try:
                    with torch.cuda.stream(self._copy_stream):
                        encoded_obj = self._walk_encode_pool(data, slot)
                except _SlotOverflowError as e:
                    self._copy_done_event.record(self._copy_stream)
                    self._copy_done_event.synchronize()
                    credit_returned = True
                    self._credit_queue.put_nowait(slot_offset)
                    return self._put_cpu_fallback(
                        from_stage,
                        to_stage,
                        put_key,
                        composite_key,
                        data,
                        reason=f"slot_overflow nbytes={e.nbytes} slot={e.slot_size}",
                    )
                ret = self._cudart.cudaEventRecord(self._ipc_event, ctypes.c_void_p(self._copy_stream.cuda_stream))
                if ret != 0:
                    logger.warning("cudaEventRecord (IPC) failed: %d", ret)
                # BLOCK until the pack D2D (source -> pool) actually completes before
                # returning from put(). The source is a runner-owned tensor (thinker
                # hidden states); once put() returns, the next forward may overwrite/free
                # it. Without this sync the pack copy on _copy_stream could still be
                # reading the source when it is freed -> read-after-free corruption that
                # surfaces later as an illegal access on the receiver. Cost: the save
                # thread blocks ~ms per big chunk (once/request). Makes the whole pool
                # D2D path synchronous on BOTH ends -> the async-race class is gone.
                self._copy_stream.synchronize()
                descriptor = OmniSerializer.serialize(
                    {
                        _POOL_MARKER: True,
                        "slot_offset": slot_offset,
                        "slot_index": slot_offset // self._slot_size,
                        "payload": encoded_obj,
                    }
                )
            except Exception:
                if not credit_returned:
                    self._credit_queue.put_nowait(slot_offset)
                raise
            # The descriptor carries CPU-side payload (token-id lists etc.) and can grow
            # past the ring body_max with sequence length. If it won't fit, degrade to the
            # CPU fallback (read by the receiver via _try_get_shm_compat) — never crash.
            if len(descriptor) > self._ring._body_max:
                self._credit_queue.put_nowait(slot_offset)
                return self._put_cpu_fallback(
                    from_stage,
                    to_stage,
                    put_key,
                    composite_key,
                    data,
                    reason=f"descriptor_too_big {len(descriptor)}>{self._ring._body_max}",
                )
            with self._held_lock:
                self._held_credits[composite_key] = (_time_mod.time(), slot_offset)
            try:
                self._ring.publish(
                    kh,
                    _RING_PCLASS_POOL,
                    descriptor,
                    ts=int(_time_mod.time()),
                    ttl_sec=int(self.tensor_lifetime_sec),
                )
            except RingFullError:
                with self._held_lock:
                    self._held_credits.pop(composite_key, None)
                self._credit_queue.put_nowait(slot_offset)
                return self._put_cpu_fallback(from_stage, to_stage, put_key, composite_key, data, reason="ring_full")
            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += len(descriptor)
            return True, len(descriptor), {"ring": True, "size": len(descriptor)}
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("CudaIPCConnector ring put failed for %s: %s", put_key, e, exc_info=True)
            return False, 0, None

    def _open_ring_receiver(self, from_stage, to_stage):
        edge = (from_stage, to_stage)
        ring = self._opened_rings.get(edge)
        if ring is None:
            try:
                ring = CudaIpcControlRing.open(self._ring_name(from_stage, to_stage))
            except FileNotFoundError:
                return None  # sender not up yet; poll loop tolerates None
            self._opened_rings[edge] = ring
            self._ring_edge_handles[edge] = self._parse_ring_header(ring.read_header(_RING_HEADER_BYTES))
        return ring

    def _get_ring(self, from_stage, to_stage, get_key, composite_key):
        try:
            ring = self._open_ring_receiver(from_stage, to_stage)
            if ring is None:
                return None
            r = ring.poll(self._key_hash16(composite_key))
            if r is None:
                # Ring miss: either the chunk isn't published yet (normal poll retry),
                # OR the sender took the CPU fallback. _put_cpu_fallback writes the legacy
                # key-addressed /dev/shm segment (name=put_key), NOT the ring, on
                # slot_overflow / credits_exhausted / ring_full. Without this read the
                # fallback data is never consumed -> the talker waits forever -> hang.
                return self._try_get_shm_compat(get_key)
            pclass, body = r
            if pclass == _RING_PCLASS_INLINE:
                # Return CPU tensors and let the DOWNSTREAM model do the H2D — exactly
                # like SharedMemoryConnector (shm_connector.py:122 "H2D ... happens
                # DOWNSTREAM"). The talker forward already does .to(device) on these
                # (qwen3_omni.py:216-217,677-678), so doing the H2D here was redundant
                # AND made it a device-wide-sync that blocked on the talker forward
                # (1-7ms for a 4KB chunk). Deferring it = parity with SHM (poll+deser).
                obj = self.deserialize_obj(body)
                self._metrics["gets"] += 1
                self._metrics["bytes_transferred"] += len(body)
                return obj, len(body)

            # POOL: read the edge-constant handles from the
            # ring header and the slot descriptor from the entry body.
            pool_handle, event_handle, board_name = self._ring_edge_handles[(from_stage, to_stage)]
            raw = OmniSerializer.deserialize(body)
            pool_ptr = self._open_pool(pool_handle)
            idx = self._recv_stream_idx % len(self._recv_copy_streams)
            self._recv_stream_idx += 1
            copy_stream = self._recv_copy_streams[idx]
            copy_done_event = self._recv_copy_events[idx]
            if event_handle:
                ipc_event = self._open_ipc_event(event_handle)
                self._cudart.cudaStreamWaitEvent(ctypes.c_void_p(copy_stream.cuda_stream), ipc_event, ctypes.c_uint(0))
            copy_stream.wait_stream(torch.cuda.current_stream(self.local_device))
            obj = self._walk_decode_pool(raw["payload"], pool_ptr, raw["slot_offset"], stream=copy_stream)
            copy_done_event.record(copy_stream)
            # BLOCK until the D2D actually completes before returning the payload. get()
            # runs on the background recv thread; the payload is consumed LATER on the model
            # thread (gpu_model_runner _sync_local_stage_payloads -> t.to("cpu")). A deferred,
            # event-only barrier did NOT make that consumption safe and raced 3 ways:
            #   (A) board/TTL ABA: round-robin events starve credits -> 30s TTL force-reclaims
            #       a slot whose D2D is still in flight; a stale deferred board write reuses it.
            #   (B) dst lifetime: dst (alloc'd on the compute stream, written on copy_stream)
            #       could be reclaimed by the caching allocator before the D2D finished.
            # Synchronizing here makes the D2D done before hand-off, then we mark the board
            # SYNCHRONOUSLY (no deferred queue, no stale write, credits return promptly so TTL
            # never fires mid-transfer) — fixing A and B at once. Cost: the recv thread blocks
            # ~ms per big chunk (once/request, already on the TTFP critical path).
            copy_done_event.synchronize()
            self._mark_board_release(board_name, int(raw["slot_index"]))
            self._metrics["gets"] += 1
            self._metrics["bytes_transferred"] += len(body)
            return obj, len(body)
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("CudaIPCConnector ring get failed for %s: %s", get_key, e, exc_info=True)
            return None

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
        return self._get_ring(from_stage, to_stage, get_key, composite_key)

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

    def _move_to_device(self, obj: Any, non_blocking: bool = True) -> Any:
        """Move CPU tensors to local GPU so CUDA graph replay is safe."""
        if isinstance(obj, torch.Tensor) and obj.device.type == "cpu":
            return obj.to(self.local_device, non_blocking=non_blocking)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v, non_blocking) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._move_to_device(v, non_blocking) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._move_to_device(v, non_blocking) for v in obj)
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
        # also TTL-sweep orphaned CPU-fallback /dev/shm segments (receiver aborted before
        # reading; normally the receiver unlinks on read so these are already gone).
        segs = getattr(self, "_fallback_segs", None)
        if segs:
            stale = [name for name, ts in segs.items() if now - ts > self.tensor_lifetime_sec]
            for name in stale:
                segs.pop(name, None)
                try:
                    seg = shm_pkg.SharedMemory(name=name)
                    seg.close()
                    seg.unlink()
                    self._metrics["fallback_seg_reclaims"] = self._metrics.get("fallback_seg_reclaims", 0) + 1
                except FileNotFoundError:
                    pass  # receiver already consumed + unlinked it (the common case)
                except Exception as e:
                    logger.debug("fallback seg unlink %s: %s", name, e)

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

        # Unlink any CPU-fallback /dev/shm segments still tracked (unconsumed at shutdown).
        for name in list(getattr(self, "_fallback_segs", {}) or {}):
            try:
                seg = shm_pkg.SharedMemory(name=name)
                seg.close()
                seg.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
        if getattr(self, "_fallback_segs", None) is not None:
            self._fallback_segs.clear()

        for board in self._opened_boards.values():
            try:
                board.close()
            except Exception:
                pass
        self._opened_boards.clear()

        # Ring control plane: sender unlinks its ring; receiver closes opens.
        if self._ring is not None:
            self._ring.close()  # owner -> unlinks
            self._ring = None
        for ring in self._opened_rings.values():
            try:
                ring.close()
            except Exception:
                pass
        self._opened_rings.clear()

        self._pool = None

        if torch.cuda.is_available():
            try:
                torch.cuda.ipc_collect()
            except Exception as e:
                logger.warning(f"torch.cuda.ipc_collect() failed: {e}")

        logger.info("CudaIPCConnector closed.")

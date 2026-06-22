# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-edge SPSC keyed mailbox ring — the CudaIPC connector's control plane.

Replaces the per-transfer /dev/shm round-trip (shm_open+ftruncate+mmap+rename+
unlink, ~5-6 syscalls/chunk) with ONE pre-allocated ring per directed edge,
opened once. The sender publishes a fixed-stride entry; the receiver looks it
up by key — both fenceless, correct on x86 TSO (the NVIDIA target; ARM/POWER
would need an explicit store fence).

SPSC: exactly one producer (TP rank0 sender) and one consumer (leader
receiver) per directed edge. Lock-free correctness rests on:
  - body-first / seq-LAST publish (seq is the release marker; a reader that
    sees seq>0 has seen the whole body on TSO),
  - a seq==0 IN-PROGRESS sentinel written FIRST on (re)claim, so a reader
    never matches a half-written slot,
  - a seqlock re-read (seq before == seq after, both !=0) guarding the body,
  - a per-slot consumed byte so the producer never reuses a slot the consumer
    has not taken (bounded backpressure => no reuse-after-wrap torn read),
  - open addressing by key hash (linear probe) so arbitrary, non-sequential
    composite keys map to slots and out-of-order keyed lookup is O(1) amort.

Pure-Python (struct + POSIX shared memory), no CUDA — testable on CPU.

Per-slot layout (little-endian):
  seq      u64  @0   publish marker, written LAST; 0 = empty / in-progress
  consumed u8   @8   set by consumer; producer reuses iff consumed==1 (or seq==0)
  pclass   u8   @9   0=inline payload, 1=pool descriptor
  ts       u32  @10  publish time (epoch s) for TTL reclaim of aborted entries
  keyhash  16B  @14  sha1(composite_key)[:16]
  blen     u32  @30  body length
  body     ...  @34
"""

import struct
from multiprocessing import shared_memory as shm_pkg

_OFF_SEQ, _OFF_CONSUMED, _OFF_PCLASS, _OFF_TS, _OFF_KEY, _OFF_LEN, _OFF_BODY = 0, 8, 9, 10, 14, 30, 34
_KEY_BYTES = 16


class RingFullError(Exception):
    """Raised by publish() when no free slot exists in the probe window (backpressure)."""


class CudaIpcControlRing:
    """One directed-edge SPSC keyed mailbox. Sender side calls create()+publish();
    receiver side calls open()+poll(). A fixed header region carries edge-constant
    bytes (e.g. the pool/event/board IPC handles) published once at create()."""

    __slots__ = ("_shm", "_buf", "_n", "_slot", "_body_max", "_hdr", "_base", "_pubctr", "_owner")

    def __init__(self, shm, n_slots, body_max, header_bytes, owner):
        self._shm = shm
        self._buf = shm.buf
        self._n = n_slots
        self._body_max = body_max
        self._slot = _OFF_BODY + body_max
        self._hdr = header_bytes
        self._base = 8 + header_bytes  # u32 n_slots + u32 body_max, then header, then slots
        self._pubctr = 0
        self._owner = owner  # sender created it (responsible for unlink)

    # ---- construction -------------------------------------------------
    @classmethod
    def create(cls, name, n_slots, body_max, header_bytes=0):
        size = 8 + header_bytes + n_slots * (_OFF_BODY + body_max)
        try:
            shm_pkg.SharedMemory(name=name).unlink()
        except FileNotFoundError:
            pass
        shm = shm_pkg.SharedMemory(create=True, size=size, name=name)
        shm.buf[:size] = bytes(size)  # zero => all slots empty (seq==0)
        struct.pack_into("<II", shm.buf, 0, n_slots, body_max)
        return cls(shm, n_slots, body_max, header_bytes, owner=True)

    @classmethod
    def open(cls, name):
        shm = shm_pkg.SharedMemory(name=name)
        n_slots, body_max = struct.unpack_from("<II", shm.buf, 0)
        # header size is implied by the on-wire layout the sender chose; the caller
        # passes the same header_bytes contract. We recover it from total size.
        total = shm.size
        header_bytes = total - 8 - n_slots * (_OFF_BODY + body_max)
        return cls(shm, n_slots, body_max, header_bytes, owner=False)

    # ---- header (edge-constant, written once by the sender) -----------
    def write_header(self, data: bytes) -> None:
        if len(data) > self._hdr:
            raise ValueError(f"header {len(data)}B exceeds reserved {self._hdr}B")
        self._buf[8 : 8 + len(data)] = data

    def read_header(self, n: int) -> bytes:
        return bytes(self._buf[8 : 8 + n])

    # ---- sender -------------------------------------------------------
    def publish(self, key_hash: bytes, pclass: int, body: bytes, ts: int = 0, ttl_sec: int = 0) -> None:
        """Claim a free slot (open-addressed by key_hash) and publish. Raises RingFullError
        when the probe window has no free slot (caller falls back, never blocks).

        C4 (abort reclaim): when ttl_sec>0, an occupied-but-unconsumed slot whose entry
        is older than ttl_sec (an aborted / never-polled request) is treated as free and
        reclaimed in place. This is race-free: it runs only here, on the single producer
        thread; the SPSC consumer would never poll an aborted key, so reusing it is safe."""
        if len(body) > self._body_max:
            raise ValueError(f"body {len(body)}B exceeds slot body_max {self._body_max}B")
        buf = self._buf
        home = struct.unpack_from("<Q", key_hash, 0)[0] % self._n
        for p in range(self._n):
            idx = (home + p) % self._n
            off = self._base + idx * self._slot
            seq = struct.unpack_from("<Q", buf, off + _OFF_SEQ)[0]
            free = seq == 0 or buf[off + _OFF_CONSUMED] == 1
            if not free and ttl_sec and ts:  # TTL-reclaim a stale unconsumed entry
                slot_ts = struct.unpack_from("<I", buf, off + _OFF_TS)[0]
                if slot_ts and (ts - slot_ts) > ttl_sec:
                    free = True
            if free:
                struct.pack_into("<Q", buf, off + _OFF_SEQ, 0)  # in-progress sentinel FIRST
                buf[off + _OFF_CONSUMED] = 0
                buf[off + _OFF_PCLASS] = pclass
                struct.pack_into("<I", buf, off + _OFF_TS, ts & 0xFFFFFFFF)
                buf[off + _OFF_KEY : off + _OFF_KEY + _KEY_BYTES] = key_hash[:_KEY_BYTES]
                struct.pack_into("<I", buf, off + _OFF_LEN, len(body))
                buf[off + _OFF_BODY : off + _OFF_BODY + len(body)] = body
                self._pubctr += 1
                struct.pack_into("<Q", buf, off + _OFF_SEQ, self._pubctr)  # publish LAST
                return
        raise RingFullError()

    # ---- receiver -----------------------------------------------------
    def poll(self, key_hash: bytes):
        """Return (pclass, body) for key_hash and mark it consumed, or None if absent.
        Open-addressed probe: stop at an empty (seq==0) slot — under the consumed-gate a
        present key is always found before the first empty slot in its probe path."""
        buf = self._buf
        home = struct.unpack_from("<Q", key_hash, 0)[0] % self._n
        target = key_hash[:_KEY_BYTES]
        for p in range(self._n):
            idx = (home + p) % self._n
            off = self._base + idx * self._slot
            seq_a = struct.unpack_from("<Q", buf, off + _OFF_SEQ)[0]
            if seq_a == 0:
                return None  # empty slot in the probe path => key not present
            if buf[off + _OFF_CONSUMED] == 0 and buf[off + _OFF_KEY : off + _OFF_KEY + _KEY_BYTES] == target:
                pclass = buf[off + _OFF_PCLASS]
                ln = struct.unpack_from("<I", buf, off + _OFF_LEN)[0]
                body = bytes(buf[off + _OFF_BODY : off + _OFF_BODY + ln])
                seq_b = struct.unpack_from("<Q", buf, off + _OFF_SEQ)[0]
                if seq_b != seq_a:
                    return None  # torn (slot reused mid-read); caller retries the poll
                buf[off + _OFF_CONSUMED] = 1  # reclaim (producer may now reuse)
                return pclass, body
        return None

    # ---- lifecycle ----------------------------------------------------
    def close(self) -> None:
        try:
            self._buf = None
            self._shm.close()
            if self._owner:
                self._shm.unlink()
        except Exception:
            pass

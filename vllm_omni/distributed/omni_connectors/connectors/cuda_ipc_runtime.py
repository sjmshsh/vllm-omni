# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Low-level CUDA IPC runtime bindings for the CudaIPC connector.

Thin ctypes wrapper over the ``libcudart`` IPC symbols (mem-handle get/open/close,
event create/get/open/record, stream-wait, memcpy). Kept separate from the
connector so the transport logic doesn't interleave with raw ``ctypes`` plumbing.
"""

import ctypes


class _CudaIpcMemHandle(ctypes.Structure):
    """ctypes wrapper for ``cudaIpcMemHandle_t`` (64-byte opaque struct)."""

    _fields_ = [("reserved", ctypes.c_char * 64)]


class _CudaIpcEventHandle(ctypes.Structure):
    """ctypes wrapper for ``cudaIpcEventHandle_t`` (64-byte opaque struct)."""

    _fields_ = [("reserved", ctypes.c_char * 64)]


# CUDA runtime API constants (fixed by CUDA spec, not configurable).
_CUDA_MEMCPY_D2D = 3  # cudaMemcpyDeviceToDevice
_CUDA_EVENT_INTERPROCESS = 0x04
_CUDA_EVENT_DISABLE_TIMING = 0x02


def load_cudart():
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


# Thin call wrappers — keep the ctypes.c_void_p/c_size_t boxing and the ret!=0 raise out of
# the connector's hot path.
def memcpy_async_d2d(lib, dst: int, src: int, nbytes: int, stream: int) -> None:
    ret = lib.cudaMemcpyAsync(
        ctypes.c_void_p(dst),
        ctypes.c_void_p(src),
        ctypes.c_size_t(nbytes),
        ctypes.c_int(_CUDA_MEMCPY_D2D),
        ctypes.c_void_p(stream),
    )
    if ret != 0:
        raise RuntimeError(f"cudaMemcpyAsync (D2D) failed: {ret}")


def stream_wait_event(lib, stream: int, event) -> None:
    ret = lib.cudaStreamWaitEvent(ctypes.c_void_p(stream), event, ctypes.c_uint(0))
    if ret != 0:
        raise RuntimeError(f"cudaStreamWaitEvent failed: {ret}")

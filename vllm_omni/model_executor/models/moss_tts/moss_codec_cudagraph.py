"""CUDA Graph acceleration for the MOSS Audio Tokenizer codec decoder.

Captures MossAudioTokenizerModel._decode for a set of fixed frame-count
bucket sizes, then replays the captured graph at inference time to eliminate
kernel-launch overhead.  Inputs that exceed all captured sizes fall back to
eager execution transparently.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import NamedTuple

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer import (
    MossAudioTokenizerDecoderOutput,
    MossAudioTokenizerModel,
)

logger = init_logger(__name__)


def _decode_codec(model: object, codes: torch.Tensor, lengths: torch.Tensor) -> MossAudioTokenizerDecoderOutput:
    decode = getattr(model, "_decode", None)
    if callable(decode):
        return decode(codes, lengths)
    decode_frame = getattr(model, "_decode_frame", None)
    if callable(decode_frame):
        return decode_frame(codes, lengths)
    raise AttributeError("MOSS codec model must expose _decode or _decode_frame")


class MossTTSCUDAGraphCodecWrapper:
    """CUDA Graph wrapper for MossAudioTokenizerModel._decode.

    Graphs are keyed by padded_T (int).  On each call the actual T is
    bucket-matched to the smallest pre-captured size >= T.  The static code
    buffer [NQ, 1, padded_T] is filled left-aligned (right-zero-padded) and
    the graph is replayed.  The output audio is sliced to the correct length
    by scaling from the captured audio shape (actual_T / padded_T * captured_len),
    avoiding any assumption about downsample_rate vs effective decoder upsample.
    The slice is cloned before returning so the static buffer can be reused.

    Usage::

        wrapper = MossTTSCUDAGraphCodecWrapper(codec_model, capture_sizes, nq)
        wrapper.warmup(device)

        # per-request decode:
        out = wrapper.decode(codes_nq_t)   # codes_nq_t: [NQ, T]
    """

    def __init__(
        self,
        model: MossAudioTokenizerModel,
        capture_sizes: list[int],
        num_quantizers: int,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.capture_sizes: list[int] = sorted(capture_sizes)
        self.num_quantizers = num_quantizers
        self.enabled = enabled

        # All dicts keyed by padded_T.
        self.graphs: dict[int, CUDAGraph] = {}
        self.static_codes: dict[int, torch.Tensor] = {}  # [NQ, 1, padded_T]
        # static_lengths is kept alive here — the captured graph holds a
        # reference to the underlying storage and must not be GC'd.
        self.static_lengths: dict[int, torch.Tensor] = {}  # [1]
        self.static_audio: dict[int, torch.Tensor] = {}  # [1, 1, padded_T * effective_upsample]

        self._warmed_up = False

    # ------------------------------------------------------------------
    # Size helpers
    # ------------------------------------------------------------------

    def _get_padded_size(self, actual_t: int) -> int | None:
        """Return the smallest capture size >= actual_t, or None if too large."""
        for s in self.capture_sizes:
            if actual_t <= s:
                return s
        return None

    # ------------------------------------------------------------------
    # Warmup / capture
    # ------------------------------------------------------------------

    def warmup(self, device: torch.device) -> None:
        """Allocate static buffers and capture CUDA Graphs for all sizes."""
        if device.type != "cuda" or not self.enabled or self._warmed_up:
            return

        nq = self.num_quantizers
        logger.info(
            "MOSS-TTS codec CUDA Graph warmup: nq=%d capture_sizes=%s",
            nq,
            self.capture_sizes,
        )
        t0 = time.perf_counter()

        # One eager run per size to let cuDNN / CUDA allocate memory before
        # the capture window (graph capture forbids new CUDA allocs during it).
        for size in self.capture_sizes:
            dummy_codes = torch.zeros(nq, 1, size, dtype=torch.long, device=device)
            dummy_lengths = torch.tensor([size], dtype=torch.long, device=device)
            with torch.no_grad():
                _ = _decode_codec(self.model, dummy_codes, dummy_lengths)

        torch.accelerator.synchronize(device)

        for size in self.capture_sizes:
            try:
                self._capture(size, device)
                logger.info("  Captured CUDA Graph for size=%d", size)
            except Exception:
                logger.warning(
                    "  Failed to capture CUDA Graph for size=%d; this size will fall back to eager decode",
                    size,
                    exc_info=True,
                )

        self._warmed_up = True
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "MOSS-TTS codec CUDA Graph warmup complete: %d/%d captured in %.1f ms",
            len(self.graphs),
            len(self.capture_sizes),
            elapsed_ms,
        )

    def _capture(self, size: int, device: torch.device) -> None:
        nq = self.num_quantizers
        static_codes = torch.zeros(nq, 1, size, dtype=torch.long, device=device)
        # lengths holds the number of valid code frames; set to full size at
        # capture time so the decoder emits a full-size audio buffer.
        static_lengths = torch.tensor([size], dtype=torch.long, device=device)

        # Extra eager warmup inside capture to ensure all kernels are compiled.
        with torch.no_grad():
            _ = _decode_codec(self.model, static_codes, static_lengths)
        torch.accelerator.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_out = _decode_codec(self.model, static_codes, static_lengths)

        self.graphs[size] = graph
        self.static_codes[size] = static_codes
        self.static_lengths[size] = static_lengths
        # static_out.audio is a static buffer reused every replay; hold a
        # reference so it is not garbage-collected.
        self.static_audio[size] = static_out.audio  # [1, 1, size * effective_upsample]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode(self, codes_nq_t: torch.Tensor) -> MossAudioTokenizerDecoderOutput:
        """Decode [NQ, T] codes to waveform using a CUDA Graph when possible.

        Falls back to eager batch_decode when:
          - CUDA Graph is disabled or not yet warmed up
          - an outer CUDA stream capture is active (e.g. vLLM FULL graph mode)
          - actual T exceeds all pre-captured sizes
        """
        if not self.enabled or not self._warmed_up:
            return self.model.batch_decode(codes_list=[codes_nq_t], num_quantizers=self.num_quantizers)

        # Replaying a graph inside an active stream capture would corrupt the
        # outer graph.  Fall back to eager so the caller can complete its capture.
        if torch.cuda.is_current_stream_capturing():
            return self.model.batch_decode(codes_list=[codes_nq_t], num_quantizers=self.num_quantizers)

        actual_t = int(codes_nq_t.shape[-1])
        padded_size = self._get_padded_size(actual_t)

        if padded_size is None or padded_size not in self.graphs:
            return self.model.batch_decode(codes_list=[codes_nq_t], num_quantizers=self.num_quantizers)

        # --- Fill static buffers then replay ---

        static_codes = self.static_codes[padded_size]  # [NQ, 1, padded_size]

        if actual_t == padded_size:
            # Exact fit: copy the whole buffer at once.
            # codes_nq_t is [NQ, T]; unsqueeze(1) → [NQ, 1, T] matching static_codes.
            static_codes.copy_(codes_nq_t.unsqueeze(1))
        else:
            # Smaller input: zero the buffer first (right-zero-pad), then fill
            # left-aligned.  static_codes[:, 0, :actual_t] is [NQ, actual_t]
            # and codes_nq_t is [NQ, actual_t].
            static_codes.zero_()
            static_codes[:, 0, :actual_t].copy_(codes_nq_t)

        self.graphs[padded_size].replay()

        # static_audio[padded_size] is the live output buffer of the graph.
        # Slice to the real audio length and clone before returning; without
        # the clone the next replay would overwrite the caller's tensor.
        # Derive the slice length from the captured audio shape rather than
        # downsample_rate: the decoder's effective upsample (product of all
        # PatchedPretransform patch sizes) can differ from the config attribute.
        captured_len = self.static_audio[padded_size].shape[-1]
        actual_wav_len = captured_len * actual_t // padded_size
        audio = self.static_audio[padded_size][..., :actual_wav_len].clone()
        audio_lengths = torch.tensor([actual_wav_len], dtype=torch.long, device=audio.device)
        return MossAudioTokenizerDecoderOutput(audio=audio, audio_lengths=audio_lengths)


class _CapturedStreamingCodecGraph(NamedTuple):
    graph: CUDAGraph
    static_codes: torch.Tensor
    static_lengths: torch.Tensor
    static_audio: torch.Tensor
    static_audio_lengths: torch.Tensor


class MossTTSStreamingCUDAGraphCodecWrapper:
    """CUDA Graph replay for stateful MOSS codec streaming steps.

    This wrapper is intentionally exact-T and full-slot-width: each graph is
    captured for ``[NQ, stream_slots, T]`` while the codec is already inside a
    persistent ``codec.streaming(stream_slots)`` session. The caller provides
    the live ``exec_mask`` before replay so inactive slots keep their streaming
    state unchanged.
    """

    def __init__(
        self,
        model: object,
        capture_sizes: list[int],
        num_quantizers: int,
        batch_size: int,
        *,
        min_free_gb: float = 3.0,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.capture_sizes = sorted({int(size) for size in capture_sizes if int(size) > 0}, reverse=True)
        self.num_quantizers = int(num_quantizers)
        self.batch_size = int(batch_size)
        self.enabled = bool(enabled)
        self.min_free_bytes = int(float(min_free_gb) * (1024**3))
        self.graphs: dict[int, _CapturedStreamingCodecGraph] = {}
        self._warmed_up = False
        self._graph_t: Counter[int] = Counter()
        self._eager_t: Counter[int] = Counter()

    def _enough_free_vram(self, device: torch.device) -> tuple[bool, int]:
        with torch.cuda.device(device):
            free, _ = torch.cuda.mem_get_info()
        return free >= self.min_free_bytes, int(free)

    @torch.no_grad()
    def _reset_all_slots(self, device: torch.device) -> None:
        reset_mask = torch.ones(self.batch_size, dtype=torch.bool, device=device)

        def _reset(module: object) -> None:
            state = getattr(module, "_streaming_state", None)
            if state is not None:
                state.reset(reset_mask.to(state.device))

        self.model.apply(_reset)

    def warmup(self, device: torch.device) -> None:
        if device.type != "cuda" or not self.enabled or self._warmed_up:
            return

        logger.info(
            "MOSS-TTS streaming codec CUDA Graph warmup: nq=%d batch=%d capture_sizes=%s",
            self.num_quantizers,
            self.batch_size,
            sorted(self.capture_sizes),
        )
        t0 = time.perf_counter()

        for size in self.capture_sizes:
            enough, free = self._enough_free_vram(device)
            if not enough:
                logger.warning(
                    "MOSS-TTS streaming codec CUDA Graph skipped: free VRAM %.1fGB < %.1fGB.",
                    free / 1024**3,
                    self.min_free_bytes / 1024**3,
                )
                break
            try:
                self._capture(size, device)
                logger.info("  Captured streaming CUDA Graph for size=%d", size)
            except Exception:
                self.graphs.pop(size, None)
                logger.warning(
                    "  Failed to capture streaming CUDA Graph for size=%d; this size will use eager.",
                    size,
                    exc_info=True,
                )

        self._reset_all_slots(device)
        self._warmed_up = True
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "MOSS-TTS streaming codec CUDA Graph warmup complete: %d/%d captured in %.1f ms",
            len(self.graphs),
            len(self.capture_sizes),
            elapsed_ms,
        )

    def _capture(self, size: int, device: torch.device) -> None:
        nq = self.num_quantizers
        batch_size = self.batch_size
        static_codes = torch.zeros(nq, batch_size, size, dtype=torch.long, device=device)
        static_lengths = torch.full((batch_size,), size, dtype=torch.long, device=device)
        exec_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        self.model._set_streaming_exec_mask(exec_mask)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            with torch.no_grad():
                _ = self.model._decode_frame(static_codes, static_lengths)
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.accelerator.synchronize(device)

        self._reset_all_slots(device)
        self.model._set_streaming_exec_mask(exec_mask)
        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(
                graph,
                pool=current_platform.get_global_graph_pool(),
                capture_error_mode="thread_local",
            ):
                static_out = self.model._decode_frame(static_codes, static_lengths)

        self.graphs[size] = _CapturedStreamingCodecGraph(
            graph=graph,
            static_codes=static_codes,
            static_lengths=static_lengths,
            static_audio=static_out.audio,
            static_audio_lengths=static_out.audio_lengths,
        )

    @torch.no_grad()
    def decode_step(
        self,
        codes_step: torch.Tensor,
        exec_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.enabled or not self._warmed_up or not codes_step.is_cuda:
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        nq, batch_size, step_t = codes_step.shape
        if nq != self.num_quantizers or batch_size != self.batch_size:
            return None
        entry = self.graphs.get(int(step_t))
        if entry is None:
            self._eager_t[int(step_t)] += 1
            return None

        self.model._set_streaming_exec_mask(exec_mask)
        entry.static_codes.copy_(codes_step)
        entry.graph.replay()
        self._graph_t[int(step_t)] += 1
        return entry.static_audio, entry.static_audio_lengths

    def disable(self) -> None:
        self.enabled = False
        self.graphs.clear()

    def captured_sizes(self) -> list[int]:
        return sorted(self.graphs)

    def log_stats(self) -> None:
        graph = sum(self._graph_t.values())
        eager = sum(self._eager_t.values())
        total = graph + eager
        if total:
            logger.info(
                "MOSS-TTS streaming codec CUDA Graph stats: %d/%d steps graphed; graph T=%s eager T=%s",
                graph,
                total,
                dict(sorted(self._graph_t.items())),
                dict(sorted(self._eager_t.items())),
            )


__all__ = [
    "MossTTSCUDAGraphCodecWrapper",
    "MossTTSStreamingCUDAGraphCodecWrapper",
]

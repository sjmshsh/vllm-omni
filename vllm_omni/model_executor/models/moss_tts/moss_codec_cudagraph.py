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
    # Prefer the streaming decoder entry point when the codec is running inside
    # a persistent ``codec.streaming(batch_size)`` context (v2 tokenizer); fall
    # back to the batched offline decode otherwise. Direct attribute access +
    # ``hasattr`` avoids ``getattr`` per project convention.
    if hasattr(model, "_decode"):
        return model._decode(codes, lengths)
    if hasattr(model, "_decode_frame"):
        return model._decode_frame(codes, lengths)
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

    # Defensive upper bounds against a mis-configured connector (e.g. an
    # accidentally huge ``stream_decode_cudagraph_capture_sizes``): each
    # captured graph carries multi-GB of intermediates at
    # ``batch_size == stream_slots``, so an unbounded set can OOM the box.
    # Sizes exceeding ``max_frames`` are dropped with a warning; once
    # ``max_graphs`` graphs are captured, the remaining sizes are skipped
    # (they fall back to eager exactly like an unmatched T at replay time).
    _DEFAULT_MAX_FRAMES: int = 128
    _DEFAULT_MAX_GRAPHS: int = 160
    # Aligned with SGLang-Omni's MOSS vocoder graph runner: one warm iteration
    # is not always enough to force conv/cudnn workspace + algorithm-selection
    # allocations out of the capture window on all GPUs, so we pay three warm
    # iterations up front (~a few hundred ms per size, once, at startup).
    _DEFAULT_WARMUP_ITERS: int = 3
    # Log capture/replay stats every N steps for long-running sessions so an
    # operator can see graph vs eager mix without waiting for ``close()``.
    _STATS_LOG_INTERVAL: int = 2000

    def __init__(
        self,
        model: object,
        capture_sizes: list[int],
        num_quantizers: int,
        batch_size: int,
        *,
        min_free_gb: float = 3.0,
        enabled: bool = True,
        max_frames: int | None = None,
        max_graphs: int | None = None,
        warmup_iters: int | None = None,
    ) -> None:
        self.model = model
        # Sort largest-first: all per-T graphs share one CUDA mempool, and
        # capturing a larger graph *after* a smaller one grows the pool and
        # invalidates the earlier graphs' recorded addresses (replay would
        # segfault). Aligned with SGLang-Omni.
        self.capture_sizes = sorted({int(size) for size in capture_sizes if int(size) > 0}, reverse=True)
        self.num_quantizers = int(num_quantizers)
        self.batch_size = int(batch_size)
        self.enabled = bool(enabled)
        self.min_free_bytes = int(float(min_free_gb) * (1024**3))
        self.max_frames = int(max_frames) if max_frames is not None else self._DEFAULT_MAX_FRAMES
        self.max_graphs = int(max_graphs) if max_graphs is not None else self._DEFAULT_MAX_GRAPHS
        self.warmup_iters = int(warmup_iters) if warmup_iters is not None else self._DEFAULT_WARMUP_ITERS
        if self.max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {self.max_frames}")
        if self.max_graphs < 1:
            raise ValueError(f"max_graphs must be >= 1, got {self.max_graphs}")
        if self.warmup_iters < 1:
            raise ValueError(f"warmup_iters must be >= 1, got {self.warmup_iters}")
        self.graphs: dict[int, _CapturedStreamingCodecGraph] = {}
        self._warmed_up = False
        self._graph_t: Counter[int] = Counter()
        self._eager_t: Counter[int] = Counter()
        # Total graphed+eager steps served since warmup; used to trigger the
        # periodic stats log at ``_STATS_LOG_INTERVAL`` boundaries.
        self._total_steps: int = 0

    def _enough_free_vram(self, device: torch.device) -> tuple[bool, int]:
        with torch.cuda.device(device):
            free, _ = torch.cuda.mem_get_info()
        return free >= self.min_free_bytes, int(free)

    @torch.no_grad()
    def _reset_all_slots(self, device: torch.device) -> None:
        reset_mask = torch.ones(self.batch_size, dtype=torch.bool, device=device)

        def _reset(module: object) -> None:
            # ``_streaming_state`` is set on every StreamingModule as soon as
            # the codec enters ``codec.streaming(batch_size)``; on non-streaming
            # submodules the attribute simply does not exist. ``hasattr`` +
            # direct access mirrors that and avoids ``getattr``.
            if not hasattr(module, "_streaming_state"):
                return
            state = module._streaming_state
            if state is not None:
                state.reset(reset_mask.to(state.device))

        self.model.apply(_reset)

    def warmup(self, device: torch.device) -> None:
        if device.type != "cuda" or not self.enabled or self._warmed_up:
            return

        # Enforce ``max_frames`` up front so oversized entries are logged once
        # and dropped, rather than surfacing as opaque OOMs mid-capture.
        oversized = [size for size in self.capture_sizes if size > self.max_frames]
        if oversized:
            logger.warning(
                "MOSS-TTS streaming codec CUDA Graph: dropping %d capture size(s) exceeding max_frames=%d: %s",
                len(oversized),
                self.max_frames,
                oversized,
            )
        effective_sizes = [size for size in self.capture_sizes if 1 <= size <= self.max_frames]

        logger.info(
            "MOSS-TTS streaming codec CUDA Graph warmup: nq=%d batch=%d capture_sizes=%s max_frames=%d max_graphs=%d",
            self.num_quantizers,
            self.batch_size,
            effective_sizes,
            self.max_frames,
            self.max_graphs,
        )
        t0 = time.perf_counter()

        # ``effective_sizes`` is already largest-first (inherited from
        # ``self.capture_sizes``), which is what the shared graph pool needs.
        for size in effective_sizes:
            if len(self.graphs) >= self.max_graphs:
                logger.warning(
                    "MOSS-TTS streaming codec CUDA Graph cap max_graphs=%d reached; skipping remaining sizes.",
                    self.max_graphs,
                )
                break
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
            len(effective_sizes),
            elapsed_ms,
        )

    def _capture(self, size: int, device: torch.device) -> None:
        nq = self.num_quantizers
        batch_size = self.batch_size
        static_codes = torch.zeros(nq, batch_size, size, dtype=torch.long, device=device)
        static_lengths = torch.full((batch_size,), size, dtype=torch.long, device=device)
        exec_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        self.model._set_streaming_exec_mask(exec_mask)
        # Side-stream warmup forces lazy allocations (cuDNN algo pick, conv
        # workspaces, etc.) out of the capture window. A single iteration is
        # not always enough on all GPUs, so we run ``warmup_iters`` (aligned
        # with SGLang-Omni's default of 3).
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            with torch.no_grad():
                for _ in range(self.warmup_iters):
                    self.model._decode_frame(static_codes, static_lengths)
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.accelerator.synchronize(device)

        # Reset slots AFTER the warmup and BEFORE the capture: warmup advances
        # per-slot streaming offset; capturing at the warmup-advanced state
        # bakes a wrong start position (~0.4 PCM error at replay).
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
            self._bump_and_maybe_log_stats()
            return None

        self.model._set_streaming_exec_mask(exec_mask)
        entry.static_codes.copy_(codes_step)
        entry.graph.replay()
        self._graph_t[int(step_t)] += 1
        self._bump_and_maybe_log_stats()
        return entry.static_audio, entry.static_audio_lengths

    def _bump_and_maybe_log_stats(self) -> None:
        """Increment the aggregate step counter and periodically emit stats.

        Called from both the graphed and the eager-fallback branches so the
        interval reflects total decode-steps served. A long-running session
        would otherwise only log stats at ``close()``.
        """
        self._total_steps += 1
        if self._total_steps % self._STATS_LOG_INTERVAL == 0:
            self.log_stats()

    def disable(self) -> None:
        self.enabled = False
        self.graphs.clear()

    def captured_sizes(self) -> list[int]:
        return sorted(self.graphs)

    def has_captured(self) -> bool:
        """True when at least one per-T graph is live and replay is possible."""
        return self.enabled and self._warmed_up and bool(self.graphs)

    def log_stats(self) -> None:
        graph = sum(self._graph_t.values())
        eager = sum(self._eager_t.values())
        total = graph + eager
        if total:
            logger.info(
                "MOSS-TTS streaming codec CUDA Graph stats: %d/%d steps graphed (%.1f%%); graph T=%s eager T=%s",
                graph,
                total,
                100.0 * graph / total,
                dict(sorted(self._graph_t.items())),
                dict(sorted(self._eager_t.items())),
            )


__all__ = [
    "MossTTSCUDAGraphCodecWrapper",
    "MossTTSStreamingCUDAGraphCodecWrapper",
]

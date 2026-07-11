# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""MOSS-TTS Stage-1 codec decoder: RVQ codes → 24 kHz waveform."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer import (
    MossAudioTokenizerConfig,
    MossAudioTokenizerModel,
)
from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    MossAudioTokenizerModel as MossAudioTokenizerV2Model,
)
from vllm_omni.model_executor.models.moss_tts.configuration_moss_audio_tokenizer_v2 import (
    MossAudioTokenizerConfig as MossAudioTokenizerV2Config,
)
from vllm_omni.model_executor.models.moss_tts.moss_codec_cudagraph import (
    MossTTSCUDAGraphCodecWrapper,
    MossTTSStreamingCUDAGraphCodecWrapper,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


class _MossCodecStreamSession:
    """Persistent streaming decode session for vendored MOSS-Audio-Tokenizer-v2."""

    def __init__(
        self,
        codec: nn.Module,
        *,
        stream_slots: int,
        n_vq: int,
        enable_cuda_graph: bool = False,
        cuda_graph_capture_sizes: list[int] | None = None,
        cuda_graph_num_of_warmups: int | None = None,
    ) -> None:
        self._codec = codec
        self._stream_slots = int(stream_slots)
        self._batch_size = self._stream_slots
        self._n_vq = int(n_vq)
        self._device = next(codec.parameters()).device
        self._free_stream_slots = list(range(self._stream_slots))
        self._exit_stack = contextlib.ExitStack()
        self._closed = False
        self._cuda_graph_wrapper: MossTTSStreamingCUDAGraphCodecWrapper | None = None
        with torch.no_grad():
            self._exit_stack.enter_context(codec.streaming(self._batch_size))
            if enable_cuda_graph and cuda_graph_capture_sizes:
                self._cuda_graph_wrapper = MossTTSStreamingCUDAGraphCodecWrapper(
                    model=codec,
                    capture_sizes=cuda_graph_capture_sizes,
                    num_quantizers=self._n_vq,
                    batch_size=self._batch_size,
                    warmup_iters=cuda_graph_num_of_warmups,
                    enabled=True,
                )
                try:
                    self._cuda_graph_wrapper.warmup(self._device)
                except Exception:
                    logger.warning(
                        "MOSS codec streaming CUDA Graph warmup failed; serving streaming codec eagerly.",
                        exc_info=True,
                    )
                    self._cuda_graph_wrapper = None
                else:
                    if not self._cuda_graph_wrapper.captured_sizes():
                        self._cuda_graph_wrapper = None

    def acquire(self) -> int | None:
        if not self._free_stream_slots:
            return None
        return self._free_stream_slots.pop()

    def release(self, slot: int) -> None:
        if self._closed:
            return
        self.reset_slots([slot])
        self._free_stream_slots.append(slot)

    def reset_slots(self, slots: list[int]) -> None:
        if not slots:
            return
        reset_mask = torch.zeros(self._batch_size, dtype=torch.bool, device=self._device)
        reset_mask[slots] = True

        def _reset(module: nn.Module) -> None:
            # ``_streaming_state`` is set on every StreamingModule while the
            # codec is inside ``codec.streaming(batch_size)``; non-streaming
            # submodules simply do not define it. ``hasattr`` + direct access
            # avoids ``getattr`` per project convention.
            if not hasattr(module, "_streaming_state"):
                return
            state = module._streaming_state
            if state is not None:
                state.reset(reset_mask.to(state.device))

        with torch.no_grad():
            self._codec.apply(_reset)

    def has_cuda_graph(self) -> bool:
        """True when at least one per-T streaming graph is captured and live."""
        return self._cuda_graph_wrapper is not None and self._cuda_graph_wrapper.has_captured()

    def captured_sizes(self) -> list[int]:
        """Sorted list of frame counts backed by a captured CUDA graph."""
        return self._cuda_graph_wrapper.captured_sizes() if self._cuda_graph_wrapper is not None else []

    def close(self) -> None:
        if self._closed:
            return
        if self._cuda_graph_wrapper is not None:
            self._cuda_graph_wrapper.log_stats()
        with torch.no_grad():
            self._exit_stack.close()
        self._closed = True

    @torch.no_grad()
    def step(self, slot_codes: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        if not slot_codes:
            return {}
        step_lengths = {int(codes.shape[1]) for codes in slot_codes.values()}
        if len(step_lengths) != 1:
            raise ValueError(f"MOSS codec streaming step needs uniform T, got {sorted(step_lengths)}")
        (step_t,) = step_lengths
        codes_step = torch.zeros(
            self._n_vq,
            self._batch_size,
            step_t,
            dtype=torch.long,
            device=self._device,
        )
        codes_lengths = torch.zeros(self._batch_size, dtype=torch.long, device=self._device)
        exec_mask = torch.zeros(self._batch_size, dtype=torch.bool, device=self._device)
        slots = list(slot_codes)
        for slot, codes in slot_codes.items():
            codes_step[:, slot, :] = codes.to(device=self._device, dtype=torch.long)
            codes_lengths[slot] = int(codes.shape[1])
            exec_mask[slot] = True

        graph_out: tuple[torch.Tensor, torch.Tensor] | None = None
        graph_failed = False
        graph_used = False
        try:
            if self._cuda_graph_wrapper is not None:
                try:
                    graph_out = self._cuda_graph_wrapper.decode_step(codes_step, exec_mask)
                except Exception:
                    graph_failed = True
                    raise
            if graph_out is not None:
                graph_used = True
                audio, lengths = graph_out
            else:
                self._codec._set_streaming_exec_mask(exec_mask)
                result = self._codec._decode_frame(codes_step, codes_lengths)
                audio = result.audio
                lengths = result.audio_lengths

            if audio is None:
                return {}
            # CUDA Graph replay errors can surface asynchronously on materialization,
            # so the D2H copy stays inside the replay failure guard.
            audio_cpu = audio[slots].detach().to("cpu", torch.float32)
            lengths_cpu = lengths[slots].detach().to("cpu") if lengths is not None else None
        except Exception:
            if self._cuda_graph_wrapper is not None and (graph_failed or graph_used):
                logger.exception(
                    "MOSS codec streaming CUDA Graph replay failed; disabling graph runner. "
                    "The current streaming chunk will fail and subsequent chunks will use eager decode."
                )
                self._cuda_graph_wrapper.disable()
                self._cuda_graph_wrapper = None
            raise

        out: dict[int, torch.Tensor] = {}
        for index, slot in enumerate(slots):
            wav = audio_cpu[index]
            if lengths_cpu is not None:
                wav = wav[..., : int(lengths_cpu[index].item())]
            out[slot] = wav.contiguous()
        return out


class MossTTSCodecDecoder(nn.Module):
    """Stage-1 decoder for all MOSS-TTS variants.

    Consumes ``(NQ, T)`` audio VQ codes emitted by Stage 0 and decodes them
    to a 24 kHz mono waveform using the vendored
    ``MossAudioTokenizerModel``.

    All five variants share the same codec checkpoint
    ``OpenMOSS-Team/MOSS-Audio-Tokenizer``.  The number of quantizers
    (``n_vq``) is read from ``hf_config`` at construction time and fixed for
    the lifetime of the instance; the same checkpoint can be configured as
    ``n_vq=32`` (MOSS-TTS) or ``n_vq=16`` (all other variants) without
    swapping weights.

    The codec checkpoint path comes from
    ``vllm_config.model_config.hf_config.codec_model_name_or_path``.
    """

    input_modalities = "audio"

    have_multimodal_outputs: bool = True
    has_preprocess: bool = False
    has_postprocess: bool = False
    enable_update_additional_information: bool = True
    requires_raw_input_tokens: bool = True

    _OUTPUT_SAMPLE_RATE: int = 24_000

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config

        cfg = vllm_config.model_config.hf_config
        # ``n_vq`` is the current MOSS naming; some vendored variants still
        # ship the older ``rvq``. Fall through direct attribute access +
        # ``hasattr`` (project convention forbids ``getattr``).
        if hasattr(cfg, "n_vq"):
            self._n_vq: int = int(cfg.n_vq)
        elif hasattr(cfg, "rvq"):
            self._n_vq = int(cfg.rvq)
        else:
            self._n_vq = 16
        # Same story for the codec checkpoint path: modern configs use
        # ``codec_model_name_or_path``; the legacy vendored config exposes
        # ``audio_tokenizer_name_or_path``.
        if hasattr(cfg, "codec_model_name_or_path"):
            self._codec_path: str = str(cfg.codec_model_name_or_path)
        elif hasattr(cfg, "audio_tokenizer_name_or_path"):
            self._codec_path = str(cfg.audio_tokenizer_name_or_path)
        else:
            self._codec_path = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

        # Resolved at load_weights() time, once the codec checkpoint's own
        # config (sampling rate, channel count) is known.
        self._codec: MossAudioTokenizerModel | None = None
        self._cuda_graph_wrapper: MossTTSCUDAGraphCodecWrapper | None = None
        self._n_channels: int = 1
        self._sr_tensor = torch.tensor(self._OUTPUT_SAMPLE_RATE, dtype=torch.int32)
        self._stream_session: _MossCodecStreamSession | None = None
        self._stream_slots: int = self._connector_int("codec_stream_slots", default=0)
        self._stream_chunk_frames: int = self._connector_int("codec_chunk_frames", default=15)
        self._stream_max_step_frames: int = self._connector_int("codec_max_step_frames", default=100)
        self._stream_req_slots: dict[str, int] = {}
        self._stream_pending_codes: dict[str, list[torch.Tensor]] = {}
        self._stream_starved_reqs: set[str] = set()

    # ------------------------------------------------------------------
    # vLLM-Omni stubs (codec has no AR loop)
    # ------------------------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> None:
        return None

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode audio VQ codes to waveform.

        Stage 0 emits flat codebook-major ``[NQ * T_chunk]`` audio codes. The
        chunk transfer adapter assigns those to ``request.prompt_token_ids``,
        which arrives here as ``input_ids`` concatenated across all requests.
        Per-request slice boundaries are computed from
        ``kwargs["seq_token_counts"]`` (token counts, one per request).
        ``runtime_additional_information`` carries per-request metadata such as
        ``left_context_size``.

        Returns
        -------
        OmniOutput with:
          multimodal_outputs["model_outputs"] — list of (T_wav,) float32 tensors
          multimodal_outputs["sr"]            — list of scalar int32 tensors
        """
        sr_tensor = self._sr_tensor
        empty = self._empty_audio()
        info_list: list[dict[str, Any]] = list(runtime_additional_information or [{}])
        num_req = max(len(info_list), 1)

        if self._codec is None:
            logger.warning("MossTTSCodecDecoder called before load_weights(); returning silence.")
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty] * num_req,
                    "sr": [sr_tensor] * num_req,
                },
            )

        audios: list[torch.Tensor] = [empty] * num_req
        srs: list[torch.Tensor] = [sr_tensor] * num_req
        device = next(self._codec.parameters()).device
        streaming_work: list[tuple[int, str, torch.Tensor, bool]] = []

        if input_ids is None or input_ids.numel() == 0:
            for i, wav in self._finish_empty_streaming_requests(info_list).items():
                audios[i] = wav.reshape(-1) if wav.ndim == 1 or int(wav.shape[0]) == 1 else wav
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": audios, "sr": srs},
            )

        # ``input_ids`` is concatenated across all requests. vLLM-Omni runners
        # pass the per-request lengths via the shared code2wav contract
        # ``seq_token_counts``.
        ids_flat = input_ids.reshape(-1).to(dtype=torch.long)
        token_counts = self._normalize_seq_token_counts(kwargs.get("seq_token_counts"))
        if token_counts is None:
            raise RuntimeError(
                "MossTTS codec requires seq_token_counts; otherwise concatenated "
                "codec tokens cannot be split per request."
            )
        input_token_count = sum(token_counts)
        if input_token_count > int(ids_flat.shape[0]):
            raise RuntimeError(
                "MossTTS codec seq_token_counts mismatch: "
                f"counts={token_counts}, sum={input_token_count}, input_tokens={int(ids_flat.shape[0])}."
            )
        # vLLM CUDA Graph replay pads the flattened input to a configured
        # capture size. seq_token_counts describes the real request payload,
        # so discard only that trailing runner padding before splitting.
        ids_flat = ids_flat[:input_token_count]

        num_req = len(token_counts)
        if len(info_list) < num_req:
            info_list.extend({} for _ in range(num_req - len(info_list)))
        elif len(info_list) > num_req:
            info_list = info_list[:num_req]
        if len(audios) < num_req:
            audios.extend(empty for _ in range(num_req - len(audios)))
            srs.extend(sr_tensor for _ in range(num_req - len(srs)))
        elif len(audios) > num_req:
            audios = audios[:num_req]
            srs = srs[:num_req]

        offsets = [0]
        for n in token_counts:
            offsets.append(offsets[-1] + int(n))

        for i, info in enumerate(info_list):
            if i + 1 >= len(offsets):
                break
            seg = ids_flat[offsets[i] : offsets[i + 1]]
            if seg.numel() == 0:
                continue
            meta = (info.get("meta", {}) if isinstance(info, dict) else {}) or {}
            finished = bool(meta.get("stream_finished", meta.get("finished", False)))
            streaming_enabled = bool(meta.get("codec_streaming", False))
            code_flat_numel = meta.get("code_flat_numel")
            if streaming_enabled and finished and code_flat_numel is not None and int(code_flat_numel) == 0:
                for _, wav in self._finish_empty_streaming_requests([info]).items():
                    audios[i] = wav.reshape(-1) if wav.ndim == 1 or int(wav.shape[0]) == 1 else wav
                continue
            if seg.numel() % self._n_vq != 0:
                logger.warning(
                    "MossTTS codec input length %d not divisible by n_vq %d; skipping.",
                    int(seg.numel()),
                    self._n_vq,
                )
                continue
            t_chunk = int(seg.numel() // self._n_vq)
            codes_nq_t = seg.reshape(self._n_vq, t_chunk).to(device=device)
            # Clamp out-of-range codes: the talker uses ``audio_pad_code``
            # (= ``codebook_size``) for delay-pattern padding.  The stage input
            # processor de-delays and drops pad rows before forwarding here, but
            # clamp as a defensive guard against any edge-case leakage.
            codebook_size = self._codec.config.codebook_size
            codes_nq_t = codes_nq_t.clamp_(0, int(codebook_size) - 1)

            left_ctx = meta.get("left_context_size", 0)
            if isinstance(left_ctx, (list, tuple)):
                left_ctx = int(left_ctx[0]) if left_ctx else 0
            elif isinstance(left_ctx, torch.Tensor):
                left_ctx = int(left_ctx.reshape(-1)[0].item()) if left_ctx.numel() else 0
            left_ctx = int(left_ctx)

            req_key = self._runtime_request_key(info, meta, i)

            if streaming_enabled:
                streaming_work.append((i, req_key, codes_nq_t, finished))
                continue

            if self._cuda_graph_wrapper is not None:
                out = self._cuda_graph_wrapper.decode(codes_nq_t)
            else:
                out = self._codec.batch_decode(codes_list=[codes_nq_t], num_quantizers=self._n_vq)

            if out.audio is None:
                continue

            # ``out.audio`` is ``(1, C, T)``; keep the channel axis for
            # stereo codecs (Local-v1.5) and flatten to ``(T,)`` for mono
            # ones (Delay/Realtime) to preserve their existing output shape.
            wav = out.audio[0].to(dtype=torch.float32).cpu()
            if out.audio_lengths is not None:
                wav = wav[..., : int(out.audio_lengths[0].item())]

            wav = self._trim_left_context(wav, left_ctx)

            audios[i] = wav.reshape(-1) if wav.ndim == 1 or int(wav.shape[0]) == 1 else wav

        if streaming_work:
            for i, wav in self._decode_streaming_batch(streaming_work).items():
                audios[i] = wav.reshape(-1) if wav.ndim == 1 or int(wav.shape[0]) == 1 else wav

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    def _finish_empty_streaming_requests(self, info_list: list[dict[str, Any]]) -> dict[int, torch.Tensor]:
        """Release codec stream state for empty finish sentinels.

        Stage-0 can finish on a step that emits no new audio frame. The stage
        input processor forwards that as an empty payload with finished=true.
        Any buffered codes for a request that never acquired a stream slot are
        drained through the streaming session's held slot so the final delta
        is delivered.
        """
        session = self._stream_session
        outputs: dict[int, torch.Tensor] = {}
        if session is None:
            return outputs
        for i, info in enumerate(info_list):
            if not isinstance(info, dict):
                continue
            meta = (info.get("meta", {}) or {}) if isinstance(info.get("meta", {}), dict) else {}
            if not bool(meta.get("codec_streaming", False)):
                continue
            finished = bool(meta.get("stream_finished", meta.get("finished", False)))
            if not finished:
                continue
            req_key = self._runtime_request_key(info, meta, i)
            slot = self._stream_req_slots.get(req_key)
            pending = req_key in self._stream_pending_codes
            if slot is not None or pending:
                try:
                    if pending and slot is not None:
                        pending_codes = self._pop_stream_pending(req_key)
                        if pending_codes.numel() > 0:
                            wav = self._decode_stream_slot_sequence(session, slot, pending_codes)
                            if wav is not None:
                                outputs[i] = wav
                finally:
                    self._finish_stream_request(req_key, session, slot)
        return outputs

    @staticmethod
    def _normalize_seq_token_counts(value: Any) -> list[int] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                "MossTTS codec expects seq_token_counts to be a list/tuple of per-request token counts, "
                f"got {type(value).__name__}."
            )
        counts = [int(item) for item in value]
        if not counts:
            return None
        for count in counts:
            if count < 0:
                raise ValueError(f"MossTTS codec seq_token_counts must be non-negative, got {counts}.")
        return counts

    def _runtime_request_key(self, info: Any, meta: dict[str, Any], index: int) -> str:
        for value in (
            meta.get("req_id"),
            info.get("request_id") if isinstance(info, dict) else None,
        ):
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            if value is not None:
                return str(value)
        return f"moss-codec-stream-{index}"

    def _empty_audio(self) -> torch.Tensor:
        if self._n_channels > 1:
            return torch.zeros((self._n_channels, 0), dtype=torch.float32)
        return torch.zeros((0,), dtype=torch.float32)

    def _trim_left_context(self, wav: torch.Tensor, left_ctx: int) -> torch.Tensor:
        if left_ctx <= 0 or self._codec is None:
            return wav
        trim_samples = left_ctx * self._codec.downsample_rate
        trim = min(trim_samples, wav.shape[-1])
        if trim < trim_samples:
            logger.warning(
                "left_ctx trim (%d samples) exceeds wav length (%d); returning empty audio.",
                trim_samples,
                wav.shape[-1],
            )
        return wav[..., trim:]

    def _ensure_stream_session(self) -> _MossCodecStreamSession | None:
        if self._codec is None:
            return None
        if self._stream_session is not None:
            return self._stream_session
        slots = self._stream_slots
        # ``vllm_config.scheduler_config.max_num_seqs`` is always populated in
        # a live vLLM engine, but we keep a defensive fallback (1) for unit
        # tests that pass a bare-bones config.
        default_slots = 1
        if hasattr(self.vllm_config, "scheduler_config"):
            scheduler_cfg = self.vllm_config.scheduler_config
            if scheduler_cfg is not None and hasattr(scheduler_cfg, "max_num_seqs"):
                default_slots = int(scheduler_cfg.max_num_seqs or 1)
        if slots <= 0:
            slots = default_slots
        self._stream_session = _MossCodecStreamSession(
            self._codec,
            stream_slots=max(1, slots),
            n_vq=self._n_vq,
            enable_cuda_graph=self._stream_cudagraph_enabled(),
            cuda_graph_capture_sizes=self._stream_cudagraph_capture_sizes(),
            cuda_graph_num_of_warmups=self._stream_cudagraph_num_of_warmups(),
        )
        return self._stream_session

    def _decode_streaming_batch(
        self,
        items: list[tuple[int, str, torch.Tensor, bool]],
    ) -> dict[int, torch.Tensor]:
        session = self._ensure_stream_session()
        if session is None:
            return {}

        outputs: dict[int, torch.Tensor] = {}
        grouped: dict[int, list[tuple[int, str, int, torch.Tensor, bool]]] = {}
        step_frame_limit = self._stream_step_frame_limit()

        for output_index, request_id, codes_nq_t, finished in items:
            pending = self._stream_pending_codes.get(request_id)
            slot = self._stream_req_slots.get(request_id)
            if slot is None:
                slot = session.acquire()
                if slot is None:
                    self._append_stream_pending(request_id, codes_nq_t)
                    if request_id not in self._stream_starved_reqs:
                        logger.warning(
                            "MOSS codec streaming slots exhausted; buffering %s until a stream slot is available.",
                            request_id,
                        )
                        self._stream_starved_reqs.add(request_id)
                    if finished:
                        logger.warning(
                            "MOSS codec stream request %s finished before a codec stream slot became available; "
                            "dropping buffered codes.",
                            request_id,
                        )
                        self._finish_stream_request(request_id, session, None)
                    continue
                self._stream_req_slots[request_id] = slot

            if pending:
                self._append_stream_pending(request_id, codes_nq_t)
                replay_codes = self._pop_stream_pending(request_id)
                try:
                    wav = self._decode_stream_slot_sequence(session, slot, replay_codes)
                except Exception:
                    self._finish_stream_request(request_id, session, slot)
                    raise
                if wav is not None:
                    outputs[output_index] = wav
                if finished:
                    self._finish_stream_request(request_id, session, slot)
                continue

            if int(codes_nq_t.shape[1]) > step_frame_limit:
                try:
                    wav = self._decode_stream_slot_sequence(session, slot, codes_nq_t)
                except Exception:
                    self._finish_stream_request(request_id, session, slot)
                    raise
                if wav is not None:
                    outputs[output_index] = wav
                if finished:
                    self._finish_stream_request(request_id, session, slot)
                continue

            grouped.setdefault(int(codes_nq_t.shape[1]), []).append(
                (output_index, request_id, slot, codes_nq_t, finished)
            )

        for group in grouped.values():
            plan = {slot: codes_nq_t for _, _, slot, codes_nq_t, _ in group}
            try:
                decoded = session.step(plan)
            except Exception:
                for _, request_id, slot, _, _ in group:
                    self._finish_stream_request(request_id, session, slot)
                raise
            for output_index, request_id, slot, _, finished in group:
                wav = decoded.get(slot)
                if wav is not None:
                    outputs[output_index] = wav
                if finished:
                    self._finish_stream_request(request_id, session, slot)

        return outputs

    def _append_stream_pending(self, request_id: str, codes_nq_t: torch.Tensor) -> None:
        self._stream_pending_codes.setdefault(request_id, []).append(
            codes_nq_t.detach().to("cpu", torch.long).contiguous()
        )

    def _pop_stream_pending(self, request_id: str) -> torch.Tensor:
        pending = self._stream_pending_codes.pop(request_id, [])
        if not pending:
            return torch.empty((self._n_vq, 0), dtype=torch.long)
        return torch.cat(pending, dim=1).contiguous()

    def _decode_stream_slot_sequence(
        self,
        session: _MossCodecStreamSession,
        slot: int,
        codes_nq_t: torch.Tensor,
    ) -> torch.Tensor | None:
        if codes_nq_t.numel() == 0:
            return None
        step_frame_limit = self._stream_step_frame_limit()
        parts: list[torch.Tensor] = []
        for start in range(0, int(codes_nq_t.shape[1]), step_frame_limit):
            chunk = codes_nq_t[:, start : start + step_frame_limit]
            decoded = session.step({slot: chunk})
            wav = decoded.get(slot)
            if wav is not None:
                parts.append(wav)
        if not parts:
            return None
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def _finish_stream_request(
        self,
        request_id: str,
        session: _MossCodecStreamSession,
        slot: int | None,
    ) -> None:
        if slot is not None:
            session.release(slot)
        self._stream_req_slots.pop(request_id, None)
        self._stream_pending_codes.pop(request_id, None)
        self._stream_starved_reqs.discard(request_id)

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        """Release codec streaming slots when requests finish outside payload flow.

        Normal streaming completion releases slots from ``_decode_streaming_batch``
        when the Stage-0 payload carries ``finished=True``. Client disconnects
        and engine-side aborts can finish a request without delivering that
        terminal payload, so the runner calls this hook from its finished-request
        path to avoid leaking stream slots and buffered codes.
        """
        session = self._stream_session
        for req_id in finished_req_ids:
            request_id = str(req_id)
            slot = self._stream_req_slots.get(request_id)
            has_state = (
                slot is not None or request_id in self._stream_pending_codes or request_id in self._stream_starved_reqs
            )
            if not has_state:
                continue
            if session is not None:
                self._finish_stream_request(request_id, session, slot)
            else:
                self._stream_req_slots.pop(request_id, None)
                self._stream_pending_codes.pop(request_id, None)
                self._stream_starved_reqs.discard(request_id)

    def _connector_int(self, name: str, default: int = 0) -> int:
        value = self._connector_value(name)
        if value is not None:
            return int(value)
        return default

    def _connector_bool(self, name: str, default: bool = False) -> bool:
        value = self._connector_value(name)
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "off", "no"}
        return bool(value)

    def _connector_int_list(self, name: str) -> list[int] | None:
        value = self._connector_value(name)
        if value is None:
            return None
        if isinstance(value, str):
            parsed = [int(v.strip()) for v in value.split(",") if v.strip()]
        elif isinstance(value, (list, tuple)):
            parsed = [int(v) for v in value]
        else:
            parsed = [int(value)]
        return sorted({v for v in parsed if v > 0})

    def _connector_value(self, name: str) -> Any:
        # ``vllm_config.model_config`` is always present on the vLLM engine's
        # config object; ``stage_connector_config`` is a vLLM-Omni extension
        # so its presence is optional. Use ``hasattr`` + direct access
        # (avoids ``getattr`` per project convention).
        if not hasattr(self.vllm_config, "model_config"):
            return None
        model_cfg = self.vllm_config.model_config
        if model_cfg is None or not hasattr(model_cfg, "stage_connector_config"):
            return None
        connector_cfg = model_cfg.stage_connector_config
        if isinstance(connector_cfg, dict):
            extra_cfg: dict | None = connector_cfg.get("extra", connector_cfg)
        elif connector_cfg is not None and hasattr(connector_cfg, "extra"):
            extra_cfg = connector_cfg.extra
        else:
            extra_cfg = None
        if isinstance(extra_cfg, dict) and name in extra_cfg:
            return extra_cfg[name]
        return None

    def _stream_cudagraph_capture_sizes(self) -> list[int]:
        compilation_config = self.vllm_config.compilation_config
        frame_sizes = {
            int(size) // self._n_vq
            for size in compilation_config.cudagraph_capture_sizes
            if int(size) > 0 and int(size) % self._n_vq == 0
        }
        return sorted(size for size in frame_sizes if size <= self._stream_step_frame_limit())

    def _stream_cudagraph_enabled(self) -> bool:
        """Follow vLLM's ``enforce_eager`` CUDA Graph switch."""
        return not self.vllm_config.model_config.enforce_eager and bool(
            self._stream_cudagraph_capture_sizes()
        )

    def _stream_cudagraph_num_of_warmups(self) -> int:
        return int(self.vllm_config.compilation_config.cudagraph_num_of_warmups)

    def _stream_step_frame_limit(self) -> int:
        """Cap streaming codec steps to the steady chunk size.

        This mirrors SGLang-Omni's MOSS Local streaming vocoder: the scheduler
        captures CUDA Graphs for every T in ``1..stream_chunk_frames`` and caps
        backlog/final-tail decode steps at that same ceiling. That keeps the
        normal streaming path on captured exact-T graphs instead of relying on
        large, sparsely used fallback shapes.
        """
        # Both attributes are always initialised in ``__init__``; direct access
        # per project convention.
        stream_chunk_frames = int(self._stream_chunk_frames or 15)
        max_step_frames = int(self._stream_max_step_frames or 100)
        return max(1, min(stream_chunk_frames, max_step_frames))

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Drain the Stage-0 weights iterator, then load the codec from its own checkpoint.

        The codec lives in a separate HuggingFace repo
        (``OpenMOSS-Team/MOSS-Audio-Tokenizer``) and is loaded independently
        of the talker weights.
        """
        # Drain the incoming weights iterator — all Stage-0 weights are
        # irrelevant to this stage.
        for _ in weights:
            pass

        codec_path = self._codec_path
        logger.info("Loading MOSS Audio Tokenizer from %s", codec_path)

        codec_cfg, codec = self._build_codec(codec_path)

        model_loader = DefaultModelLoader(self.vllm_config.load_config)
        source = DefaultModelLoader.Source(
            model_or_path=codec_path,
            revision=None,
            subfolder=None,
        )
        codec_weights = model_loader._get_weights_iterator(source)
        params_dict = dict(codec.named_parameters())

        # Upstream MossAudioTokenizer uses different submodule names than the
        # vendored re-implementation in ``audio_tokenizer.py``. Without this
        # remap only ~half the codec parameters load (codebooks + WN convs)
        # and the rest stay at their random init, which produces noise that
        # sounds correct in duration but is structurally garbage.
        _SUFFIX_REMAP: list[tuple[str, str]] = [
            # v1 (MOSS-Audio-Tokenizer) naming.
            (".self_attn.in_projs.0.", ".attn.in_proj."),
            (".self_attn.out_projs.0.", ".attn.out_proj."),
            (".linear1.", ".ff1."),
            (".linear2.", ".ff2."),
            # v2 checkpoint names use singular in_proj/out_proj and ffn.{0,2};
            # the vendored module keeps the original MOSS layer names.
            (".self_attn.in_proj.", ".self_attn.in_projs.0."),
            (".self_attn.out_proj.", ".self_attn.out_projs.0."),
            (".ffn.0.", ".linear1."),
            (".ffn.2.", ".linear2."),
            (".layer_scale_1.", ".ls1."),
            (".layer_scale_2.", ".ls2."),
            (".input_proj.", ".in_proj."),
            (".output_proj.", ".out_proj."),
        ]

        def _remap(name: str) -> str:
            for src, dst in _SUFFIX_REMAP:
                if src in name:
                    return name.replace(src, dst)
            return name

        loaded_names: set[str] = set()
        skipped: list[str] = []
        shape_mismatches: list[tuple[str, str, tuple[int, ...], tuple[int, ...]]] = []
        for name, tensor in codec_weights:
            # Try direct name first (e.g. ``quantizer.input_proj.*`` exists
            # under the same name in both layouts), then the remap (transformer
            # submodules need ``.linear1.``→``.ff1.`` etc.).
            tgt = name if name in params_dict else _remap(name)
            if tgt in params_dict:
                expected_shape = tuple(params_dict[tgt].shape)
                actual_shape = tuple(tensor.shape)
                if expected_shape != actual_shape:
                    shape_mismatches.append((name, tgt, actual_shape, expected_shape))
                    continue
                default_weight_loader(params_dict[tgt], tensor)
                loaded_names.add(tgt)
            else:
                skipped.append(name)

        missing = sorted(set(params_dict) - loaded_names)
        if missing or skipped or shape_mismatches:
            raise RuntimeError(
                "MOSS Audio Tokenizer weights were not fully loaded: "
                f"loaded={len(loaded_names)}/{len(params_dict)} "
                f"missing={len(missing)} skipped={len(skipped)} "
                f"shape_mismatches={len(shape_mismatches)}; "
                f"first_missing={missing[:5]} "
                f"first_skipped={skipped[:5]} "
                f"first_shape_mismatches={shape_mismatches[:3]}"
            )
        logger.info(
            "MOSS Audio Tokenizer weights: loaded=%d/%d skipped=%d (first skipped: %s)",
            len(loaded_names),
            len(params_dict),
            len(skipped),
            skipped[:3] if skipped else "none",
        )

        device = self.vllm_config.device_config.device
        codec.to(device=device, dtype=torch.float32)
        codec.eval()
        self._codec = codec
        inferred_channels = 2 if "v2" in codec_path.lower() else 1
        # v2 configs expose ``number_channels``; the legacy vendored config
        # uses ``num_channels``; fall back to a path-based inference.
        if hasattr(codec_cfg, "number_channels") and codec_cfg.number_channels:
            self._n_channels = int(codec_cfg.number_channels)
        elif hasattr(codec_cfg, "num_channels") and codec_cfg.num_channels:
            self._n_channels = int(codec_cfg.num_channels)
        else:
            self._n_channels = int(inferred_channels)
        self._sr_tensor = torch.tensor(int(codec_cfg.sampling_rate), dtype=torch.int32)

        logger.info(
            "MOSS Audio Tokenizer loaded: sampling_rate=%d, n_vq=%d, n_channels=%d",
            codec_cfg.sampling_rate,
            codec_cfg.num_quantizers,
            self._n_channels,
        )

        self._maybe_enable_decoder_cudagraph(device)

        # Factory-time streaming CUDA-graph warmup. Aligned with SGLang-Omni's
        # ``scheduler.warmup_now()``: we capture the exact-T streaming graphs
        # at load-time -- codec weights are on device, the GPU is quiescent,
        # and this runs before the stage process is marked ready to serve. The
        # first streaming request therefore hits an already-warm session
        # instead of paying the (potentially multi-second) capture cost inline.
        # The call is best-effort: on non-CUDA devices, when streaming
        # cudagraph is disabled, or on capture failure the session degrades to
        # eager and serving continues.
        self.warmup_now()

        # vLLM's track_weights_loading() compares the returned set against
        # ``self.named_parameters()``. After ``self._codec = codec`` above,
        # those parameters are registered with the ``_codec.`` prefix, so
        # mirror that here.
        return {f"_codec.{name}" for name, _ in codec.named_parameters()}

    # ------------------------------------------------------------------
    # Factory-time warmup
    # ------------------------------------------------------------------

    def warmup_now(self) -> None:
        """Eagerly build the streaming session and capture its CUDA graphs.

        Mirrors SGLang-Omni's ``MossTTSLocalStreamingVocoderScheduler.warmup_now``:
        capturing at factory build time (before the stage process is ready to
        serve) avoids racing a half-captured graph with the serving loop and
        removes the first-streaming-request TTFA spike caused by lazy graph
        capture. Safe to call multiple times -- it no-ops once the session
        already exists.
        """
        if self._codec is None:
            return
        if self._stream_session is not None:
            return
        if not self._stream_cudagraph_enabled():
            # No graph to capture; there is no benefit to eagerly building the
            # streaming session either -- offline non-streaming requests are
            # served through ``self._cuda_graph_wrapper`` (or eager ``batch_decode``)
            # and never touch the streaming session.
            return
        device = next(self._codec.parameters()).device
        if device.type != "cuda":
            return
        session = self._ensure_stream_session()
        if session is None:
            return
        if session.has_cuda_graph():
            logger.info(
                "MOSS-TTS streaming codec CUDA graphs captured at startup: T=%s",
                session.captured_sizes(),
            )
        else:
            logger.warning(
                "MOSS-TTS streaming codec CUDA graphs did not seal at startup "
                "(disabled, low VRAM, or capture failure); serving eager.",
            )

    def _build_codec(self, codec_path: str) -> tuple[Any, nn.Module]:
        try:
            codec_cfg = MossAudioTokenizerV2Config.from_pretrained(codec_path)
            codec = MossAudioTokenizerV2Model(codec_cfg)
            logger.info("Using vendored MOSS Audio Tokenizer v2 classes from %s", codec_path)
            return codec_cfg, codec
        except Exception:
            logger.exception(
                "Failed to instantiate vendored MOSS Audio Tokenizer v2; falling back to legacy vendored codec."
            )

        codec_cfg = MossAudioTokenizerConfig.from_pretrained(codec_path)
        codec = MossAudioTokenizerModel(codec_cfg)
        return codec_cfg, codec

    def _maybe_enable_decoder_cudagraph(self, device: torch.device) -> None:
        """Enable the legacy inner decoder graph when vLLM graphs are off."""
        # Do not nest the codec's legacy graph replay inside a vLLM graph.
        # Deployments using vLLM's compilation config rely on the outer graph;
        # the custom wrapper remains only for backward-compatible configs.
        if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            return
        if self.vllm_config.model_config.enforce_eager:
            return
        if self._codec is None:
            return
        if not self._connector_bool("decode_cudagraph", default=True):
            return

        capture_sizes = self._connector_int_list("decode_cudagraph_capture_sizes") or [
            4,
            8,
            16,
            25,
            32,
            50,
            64,
            100,
            128,
            200,
            256,
        ]

        self._cuda_graph_wrapper = MossTTSCUDAGraphCodecWrapper(
            model=self._codec,
            capture_sizes=capture_sizes,
            num_quantizers=self._n_vq,
            enabled=True,
        )
        self._cuda_graph_wrapper.warmup(device)


__all__ = ["MossTTSCodecDecoder"]

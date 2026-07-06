# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import (
    MossTTSCodecDecoder,
    _MossCodecStreamSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeStreamSession:
    def __init__(self) -> None:
        self.step_sizes: list[int] = []

    def step(self, plan: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        assert len(plan) == 1
        slot, codes = next(iter(plan.items()))
        step_t = int(codes.shape[1])
        self.step_sizes.append(step_t)
        return {slot: torch.ones((1, step_t), dtype=torch.float32)}


class _FakeCodecForForward:
    def __init__(self) -> None:
        self.config = type("_Cfg", (), {"codebook_size": 1024})()
        self.downsample_rate = 1
        self.batch_decode_calls = 0
        self._param = torch.nn.Parameter(torch.zeros(()))

    def parameters(self):
        return iter([self._param])

    def batch_decode(self, *args, **kwargs):
        self.batch_decode_calls += 1
        raise AssertionError("batch_decode should not run in this test")


class _FakeDecodeOutput:
    def __init__(self, audio: torch.Tensor, audio_lengths: torch.Tensor) -> None:
        self.audio = audio
        self.audio_lengths = audio_lengths


class _FakeStreamingCodec:
    def __init__(self) -> None:
        self.eager_calls = 0
        self.exec_masks: list[torch.Tensor] = []

    def _set_streaming_exec_mask(self, exec_mask: torch.Tensor) -> None:
        self.exec_masks.append(exec_mask.detach().cpu().clone())

    def _decode_frame(self, codes: torch.Tensor, lengths: torch.Tensor) -> _FakeDecodeOutput:
        self.eager_calls += 1
        audio = torch.ones((codes.shape[1], 1, codes.shape[2]), dtype=torch.float32)
        return _FakeDecodeOutput(audio=audio, audio_lengths=lengths.detach().cpu())


class _FailingGraphWrapper:
    def __init__(self, output: tuple[object, object] | None = None) -> None:
        self.output = output
        self.disabled = False
        self.calls = 0

    def decode_step(self, codes_step: torch.Tensor, exec_mask: torch.Tensor):
        self.calls += 1
        if self.output is None:
            raise RuntimeError("simulated replay failure")
        return self.output

    def disable(self) -> None:
        self.disabled = True


class _MaterializationFailure:
    def __getitem__(self, index: object) -> object:
        raise RuntimeError("simulated async replay failure")


class _FailingBatchSession:
    def __init__(self) -> None:
        self.released: list[int] = []
        self.plan: dict[int, torch.Tensor] | None = None

    def step(self, plan: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        self.plan = plan
        raise RuntimeError("simulated streaming step failure")

    def release(self, slot: int) -> None:
        self.released.append(slot)


def _decoder_stub(
    *,
    stream_chunk_frames: int = 4,
    max_step_frames: int = 100,
    configured_capture_sizes: list[int] | None = None,
) -> MossTTSCodecDecoder:
    decoder = MossTTSCodecDecoder.__new__(MossTTSCodecDecoder)
    object.__setattr__(decoder, "_stream_chunk_frames", stream_chunk_frames)
    object.__setattr__(decoder, "_stream_max_step_frames", max_step_frames)
    object.__setattr__(
        decoder,
        "_connector_int_list",
        lambda name: configured_capture_sizes,
    )
    return decoder


def _decoder_forward_stub() -> tuple[MossTTSCodecDecoder, _FakeCodecForForward]:
    decoder = _decoder_stub(stream_chunk_frames=4, max_step_frames=7)
    codec = _FakeCodecForForward()
    object.__setattr__(decoder, "_codec", codec)
    object.__setattr__(decoder, "_n_vq", 2)
    object.__setattr__(decoder, "_n_channels", 1)
    object.__setattr__(decoder, "_sr_tensor", torch.tensor(24_000, dtype=torch.int32))
    object.__setattr__(decoder, "_cuda_graph_wrapper", None)
    object.__setattr__(decoder, "_stream_req_slots", {})
    object.__setattr__(decoder, "_stream_pending_codes", {})
    object.__setattr__(decoder, "_stream_starved_reqs", set())
    return decoder, codec


def _session_stub(
    codec: _FakeStreamingCodec,
    wrapper: _FailingGraphWrapper | None,
) -> _MossCodecStreamSession:
    session = _MossCodecStreamSession.__new__(_MossCodecStreamSession)
    object.__setattr__(session, "_codec", codec)
    object.__setattr__(session, "_n_vq", 2)
    object.__setattr__(session, "_batch_size", 1)
    object.__setattr__(session, "_device", torch.device("cpu"))
    object.__setattr__(session, "_cuda_graph_wrapper", wrapper)
    return session


def test_default_stream_cudagraph_sizes_cover_steady_chunk_exactly():
    decoder = _decoder_stub(stream_chunk_frames=4, max_step_frames=100)

    assert decoder._stream_step_frame_limit() == 4
    assert decoder._stream_cudagraph_capture_sizes() == [1, 2, 3, 4]


def test_stream_step_limit_respects_max_step_frames():
    decoder = _decoder_stub(stream_chunk_frames=64, max_step_frames=16)

    assert decoder._stream_step_frame_limit() == 16
    assert decoder._stream_cudagraph_capture_sizes() == list(range(1, 17))


def test_configured_stream_cudagraph_sizes_still_override_default():
    decoder = _decoder_stub(
        stream_chunk_frames=4,
        max_step_frames=100,
        configured_capture_sizes=[2, 5],
    )

    assert decoder._stream_cudagraph_capture_sizes() == [2, 5]


def test_long_streaming_sequence_splits_at_steady_chunk_limit():
    decoder = _decoder_stub(stream_chunk_frames=4, max_step_frames=100)
    session = _FakeStreamSession()
    codes = torch.arange(2 * 10, dtype=torch.long).reshape(2, 10)

    wav = decoder._decode_stream_slot_sequence(session, slot=3, codes_nq_t=codes)

    assert session.step_sizes == [4, 4, 2]
    assert wav is not None
    assert tuple(wav.shape) == (1, 10)


def test_streaming_graph_replay_failure_disables_runner_without_eager_retry():
    codec = _FakeStreamingCodec()
    wrapper = _FailingGraphWrapper()
    session = _session_stub(codec, wrapper)
    codes = torch.zeros((2, 3), dtype=torch.long)

    with pytest.raises(RuntimeError, match="simulated replay failure"):
        session.step({0: codes})

    assert wrapper.disabled
    assert session._cuda_graph_wrapper is None
    assert codec.eager_calls == 0


def test_streaming_graph_materialization_failure_disables_runner():
    codec = _FakeStreamingCodec()
    lengths = torch.ones((1,), dtype=torch.long)
    wrapper = _FailingGraphWrapper(output=(_MaterializationFailure(), lengths))
    session = _session_stub(codec, wrapper)
    codes = torch.zeros((2, 3), dtype=torch.long)

    with pytest.raises(RuntimeError, match="simulated async replay failure"):
        session.step({0: codes})

    assert wrapper.disabled
    assert session._cuda_graph_wrapper is None
    assert codec.eager_calls == 0


def test_streaming_batch_failure_releases_participating_slots():
    decoder = _decoder_stub(stream_chunk_frames=4, max_step_frames=100)
    session = _FailingBatchSession()
    object.__setattr__(decoder, "_stream_req_slots", {"r1": 1, "r2": 2})
    object.__setattr__(decoder, "_stream_pending_codes", {"r1": [], "r2": []})
    object.__setattr__(decoder, "_stream_starved_reqs", {"r1", "r2"})
    object.__setattr__(decoder, "_ensure_stream_session", lambda: session)
    codes = torch.zeros((2, 3), dtype=torch.long)

    with pytest.raises(RuntimeError, match="simulated streaming step failure"):
        decoder._decode_streaming_batch(
            [
                (0, "r1", codes, False),
                (1, "r2", codes, False),
            ]
        )

    assert session.released == [1, 2]
    assert decoder._stream_req_slots == {}
    assert decoder._stream_pending_codes == {}
    assert decoder._stream_starved_reqs == set()

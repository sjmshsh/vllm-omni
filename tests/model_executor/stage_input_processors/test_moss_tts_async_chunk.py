# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.moss_tts import talker2codec_raw_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _req(rid: str, *, initial_codec_chunk_frames: int | None = None):
    ai = None
    if initial_codec_chunk_frames is not None:
        entry = SimpleNamespace(list_data=[initial_codec_chunk_frames])
        ai = SimpleNamespace(entries={"initial_codec_chunk_frames": entry})
    return SimpleNamespace(
        external_req_id=rid,
        request_id=rid,
        additional_information=ai,
    )


def _tm(*, chunk_frames: int = 10, initial_chunk_frames: int = 0):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        put_req_chunk=defaultdict(int),
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "initial_codec_chunk_frames": initial_chunk_frames,
                }
            }
        ),
    )


def _call(tm, rid: str, frame: list[int], *, finished: bool = False, req_ic: int | None = None):
    return talker2codec_raw_async_chunk(
        transfer_manager=tm,
        multimodal_output={"codes": {"audio": torch.tensor([frame], dtype=torch.long)}},
        request=_req(rid, initial_codec_chunk_frames=req_ic),
        is_finished=finished,
    )


def test_request_initial_codec_chunk_frames_overrides_connector_first_chunk():
    tm = _tm(chunk_frames=10, initial_chunk_frames=0)

    assert _call(tm, "r", [1, 2, 3], req_ic=2) is None
    payload = _call(tm, "r", [4, 5, 6], req_ic=2)

    assert payload is not None
    assert payload.meta.codec_streaming is True
    assert payload.meta.codec_chunk_frames == 2
    assert payload.meta.stream_finished.item() is False
    assert payload.codes.audio.tolist() == [1, 4, 2, 5, 3, 6]


def test_request_initial_codec_chunk_frames_only_controls_first_chunk():
    tm = _tm(chunk_frames=4, initial_chunk_frames=0)

    first = _call(tm, "r", [1, 2], req_ic=2)
    assert first is None
    first = _call(tm, "r", [3, 4], req_ic=2)
    assert first is not None
    tm.put_req_chunk["r"] += 1

    assert _call(tm, "r", [5, 6], req_ic=2) is None
    assert _call(tm, "r", [7, 8], req_ic=2) is None
    assert _call(tm, "r", [9, 10], req_ic=2) is None
    second = _call(tm, "r", [11, 12], req_ic=2)

    assert second is not None
    assert second.meta.codec_chunk_frames == 4
    assert second.codes.audio.tolist() == [5, 7, 9, 11, 6, 8, 10, 12]

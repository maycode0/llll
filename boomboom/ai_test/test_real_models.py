from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ai_coding.models.factory import build_surrogate, build_victim
from ai_coding.models.hf_text_classifier import detokenize_words
from ai_coding.models.real_surrogate import RealSurrogateModel
from ai_coding.models.real_victim import RealVictimModel


def test_detokenize_words_handles_punctuation() -> None:
    words = ["that", "'s", "a", "film", ",", "really", "."]
    assert detokenize_words(words) == "that's a film, really."


def test_build_surrogate_rejects_unknown_kind(experiment_config) -> None:
    with pytest.raises(ValueError):
        build_surrogate("unknown", "unused", config=experiment_config, device="cpu")


def test_build_victim_rejects_unknown_kind(experiment_config) -> None:
    with pytest.raises(ValueError):
        build_victim("unknown", "unused", config=experiment_config, device="cpu")


@pytest.mark.skipif(not Path(r"E:\modelHub\bert-base-uncased-SST-2").exists(), reason="Local BERT SST-2 model is unavailable")
def test_real_surrogate_loads_local_hf_model(experiment_config) -> None:
    surrogate = build_surrogate(
        "hf",
        r"E:\modelHub\bert-base-uncased-SST-2",
        config=experiment_config,
        device="cpu",
        max_length=64,
    )
    assert isinstance(surrogate, RealSurrogateModel)
    support = surrogate.score_label_support(["a", "beautiful", "film"], 1)
    assert isinstance(support, float)


@pytest.mark.skipif(not Path(r"E:\modelHub\gpt2-finetuned-sst2").exists(), reason="Local GPT-2 SST-2 model is unavailable")
def test_real_victim_loads_local_hf_model(experiment_config) -> None:
    victim = build_victim(
        "hf",
        r"E:\modelHub\gpt2-finetuned-sst2",
        config=experiment_config,
        device="cpu",
        max_length=64,
    )
    assert isinstance(victim, RealVictimModel)
    label = victim.predict_label(["a", "beautiful", "film"])
    assert label in (0, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(not Path(r"E:\modelHub\bert-base-uncased-SST-2").exists(), reason="Local BERT SST-2 model is unavailable")
def test_real_surrogate_can_use_cuda(experiment_config) -> None:
    surrogate = build_surrogate(
        "hf",
        r"E:\modelHub\bert-base-uncased-SST-2",
        config=experiment_config,
        device="cuda",
        max_length=64,
    )
    assert isinstance(surrogate, RealSurrogateModel)
    assert surrogate.classifier.device.type == "cuda"

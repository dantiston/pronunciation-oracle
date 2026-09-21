from types import SimpleNamespace

from pronunciation_oracle.align.torchaudio_ctc import TorchaudioCTCAligner, _resolve_auto_device


def _fake_torch(*, cuda: bool, mps: bool):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


def test_resolve_auto_device_prefers_cuda():
    assert _resolve_auto_device(_fake_torch(cuda=True, mps=True)) == "cuda"


def test_resolve_auto_device_falls_back_to_mps():
    assert _resolve_auto_device(_fake_torch(cuda=False, mps=True)) == "mps"


def test_resolve_auto_device_falls_back_to_cpu():
    assert _resolve_auto_device(_fake_torch(cuda=False, mps=False)) == "cpu"


def test_default_device_is_auto_and_unresolved_before_use():
    aligner = TorchaudioCTCAligner()
    assert aligner._device == "auto"


def test_explicit_device_is_kept_as_given():
    aligner = TorchaudioCTCAligner(device="cpu")
    assert aligner._device == "cpu"

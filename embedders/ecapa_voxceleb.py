"""ECAPA-TDNN trained on VoxCeleb, from SpeechBrain.

    pip install speechbrain
"""
import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier

_model = None


def _load():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = (EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                 run_opts={"device": device}), device)
    return _model


def embed(waveforms, lengths):
    model, device = _load()
    x = torch.from_numpy(np.stack(waveforms)).to(device)
    rel = torch.tensor([n / x.shape[1] for n in lengths], dtype=torch.float, device=device)
    with torch.no_grad():
        return model.encode_batch(x, wav_lens=rel).squeeze(1).cpu().numpy()

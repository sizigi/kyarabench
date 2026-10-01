"""The WavLM-base+ speaker-verification head from Microsoft.

    pip install transformers
"""
import numpy as np
import torch
from transformers import WavLMForXVector

_model = None


def _load():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = (WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval(),
                  device)
    return _model


def embed(waveforms, lengths):
    model, device = _load()
    x = torch.from_numpy(np.stack(waveforms)).to(device)
    with torch.no_grad():
        return model(input_values=x, attention_mask=None).embeddings.cpu().numpy()

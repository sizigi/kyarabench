"""KyaraEmbed.

Set KYARAEMBED_CKPT to the checkpoint. The architecture is the public anime ECAPA-TDNN
character embedder, so that package supplies the module and this file loads the weights.

    pip install anime-speaker-embedding
"""
import os
import numpy as np
import torch
from anime_speaker_embedding import AnimeSpeakerEmbedding

_model = None


def _load():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = AnimeSpeakerEmbedding(device=device, variant="char")
        ckpt = os.environ.get("KYARAEMBED_CKPT")
        if ckpt:
            m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        _model = (m, device)
    return _model


def embed(waveforms, lengths):
    """Fixed-length waveforms -> (N, 192) embeddings."""
    model, device = _load()
    x = torch.from_numpy(np.stack(waveforms))
    with torch.no_grad():
        return model(x.to(device)).squeeze(1).cpu().numpy()

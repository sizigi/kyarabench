"""ReDimNet-b2, the VoxCeleb2 fine-tuned checkpoint from the authors' torch.hub entry.

    pip install torch
"""
import numpy as np
import torch

_model = None


def _load():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = torch.hub.load("IDRnD/ReDimNet", "ReDimNet", model_name="b2", train_type="ft_lm",
                           dataset="vox2", trust_repo=True).to(device).eval()
        _model = (m, device)
    return _model


def embed(waveforms, lengths):
    model, device = _load()
    x = torch.from_numpy(np.stack(waveforms)).to(device)
    with torch.no_grad():
        return model(x).cpu().numpy()

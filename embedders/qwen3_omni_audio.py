"""The audio tower of Qwen3-Omni, mean-pooled over encoder states.

Only the first shard of the checkpoint is needed: the audio encoder lives there entirely. The
tower is instantiated from the model config and the audio weights are loaded into it, so the
language model is never built. Clips go through one at a time because the Whisper-style feature
extractor pads to its own fixed window anyway.

    pip install torch transformers huggingface-hub safetensors
"""
import json
import numpy as np
import torch

_state = None


def _load():
    global _state
    if _state is None:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import WhisperFeatureExtractor
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig

        model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        shard = hf_hub_download(model_id, "model-00001-of-00015.safetensors")
        config = json.load(open(hf_hub_download(model_id, "config.json")))
        audio_config = config.get("thinker_config", {}).get("audio_config") or config["audio_config"]
        encoder = Qwen3OmniMoeAudioEncoder(Qwen3OmniMoeAudioEncoderConfig(**audio_config))

        weights = load_file(shard)
        prefix = next(k.split("audio_tower.")[0] + "audio_tower." for k in weights if "audio_tower." in k)
        encoder.load_state_dict({k[len(prefix):]: v for k, v in weights.items()
                                 if k.startswith(prefix)}, strict=False)
        encoder = encoder.cuda().eval().half()

        preprocessor = json.load(open(hf_hub_download(model_id, "preprocessor_config.json")))
        keep = ("feature_size", "sampling_rate", "hop_length", "chunk_length", "n_fft",
                "n_samples", "nb_max_frames", "padding_side", "padding_value",
                "return_attention_mask")
        extractor = WhisperFeatureExtractor(**{k: v for k, v in preprocessor.items() if k in keep})
        _state = (encoder, extractor)
    return _state


def embed(waveforms, lengths):
    encoder, extractor = _load()
    out = []
    with torch.no_grad():
        for wav in waveforms:
            features = extractor(wav, sampling_rate=16000, return_tensors="pt")
            mel = features.input_features.cuda().half().squeeze(0)
            hidden = encoder(input_features=mel,
                             feature_lens=torch.tensor([mel.shape[-1]], device="cuda"))
            hidden = hidden.last_hidden_state if hasattr(hidden, "last_hidden_state") else hidden[0]
            hidden = hidden.float()
            out.append(hidden.mean(dim=tuple(range(hidden.dim() - 1))).cpu().numpy())
    return np.stack(out)

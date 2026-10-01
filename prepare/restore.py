#!/usr/bin/env python3
"""Sidon restoration of the KyaraBench source clips.

Restores one file at a time, following sidon_cli.py from the Sidon demo (MIT,
https://huggingface.co/spaces/sarulab-speech/sidon_demo_beta):

  load        soundfile, mono by channel mean, float32
  normalise   peak to 0.9
  filter      50 Hz high-pass (torchaudio highpass_biquad)
  resample    to 16 kHz, then 24,000 samples (1.5 s) of silence appended
  features    40 zero samples on each side, transformers SeamlessM4TFeatureExtractor
              (facebook/w2v-bert-2.0), Sidon TorchScript feature extractor and decoder in fp32
  chunking    60 s chunks at 16 kHz; later chunks are decoded with the last 5 feature frames of
              the previous chunk prepended and their first 4,800 output samples dropped
  output      first 150 samples dropped (warm-up), cut to int(48000 / sr * n_samples) so the
              clip keeps its source duration, written as 48 kHz 16-bit PCM

Differences from sidon_cli.py: the models load once instead of once per file, values beyond
+-1 are clipped instead of wrapping around in the int16 cast, and inputs/outputs follow the
KyaraBench layout.

Sidon resynthesises the waveform rather than filtering it, so two runs of this script on the
same input give perceptually equivalent but not identical audio.

Sidon ships its feature extractor and decoder as CUDA-traced TorchScript, so a working CUDA
device is required: the graphs carry CUDA device constants that map_location cannot move.

    python prepare/restore.py --input-root anim400k_audio_clips --output-root restored \
        --clips-json data/kyarabench_v1.0_source_clips.json

Input:  <input-root>/<uuid[:2]>/<uuid>.mp3 (any audio extension with --clips-json absent)
Output: <output-root>/<uuid[:2]>/<uuid>_restored.wav, the path KyaraBench's loader reads.
Existing outputs are skipped, so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import SeamlessM4TFeatureExtractor

SR_IN = 16_000
SR_OUT = 48_000
TAIL_PAD = 24_000       # 1.5 s at 16 kHz
EDGE_PAD = 40           # samples on each side of a chunk
CHUNK = SR_IN * 60      # 60 s
CACHE_FRAMES = 5        # feature frames carried into the next chunk
CHUNK_CUT = 4_800       # output samples dropped from a continued chunk
WARMUP_CUT = 50 * 3     # output samples dropped from the first chunk


def require_cuda():
    """Fail early and legibly when CUDA is missing or the torch build does not match the driver."""
    hint = ("Check the driver with `nvidia-smi` and install a matching wheel, e.g.\n"
            "  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128")
    if not torch.cuda.is_available():
        raise SystemExit(
            f"Sidon's checkpoints are CUDA-traced TorchScript and need a usable GPU, but torch "
            f"{torch.__version__} reports none.\n{hint}")
    try:
        torch.zeros(1, device="cuda")
    except RuntimeError as exc:
        raise SystemExit(
            f"A GPU is present but torch {torch.__version__} cannot use it:\n  {exc}\n{hint}")


def load_models(device: str, repo: str):
    fe = torch.jit.load(hf_hub_download(repo, "feature_extractor_cuda.pt"), map_location=device).to(device).eval()
    dec = torch.jit.load(hf_hub_download(repo, "decoder_cuda.pt"), map_location=device).to(device).eval()
    pre = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    return fe, dec, pre


@torch.inference_mode()
def restore(path: Path, fe, dec, pre, device: str) -> tuple[np.ndarray, int]:
    """Return the restored 48 kHz int16 waveform and how many samples were clipped."""
    wav, sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    wav = 0.9 * (wav / (np.abs(wav).max() + 1e-9))
    wav = torchaudio.functional.highpass_biquad(torch.from_numpy(wav).unsqueeze(0), sr, 50)
    target = int(SR_OUT / sr * wav.shape[-1])
    wav16 = torchaudio.functional.resample(wav, sr, SR_IN)
    wav16 = torch.nn.functional.pad(wav16, (0, TAIL_PAD)).view(-1)

    pieces, cache = [], None
    for chunk in wav16.split(CHUNK):
        inputs = pre(torch.nn.functional.pad(chunk, (EDGE_PAD, EDGE_PAD)).numpy(),
                     sampling_rate=SR_IN, return_tensors="pt")
        feature = fe(inputs["input_features"].to(device))["last_hidden_state"]
        if cache is not None:
            y = dec(torch.cat([cache, feature], dim=1).transpose(1, 2))[:, :, CHUNK_CUT:]
        else:
            y = dec(feature.transpose(1, 2))[:, :, WARMUP_CUT:]
        cache = feature[:, -CACHE_FRAMES:, :]
        pieces.append(y.float().cpu())

    out = torch.cat(pieces, dim=-1).view(-1)[:target].numpy()
    clipped = int((np.abs(out) > 1.0).sum())
    return (np.clip(out, -1.0, 1.0) * 32767.0).astype(np.int16), clipped


def collect(args) -> list[tuple[Path, Path, str | None]]:
    """(input, output, expected sha256) for every clip to restore."""
    root_in, root_out = args.input_root, args.output_root
    if args.clips_json:
        clips = json.load(open(args.clips_json))["clips"]
        return [(root_in / c["source_file"], root_out / c["uuid"][:2] / f"{c['uuid']}_restored.wav",
                 c.get("source_sha256")) for c in clips]
    exts = {"." + e.strip().lower() for e in args.include_ext.split(",")}
    files = sorted(p for p in root_in.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    return [(p, root_out / p.relative_to(root_in).parent / f"{p.stem}_restored.wav", None) for p in files]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--clips-json", type=Path, default=None,
                    help="kyarabench_v1.0_source_clips.json: restore exactly these clips and check their sha256")
    ap.add_argument("--include-ext", default="mp3,wav,flac", help="extensions to scan without --clips-json")
    ap.add_argument("--model-repo", default="sarulab-speech/sidon-v0.1")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    args = ap.parse_args()

    require_cuda()
    args.device = "cuda"

    # fp32 throughout, as in sidon_cli.py
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    jobs = collect(args)[args.shard_id::args.num_shards]
    missing = [str(i) for i, _, _ in jobs if not i.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} inputs are missing, e.g. {missing[0]}")
    todo = [j for j in jobs if not j[1].exists()]
    print(f"{len(jobs)} clips, {len(jobs) - len(todo)} already restored, {len(todo)} to go", flush=True)
    if not todo:
        return

    fe, dec, pre = load_models(args.device, args.model_repo)
    bad_hash, clipped_files, start = [], 0, time.time()
    for k, (src, dst, sha) in enumerate(todo, 1):
        if sha and hashlib.sha256(src.read_bytes()).hexdigest() != sha:
            bad_hash.append(src.name)
            continue
        out, clipped = restore(src, fe, dec, pre, args.device)
        clipped_files += clipped > 0
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp.wav")
        sf.write(str(tmp), out, SR_OUT, subtype="PCM_16")
        tmp.replace(dst)
        if k % 100 == 0 or k == len(todo):
            print(f"  {k}/{len(todo)}  {time.time() - start:.0f}s", flush=True)

    print(f"done; sha256 mismatches skipped: {len(bad_hash)}, files with clipped samples: {clipped_files}")
    if bad_hash:
        print("mismatched:", ", ".join(bad_hash[:10]))


if __name__ == "__main__":
    main()

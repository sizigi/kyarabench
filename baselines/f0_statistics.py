"""The pitch-only shortcut row.

Each clip is reduced to six statistics of its voiced pitch track and nothing else, so a
condition this row does well on is a condition solvable by pitch alone.

It does not go through score.py because it is not compared by cosine: features are z-scored
across the corpus, a bundle averages them, and a trial is scored by the negative Euclidean
distance between the two bundles. Conditions and the EER definition come from the protocol, so
this row is scored on the same trials as every other.

    python baselines/f0_statistics.py --clips /path/to/restored
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kyarabench.audio import SR, clip_path, load_clip
from kyarabench import protocol

WINDOW_SECONDS = 0.5
HOP = 256
FRAME = 1024
FMIN, FMAX = 65, 500
MIN_VOICED_FRAMES = 6

ap = argparse.ArgumentParser()
ap.add_argument("--clips", required=True)
ap.add_argument("--manifest", default="data/kyarabench_v1.0_manifest.json")
ap.add_argument("--cache", default=None, help="store the pitch tracks here to skip re-extraction")
ap.add_argument("--out", default=None)
A = ap.parse_args()

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
manifest = json.load(open(A.manifest if os.path.isabs(A.manifest) else f"{here}/{A.manifest}"))


def pitch_windows(wav, n_samples):
    """Median voiced pitch in semitones per half-second window, None where too little is voiced."""
    import librosa
    wav = wav[:n_samples]
    f0, voiced, _ = librosa.pyin(wav, fmin=FMIN, fmax=FMAX, sr=SR,
                                 frame_length=FRAME, hop_length=HOP)
    ok = np.isfinite(f0)
    if voiced is not None:
        ok &= voiced
    t = np.arange(len(f0)) * HOP / SR
    n_windows = max(1, int(np.ceil((n_samples / SR) / WINDOW_SECONDS)))
    out = []
    for b in range(n_windows):
        m = ok & (t >= b * WINDOW_SECONDS) & (t < (b + 1) * WINDOW_SECONDS)
        out.append(float(12 * np.log2(np.median(f0[m]))) if m.sum() >= MIN_VOICED_FRAMES else None)
    return out


def features(sequence):
    """Median, spread, range, voiced fraction, and how fast the pitch moves between windows."""
    v = np.array([x for x in sequence if x is not None], dtype=float)
    if len(v) < 3:
        return None
    d = np.abs(np.diff(v))
    return np.array([np.median(v), v.std(), np.percentile(v, 90) - np.percentile(v, 10),
                     len(v) / max(len(sequence), 1),
                     np.median(d), d.std() if len(d) > 1 else 0.0])


wanted = set()
for name, (positives, negatives) in protocol.CONDITIONS.items():
    for key in positives + ([negatives] if negatives else []):
        for t in manifest["trials"][key]:
            for side in ("a", "b"):
                wanted.update(protocol._bundle(t[side]))
for clips in manifest["xling_retrieval"]["gallery"].values():
    wanted.update(clips)
for q in manifest["xling_retrieval"]["queries"]:
    wanted.update(q["q"])
uuids = sorted(u for u in wanted if os.path.exists(clip_path(A.clips, u)))
print(f"{len(uuids)} clips")

tracks = {}
if A.cache and os.path.exists(A.cache):
    tracks = json.load(open(A.cache))
todo = [u for u in uuids if u not in tracks]
for n, u in enumerate(todo, 1):
    wav, real = load_clip(clip_path(A.clips, u))
    tracks[u] = pitch_windows(wav, real)
    if n % 200 == 0:
        print(f"  {n}/{len(todo)}", flush=True)
if A.cache and todo:
    json.dump(tracks, open(A.cache, "w"))

raw = {u: features(tracks[u]) for u in uuids}
raw = {u: f for u, f in raw.items() if f is not None}
stack = np.stack(list(raw.values()))
mean, sd = stack.mean(0), stack.std(0) + 1e-9
z = {u: (f - mean) / sd for u, f in raw.items()}
print(f"{len(z)} clips carry a usable pitch track")

index = {u: i for i, u in enumerate(sorted(z))}
F = np.stack([z[u] for u in sorted(z)])


def bundle(rows):
    return F[rows].mean(axis=0)


trials = protocol.build_trials(manifest, index)
result = {}
for name, rows in trials.items():
    score = np.array([-float(np.linalg.norm(bundle(a) - bundle(b))) for a, b, *_ in rows])
    result[name] = protocol.eer(score, protocol.labels(rows))

gallery_keys = sorted(manifest["xling_retrieval"]["gallery"])
gallery, kept = [], []
for k in gallery_keys:
    rows = [index[u] for u in manifest["xling_retrieval"]["gallery"][k] if u in index]
    if len(rows) >= 2:
        gallery.append(bundle(rows))
        kept.append(k)
gallery = np.stack(gallery)
per_character = {}
for q in manifest["xling_retrieval"]["queries"]:
    rows = [index[u] for u in q["q"] if u in index]
    if not rows or q["char"] not in kept:
        continue
    v = bundle(rows)
    predicted = kept[int(np.argmin(np.linalg.norm(gallery - v, axis=1)))]
    per_character.setdefault(q["char"], []).append(float(predicted == q["char"]))
result["R@1"] = sum(sum(v) / len(v) for v in per_character.values()) / len(per_character)
print(f"retrieval over {len(kept)} gallery characters, "
      f"{sum(len(v) for v in per_character.values())} queries")

print()
print(" ".join(f"{c:>9}" for c in protocol.REPORTED))
print(" ".join(f"{result[c]:>9.3f}" for c in protocol.REPORTED))
if A.out:
    json.dump(result, open(A.out, "w"), indent=1)
    print(f"\n-> {A.out}")

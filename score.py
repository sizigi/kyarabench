"""Score a system on KyaraBench.

A system is a Python module exposing one function:

    embed(waveforms, lengths) -> np.ndarray of shape (len(waveforms), dim)

`waveforms` are mono 16 kHz float32 arrays of a fixed 10 s, read through the reference loader.
`lengths` gives how many samples of each are real audio, for models that accept a length or an
attention mask.

    python score.py --clips /path/to/restored --embedder embedders.kyaraembed

Every clip is the same length, so a batch needs no padding and `--batch` only trades memory
for speed.
"""
import argparse, importlib, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kyarabench.audio import clip_path, load_clip
from kyarabench import protocol

ap = argparse.ArgumentParser()
ap.add_argument("--clips", required=True, help="root of the restored clips")
ap.add_argument("--embedder", required=True, help="module exposing embed(waveforms, lengths)")
ap.add_argument("--manifest", default="data/kyarabench_v1.0_manifest.json")
ap.add_argument("--batch", type=int, default=16, help="clips per embed() call")
ap.add_argument("--out", default=None, help="write the result as JSON")
ap.add_argument("--save-embeddings", default=None, help="cache the embedding matrix here")
A = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
manifest = json.load(open(A.manifest if os.path.isabs(A.manifest) else f"{here}/{A.manifest}"))

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

uuids = sorted(wanted)
missing = {u for u in uuids if not os.path.exists(clip_path(A.clips, u))}
if missing:
    print(f"warning: {len(missing)} of {len(uuids)} clips are missing; their trials are dropped")
    uuids = [u for u in uuids if u not in missing]
index = {u: i for i, u in enumerate(uuids)}
print(f"{len(uuids)} clips")

embed = importlib.import_module(A.embedder).embed
vectors = []
for i in range(0, len(uuids), A.batch):
    loaded = [load_clip(clip_path(A.clips, u)) for u in uuids[i:i + A.batch]]
    out = np.asarray(embed([w for w, _ in loaded], [n for _, n in loaded]), dtype="float32")
    if out.shape[0] != len(loaded):
        sys.exit(f"embedder returned {out.shape[0]} vectors for {len(loaded)} clips")
    vectors.append(out)
    if i and i % (A.batch * 20) == 0:
        print(f"  {i}/{len(uuids)}", flush=True)
E = np.concatenate(vectors)
E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9

if A.save_embeddings:
    np.save(A.save_embeddings, E)

result = protocol.evaluate(E, manifest, index)
print()
print(" ".join(f"{c:>9}" for c in protocol.REPORTED))
print(" ".join(f"{result[c]:>9.3f}" for c in protocol.REPORTED))
if A.out:
    json.dump(result, open(A.out, "w"), indent=1)
    print(f"\n-> {A.out}")

# KyaraBench: A Benchmark for Character Identity Verification Across Performers

> **Joonyong Park, Jerry Li**

<a href="https://huggingface.co/spellbrush/kyaraembed"><img src="https://img.shields.io/badge/Model-spellbrush%2Fkyaraembed-FF9D00.svg" alt="Model" height="26"></a>

Two clips of one anime character in Japanese and in its English dub are two different people
speaking. Two characters voiced by the same actor are one person playing two parts. Character
identity and speaker identity come apart in both cases, and KyaraBench is built to measure that
separation: **85 human-audited character identities** drawn from aligned Japanese–English dub
pairs, with one condition for each point where the two can be confused, and a released encoder
trained for the task.

This repository is **metadata and code only**. It contains no audio; you reconstruct it from Anim-400K.
The metadata is anonymised: identities carry opaque labels (`CHAR_001`, `SHOW_45`, `VAJP_23`) in
place of show, character and voice-actor names.

---

## Repository Structure

```
kyarabench/
├── data/
│   ├── kyarabench_v1.0_manifest.json      # Identities, frozen trial lists, retrieval gallery
│   ├── kyarabench_v1.0_source_clips.json  # The 1,616 source clips, with checksums
│   └── two_speaker_repair.json            # The 95 clips holding a turn exchange
├── prepare/                               # Anim-400K -> restored audio, end to end
├── kyarabench/                            # Loader contract and scoring protocol
├── embedders/                             # Reference adapters for five published models
├── baselines/                             # A pitch-only row, as a shortcut check
└── score.py                               # Scores a system on every condition
```

---

## Benchmark Overview

### Construction

Built from [Anim-400K](https://huggingface.co/datasets/davidchan/anim400k), chosen because it
holds both cross-lingual dubbing and multi-character acting by a single voice actor. Its speaker
labels are episode-local and no characters are named, so identities are recovered and then
audited by listening.

| Stage | Identities | Utterances | Clips |
|---|---:|---:|---:|
| Anim-400K aligned dub pairs | — | 436,887 | 858,986 |
| Show-level clustering | 9,399 | — | — |
| Matched to a character portrait | 4,907 | — | — |
| ≥3 agreeing votes, cast-listed | 385 | — | — |
| Coverage and quality gates | 114 | 1,772 | 3,544 |
| **Human annotation (released)** | **85** | **986** | **1,616** |

An utterance is an aligned Japanese–English dub pair. Clips count distinct Japanese and English
audio files; the 1,616 released clips are 986 Japanese and 630 English.

### Conditions

Each list was drawn once with a fixed seed and ships as drawn, so every system is scored on
identical trials. Each identity contributes at most twelve positives per condition, and impostors
come at up to three per positive.

| Condition | Confusion tested | Trials (+/−) |
|---|---|---:|
| `Mono` | Same character, same language | 1778 / 5327 |
| `XLing` | Same character, dubbed performer | 646 / 1812 |
| `Mono-VA` | Different characters, shared voice actor | — / 309 |
| `XLing-K` | Four-clip bundles, unconstrained impostor | 636 / 1908 |
| `XLing-KP` | Four-clip bundles, impostor within two semitones of the target's pitch | 648 / 1944 |
| retrieval | Japanese bundle against a 53-way English gallery | 424 queries |

`Mono` pools the Japanese and English monolingual lists. `Mono-VA` supplies impostors only and is
scored against the `Mono` positives, so the move from `Mono` to `Mono-VA` is the price a system
pays when the voice stops being a clue. Retrieval reports R@1 averaged over the gallery
characters, against a chance rate of .019.

---

## Usage

### Reconstructing the audio

1. Obtain Anim-400K under its own terms:
   [huggingface.co/datasets/davidchan/anim400k](https://huggingface.co/datasets/davidchan/anim400k).
   Access requires signing the dataset's Terms of Use and NDA. This repository grants you no
   rights to the corpus and redistributes none of it.
2. Restore the 1,616 clips the benchmark uses with [Sidon](https://arxiv.org/abs/2509.17052):

   ```bash
   python prepare/restore.py --input-root anim400k_audio_clips --output-root restored \
       --clips-json data/kyarabench_v1.0_source_clips.json
   ```

   Each source file is checked against its recorded SHA-256 first. [`prepare/README.md`](prepare/README.md)
   walks the whole path, from the gated download to a scored run.

Restoration is not bit-deterministic: the decoder resynthesises the waveform rather than
filtering the input, and GPU kernels do not reduce in a fixed order, so two runs do not produce
identical audio.

### Scoring a system

```bash
pip install -r requirements.txt
python score.py --clips /path/to/restored --embedder my_system
```

A system is a Python module exposing one function:

```python
def embed(waveforms, lengths):
    """waveforms: list of mono 16 kHz float32 arrays, all the same length.
       lengths:   how many samples of each are real audio.
       returns:   (len(waveforms), dim) array."""
```

Scoring needs only `numpy`, `soundfile` and `librosa`. The loader hands every clip over at the
same length, so a batch needs no padding and `--batch` only trades memory for speed. Embeddings
are L2-normalised for you; a trial's similarity is the cosine between the two bundles' mean-pooled
embeddings.

Reference adapters for five published models live in `embedders/`, each naming its own model
package in its module docstring. The pitch-only row has its own entry point, because it is
compared by distance rather than cosine:

```bash
python baselines/f0_statistics.py --clips /path/to/restored
```

### KyaraEmbed

A 20.8M-parameter, 192-dimensional ECAPA-TDNN trained for this task, released at
[spellbrush/kyaraembed](https://huggingface.co/spellbrush/kyaraembed):

```bash
KYARAEMBED_CKPT=kyaraembed_v1.0.pt python score.py --clips /path/to/restored \
    --embedder embedders.kyaraembed
```

---

## License

| What | License | |
|---|---|---|
| Code: `score.py`, `kyarabench/`, `embedders/`, `baselines/` | MIT | `LICENSE` |
| Metadata: `data/` | CC BY 4.0 | `LICENSE-DATA` |
| The KyaraEmbed checkpoint | CC BY-NC 4.0 | `LICENSE-MODEL`, `NOTICE` |

The metadata license covers the manifests here and nothing else. It grants no rights over
Anim-400K or its audio, which stay under the dataset's own Terms of Use and NDA.

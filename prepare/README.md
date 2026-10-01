# From Anim-400K to a scored benchmark

Four steps. Only the first needs a person; the rest are scripts.

## 1. Obtain Anim-400K

Accept the gate at <https://huggingface.co/datasets/davidchan/anim400k>, which requires signing
the dataset's Terms of Use and NDA. Then pull the audio clips, which ship as six parts of one
archive:

```bash
huggingface-cli download davidchan/anim400k --repo-type dataset \
    --include "anim400k_audio_clips/*" --local-dir anim400k
```

The video clips and character pictures are not needed. Assemble and unpack:

```bash
cd anim400k/anim400k_audio_clips
cat anim400k_audio_clips.tar.gz.part-* > anim400k_audio_clips.tar.gz
tar xzf anim400k_audio_clips.tar.gz
```

That leaves `anim400k_audio_clips/<uuid[:2]>/<uuid>.mp3`, the layout `source_file` in
`data/kyarabench_v1.0_source_clips.json` refers to. Only 1,616 of those files are used.

## 2. Restore

Needs a CUDA GPU: Sidon ships its models as CUDA-traced TorchScript, so there is no CPU path.
If the installed torch does not match your driver, the script says so and names the fix.

```bash
pip install torch torchaudio transformers huggingface-hub soundfile numpy
python prepare/restore.py \
    --input-root anim400k/anim400k_audio_clips/anim400k_audio_clips \
    --output-root restored \
    --clips-json data/kyarabench_v1.0_source_clips.json
```

Restores exactly the 1,616 clips the benchmark uses, checking each source file against its
recorded SHA-256 first and skipping any that disagree. The Sidon weights
(`sarulab-speech/sidon-v0.1`) and the feature extractor (`facebook/w2v-bert-2.0`) download on
first run; neither is gated. Output goes to
`restored/<uuid[:2]>/<uuid>_restored.wav`, which is what the loader reads. Existing files are
skipped, so an interrupted run resumes.

Expect roughly an hour on one GPU. The script's docstring states the full signal chain.
`--num-shards`/`--shard-id` split the work across GPUs.

## 3. Score

```bash
pip install -r requirements.txt
python score.py --clips restored --embedder embedders.kyaraembed
python baselines/f0_statistics.py --clips restored
```

## Reproducibility

Restoration is not bit-deterministic. The decoder resynthesises the waveform from features rather
than filtering the input, and GPU kernels do not reduce in a fixed order, so two runs give audio
of the same length and level that still differs sample by sample.

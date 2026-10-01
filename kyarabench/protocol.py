"""Conditions, scoring and confidence intervals.

A trial holds two bundles of clips and asks whether they are the same character. Similarity is
the cosine between the two bundles' mean-pooled embeddings, and each condition is summarised by
its equal error rate. Intervals resample characters rather than trials, because trials drawn
from one character are not independent of each other.

Mono is the Japanese and English monolingual lists scored together. Mono-VA reuses those
positives against impostors who share the target's voice actor. Both are assembled here, so a
condition name means one thing.
"""
import numpy as np

# condition -> (lists contributing positives, list contributing impostors or None)
CONDITIONS = {
    "Mono":     (["MONO", "MONO_EN"], None),
    "XLing":    (["XLING"], None),
    "Mono-VA":  (["MONO", "MONO_EN"], "MONO_VA"),
    "XLing-K":  (["XLING_K"], None),
    "XLing-KP": (["XLING_KP"], None),
}
REPORTED = list(CONDITIONS) + ["R@1"]


def _bundle(x):
    return x if isinstance(x, list) else [x]


def build_trials(manifest, available=None):
    """Resolve each condition into (a_rows, b_rows, label, char_a, char_b) tuples.

    `available` maps a clip uuid to its row in the embedding matrix. Clips absent from it are
    skipped, and a trial is dropped once either side has nothing left.
    """
    index = available if available is not None else {}
    trials = manifest["trials"]
    out = {}
    for name, (pos_lists, neg_list) in CONDITIONS.items():
        if neg_list:
            sources = [(trials[k], 1) for k in pos_lists] + [(trials[neg_list], 0)]
        else:
            sources = [(trials[k], None) for k in pos_lists]
        rows = []
        for lst, want in sources:
            for t in lst:
                if want is not None and t["label"] != want:
                    continue
                a = [index[u] for u in _bundle(t["a"]) if u in index]
                b = [index[u] for u in _bundle(t["b"]) if u in index]
                if not a or not b:
                    continue
                rows.append((a, b, t["label"], t["char"], t.get("char_b", t["char"])))
        out[name] = rows
    return out


def pool(embeddings, rows):
    """Mean-pool a bundle of clip embeddings and renormalise to the unit sphere."""
    v = embeddings[rows].mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def scores(embeddings, trials):
    return np.array([float(pool(embeddings, a) @ pool(embeddings, b)) for a, b, *_ in trials])


def labels(trials):
    return np.array([t[2] for t in trials])


def eer(score, label, weight=None):
    """Equal error rate at the threshold where the two error rates are closest.

    Weights carry the bootstrap: a resampled character multiplies the weight of every trial it
    appears on, and a weight of zero drops the trial.
    """
    weight = np.ones(len(score)) if weight is None else weight
    keep = weight > 0
    score, label, weight = score[keep], label[keep], weight[keep]
    order = np.argsort(-score)
    label, weight = label[order], weight[order]
    positives = (weight * (label == 1)).sum()
    negatives = (weight * (label == 0)).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    false_negative = 1 - np.cumsum(weight * (label == 1)) / positives
    false_positive = np.cumsum(weight * (label == 0)) / negatives
    i = int(np.argmin(np.abs(false_positive - false_negative)))
    return float((false_positive[i] + false_negative[i]) / 2)


def retrieval_hits(embeddings, manifest, index):
    """Per character, whether each of its Japanese query bundles retrieves its English side."""
    gallery_keys = sorted(manifest["xling_retrieval"]["gallery"])
    gallery = np.stack([
        pool(embeddings, [index[u] for u in manifest["xling_retrieval"]["gallery"][k] if u in index])
        for k in gallery_keys
    ])
    per_character = {}
    for q in manifest["xling_retrieval"]["queries"]:
        rows = [index[u] for u in q["q"] if u in index]
        if not rows:
            continue
        predicted = gallery_keys[int(np.argmax(gallery @ pool(embeddings, rows)))]
        per_character.setdefault(q["char"], []).append(float(predicted == q["char"]))
    return per_character


def retrieval_r1(embeddings, manifest, index):
    """R@1 averaged over characters, so each weighs the same however many bundles it supplies."""
    per_character = retrieval_hits(embeddings, manifest, index)
    return sum(sum(v) / len(v) for v in per_character.values()) / len(per_character)


def evaluate(embeddings, manifest, index):
    """Every reported number for one system."""
    trials = build_trials(manifest, index)
    result = {name: eer(scores(embeddings, rows), labels(rows)) for name, rows in trials.items()}
    result["R@1"] = retrieval_r1(embeddings, manifest, index)
    return result


def bootstrap(embeddings_a, embeddings_b, manifest, index, condition, rounds=600, seed=17):
    """Paired character bootstrap on the EER difference between two systems.

    A trial's weight is the product of how often its two characters were drawn. Returns the
    2.5th and 97.5th percentile of the difference.
    """
    trials = build_trials(manifest, index)[condition]
    label = labels(trials)
    characters = sorted({c for t in trials for c in (t[3], t[4])})
    slot = {c: i for i, c in enumerate(characters)}
    left = np.array([slot[t[3]] for t in trials])
    right = np.array([slot[t[4]] for t in trials])
    score_a = scores(embeddings_a, trials)
    score_b = scores(embeddings_b, trials)

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(rounds):
        drawn = np.bincount(rng.integers(0, len(characters), len(characters)),
                            minlength=len(characters)).astype(float)
        d = eer(score_a, label, drawn[left] * drawn[right]) - \
            eer(score_b, label, drawn[left] * drawn[right])
        if not np.isnan(d):
            deltas.append(d)
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_r1(embeddings_a, embeddings_b, manifest, index, rounds=600, seed=17):
    """Paired character bootstrap on the R@1 difference between two systems.

    Characters are resampled and each contributes its own retrieval accuracy. The gallery is
    held fixed, since resampling it would change the task rather than estimate its uncertainty.
    """
    hits_a = retrieval_hits(embeddings_a, manifest, index)
    hits_b = retrieval_hits(embeddings_b, manifest, index)
    characters = sorted(hits_a)
    acc_a = np.array([sum(hits_a[c]) / len(hits_a[c]) for c in characters])
    acc_b = np.array([sum(hits_b[c]) / len(hits_b[c]) for c in characters])

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(rounds):
        drawn = np.bincount(rng.integers(0, len(characters), len(characters)),
                            minlength=len(characters)).astype(float)
        deltas.append(float((drawn * acc_a).sum() / drawn.sum() - (drawn * acc_b).sum() / drawn.sum()))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)

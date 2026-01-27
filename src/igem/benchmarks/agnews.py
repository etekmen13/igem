import random
import re
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.benchmarks.utils import AvalancheDataset

from ..models.gpt2 import get_tokenizer

CHOICES = ["World", "Sports", "Business", "Sci/Tech"]

# Seed used to build and split the experiences. Held fixed across runs so every
# seed sees the same partition of the corpus; only the order of experiences
# varies per run seed (Sec. IV-A).
SPLIT_SEED = 12345

# Topic terms used to bias each experience towards a corner of a label, so the
# drift is lexical as well as prior-driven.
_BUCKETS = {
    0: [["parliament", "embassy"], ["resolution", "peace"], ["treaty", "border"]],
    1: [["soccer", "league"], ["basketball", "nba"], ["tennis", "open"]],
    2: [["stocks", "market"], ["earnings", "quarter"], ["merger", "deal"]],
    3: [["computer", "software"], ["space", "nasa"], ["biology", "genome"]],
}

# Class priors per experience. Experience k concentrates on one label; the
# ordering follows the example in Sec. VI (Sports first, then Sci/Tech).
_PRIORS = [
    [0.1, 0.7, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.7],
    [0.1, 0.1, 0.7, 0.1],
    [0.7, 0.1, 0.1, 0.1],
]


def _matches_any(text, terms):
    return any(re.search(rf"\b{re.escape(t)}\b", text, flags=re.I) for t in terms)


class AGNewsMCDataset(Dataset):
    """Scores each item as four (headline, label-word) continuations.

    Stems are tokenised once up front; the padded length is the longest
    sequence actually present, capped at ``max_len``.
    """

    def __init__(self, rows, tokenizer, max_len=512):
        self.labels = [r["label"] for r in rows]
        self.pad_id = tokenizer.pad_token_id

        self.stems = tokenizer([r["text"] for r in rows], add_special_tokens=False).input_ids
        self.options = [
            tokenizer(f" {c}", add_special_tokens=False).input_ids for c in CHOICES
        ]

        longest_opt = max(len(o) for o in self.options)
        longest = max((len(s) for s in self.stems), default=0) + longest_opt
        self.max_len = max(min(max_len, longest), longest_opt + 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        ids, mask, loss_mask = [], [], []

        for opt in self.options:
            allowed = self.max_len - len(opt)
            trimmed = stem[-allowed:] if len(stem) > allowed else stem

            full = trimmed + opt
            pad = self.max_len - len(full)

            ids.append(torch.tensor([self.pad_id] * pad + full, dtype=torch.long))
            mask.append(torch.tensor([0] * pad + [1] * len(full), dtype=torch.long))
            loss_mask.append(
                torch.tensor(
                    [0] * pad + [0] * len(trimmed) + [1] * len(opt), dtype=torch.long
                )
            )

        # (option, channel, length): the scoring wrapper indexes channels on dim 1
        x = torch.stack(
            [torch.stack(ids), torch.stack(mask), torch.stack(loss_mask)], dim=1
        )
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        t = torch.tensor(0, dtype=torch.long)
        return x, y, t


def _draw(available, terms, count):
    """Take `count` rows, preferring topical matches, and return the remainder.

    Rows are removed from `available` so experiences never share a row.
    """
    matched, rest = [], []
    for row in available:
        (matched if _matches_any(row["text"], terms) else rest).append(row)

    n_matched = min(count, len(matched))
    n_rest = count - n_matched

    taken = matched[:n_matched] + rest[:n_rest]
    leftover = matched[n_matched:] + rest[n_rest:]
    return taken, leftover


def make_agnews_benchmark(n_experiences: int, seed: int, exp_size: int = 2000,
                          max_len: int = 512):
    """Domain-incremental AG News with prior drift across experiences."""
    if n_experiences > len(_PRIORS):
        raise ValueError(f"at most {len(_PRIORS)} experiences are defined")

    tokenizer = get_tokenizer()

    ds = load_dataset("ag_news")
    by_label = defaultdict(list)
    for split in ("train", "test"):
        for item in ds[split]:
            by_label[item["label"]].append({"text": item["text"], "label": item["label"]})

    split_rng = random.Random(SPLIT_SEED)
    for label in by_label:
        split_rng.shuffle(by_label[label])

    train_sets, test_sets = [], []

    for eid, prior in enumerate(_PRIORS[:n_experiences]):
        rows = []
        for label, p in enumerate(prior):
            terms = _BUCKETS[label][eid % len(_BUCKETS[label])]
            taken, by_label[label] = _draw(by_label[label], terms, round(exp_size * p))
            rows.extend(taken)

        split_rng.shuffle(rows)
        cut = int(0.8 * len(rows))

        train_sets.append(
            AvalancheDataset(AGNewsMCDataset(rows[:cut], tokenizer, max_len))
        )
        test_sets.append(
            AvalancheDataset(AGNewsMCDataset(rows[cut:], tokenizer, max_len))
        )

    # the partition is fixed; the order the model meets it in is not
    order = list(range(n_experiences))
    random.Random(seed).shuffle(order)

    return benchmark_from_datasets(
        train=[train_sets[i] for i in order],
        test=[test_sets[i] for i in order],
    )

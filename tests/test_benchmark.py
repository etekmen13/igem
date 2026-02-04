"""Sec. IV-A requires experiences to be built from disjoint rows, so that is
checked directly rather than trusted."""

import random

import pytest
import torch

from igem.benchmarks.agnews import _BUCKETS, _PRIORS, _draw


def rows(n, text="filler copy"):
    return [{"text": f"{text} {i}", "label": 0} for i in range(n)]


def test_draw_returns_requested_count():
    taken, left = _draw(rows(100), ["soccer"], 30)
    assert len(taken) == 30
    assert len(left) == 70


def test_draw_partitions_without_loss_or_overlap():
    pool = rows(100)
    taken, left = _draw(pool, ["soccer"], 30)

    ids = [id(r) for r in pool]
    assert sorted(map(id, taken + left)) == sorted(ids)
    assert not (set(map(id, taken)) & set(map(id, left)))


def test_draw_prefers_topical_matches():
    pool = rows(50) + [{"text": f"a soccer league match {i}", "label": 1} for i in range(10)]
    random.Random(0).shuffle(pool)

    taken, _ = _draw(pool, ["soccer", "league"], 10)
    assert all("soccer" in r["text"] for r in taken)


def test_draw_falls_back_when_matches_run_out():
    pool = rows(50) + [{"text": "a soccer match", "label": 1}]
    taken, left = _draw(pool, ["soccer"], 10)

    assert len(taken) == 10
    assert sum("soccer" in r["text"] for r in taken) == 1
    assert len(left) == 41


def test_repeated_draws_never_reuse_a_row():
    """This is the property the leaky version violated."""
    available = rows(600)
    seen, drawn = set(), []

    for eid in range(3):
        taken, available = _draw(available, _BUCKETS[0][eid], 100)
        drawn.append(taken)
        for r in taken:
            assert id(r) not in seen, "row reused across experiences"
            seen.add(id(r))

    assert len(seen) == 300
    assert len(available) == 300


def test_priors_are_distributions():
    for prior in _PRIORS:
        assert len(prior) == 4
        assert abs(sum(prior) - 1.0) < 1e-9


class FakeTokenizer:
    """Mimics GPT2Tokenizer: a flat id list for one string, nested for a list."""

    pad_token_id = 0

    def __call__(self, texts, add_special_tokens=False):
        encode = lambda s: list(range(1, len(s.split()) + 1))
        ids = encode(texts) if isinstance(texts, str) else [encode(t) for t in texts]
        return type("Encoding", (), {"input_ids": ids})()


def test_fake_tokenizer_matches_the_real_shape_contract():
    tok = FakeTokenizer()
    assert isinstance(tok("one two").input_ids[0], int)
    assert isinstance(tok(["one", "two three"]).input_ids[0], list)
    assert len(tok("a b c").input_ids) == 3


def test_item_layout_matches_the_scoring_wrapper():
    """x is (option, channel, length). LMScoringWrapper slices channels on dim 1
    of the batched tensor, so a transposed layout silently drops an option."""
    from igem.benchmarks.agnews import CHOICES, AGNewsMCDataset

    ds = AGNewsMCDataset(rows(4), FakeTokenizer(), max_len=16)
    x, y, t = ds[0]

    assert x.shape == (len(CHOICES), 3, ds.max_len)
    assert x[:, 0, :].shape == (len(CHOICES), ds.max_len)
    assert y.shape == () and t.shape == ()


def test_batched_layout_gives_one_score_per_choice():
    from igem.benchmarks.agnews import CHOICES, AGNewsMCDataset

    ds = AGNewsMCDataset(rows(8), FakeTokenizer(), max_len=16)
    batch = torch.stack([ds[i][0] for i in range(4)])

    B, O, C, L = batch.shape
    assert (B, O, C) == (4, len(CHOICES), 3)
    assert batch[:, :, 0, :].shape == (4, len(CHOICES), L)


def test_loss_mask_covers_only_the_option_tokens():
    from igem.benchmarks.agnews import AGNewsMCDataset

    mixed = [{"text": "one two three four five six", "label": 0},
             {"text": "short one", "label": 1}]
    ds = AGNewsMCDataset(mixed, FakeTokenizer(), max_len=64)
    n_opt = len(ds.options[0])

    x, _, _ = ds[1]  # the short stem, so padding is required
    ids, attn, loss = x[:, 0, :], x[:, 1, :], x[:, 2, :]

    assert (loss <= attn).all(), "loss mask must lie inside the attention mask"
    assert (loss.sum(dim=1) == n_opt).all(), "only the option tokens are scored"
    assert (loss[:, -n_opt:] == 1).all(), "the option sits at the end"
    assert (ids[:, 0] == 0).all() and (attn[:, 0] == 0).all(), "padding is on the left"


def test_long_stems_are_truncated_from_the_left():
    """The tail of a headline is what the option continues, so the front goes."""
    from igem.benchmarks.agnews import AGNewsMCDataset

    long_row = [{"text": " ".join(str(i) for i in range(40)), "label": 0}]
    ds = AGNewsMCDataset(long_row, FakeTokenizer(), max_len=12)
    x, _, _ = ds[0]

    ids, attn = x[:, 0, :], x[:, 1, :]
    assert ids.shape[1] == 12
    assert (attn == 1).all(), "a truncated stem leaves no padding"
    assert ids[0, 0] > 1, "kept the tail of the stem, not the head"

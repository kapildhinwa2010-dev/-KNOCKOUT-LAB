import numpy as np
import pytest

from knockout_lab.lab import fragile_blocks, knockout_rank, run_stream

D = 4
CONST = np.zeros(D)
CONST[0] = 5.0


def make_blocks():
    return [
        lambda y: y * 2.0,            # doubles the stream
        lambda y: np.zeros(D),        # writes nothing
        lambda y: CONST,             # constant contribution
    ]


def score_fn(out):
    return float(out[0])


def test_run_stream():
    blocks = make_blocks()
    x = np.zeros(D)
    x[0] = 1.0
    out = run_stream(blocks, x)
    # x + 2x + 0 + 5 = [8, 0, 0, 0]
    assert np.allclose(out, [8.0, 0.0, 0.0, 0.0])


def test_knockout_ranking():
    blocks = make_blocks()
    x = np.zeros(D)
    x[0] = 1.0
    r = knockout_rank(blocks, x, score_fn)
    assert r["baseline_score"] == pytest.approx(8.0)
    by_index = {row["index"]: row for row in r["ranking"]}
    # removing the constant block costs 5, the doubler costs 2, the zero block costs 0
    assert by_index[2]["damage"] == pytest.approx(5.0)
    assert by_index[0]["damage"] == pytest.approx(2.0)
    assert by_index[1]["damage"] == pytest.approx(0.0)
    assert [row["index"] for row in r["ranking"]] == [2, 0, 1]
    assert r["ranking"][0]["rank"] == 1
    assert by_index[2]["importance"] == pytest.approx(5.0 / 8.0)


def test_fragile_blocks():
    blocks = make_blocks()
    x = np.zeros(D)
    x[0] = 1.0
    r = knockout_rank(blocks, x, score_fn)
    assert fragile_blocks(r, fraction=0.5) == [2]
    assert fragile_blocks(r, fraction=0.1) == [2, 0]
    assert fragile_blocks(r, fraction=0.0) == [2, 0]


def test_negative_damage_means_parasite():
    blocks = [
        lambda y: CONST,
        lambda y: -np.ones(D),  # removing this block helps
    ]
    x = np.zeros(D)
    r = knockout_rank(blocks, x, score_fn)
    by_index = {row["index"]: row for row in r["ranking"]}
    assert by_index[1]["damage"] == pytest.approx(4.0)
    # removing the constant block is worse than removing the parasite
    assert [row["index"] for row in r["ranking"]] == [1, 0]


def test_empty_blocks_error():
    with pytest.raises(ValueError):
        knockout_rank([], np.zeros(D), score_fn)
    with pytest.raises(ValueError):
        fragile_blocks({"not": "a report"})
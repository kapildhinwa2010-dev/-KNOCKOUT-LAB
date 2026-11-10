import numpy as np
import pytest

from knockout_lab.lab import fragile_blocks, knockout_rank, run_stream

D = 4
CONST = np.zeros(D)
CONST[0] = 5.0


def make_blocks():
    return [
        lambda y: y * 2.0,
        lambda y: np.zeros(D),
        lambda y: CONST,
    ]


def score_fn(out):
    return float(out[0])


def test_run_stream():
    blocks = make_blocks()
    x = np.zeros(D)
    x[0] = 1.0
    out = run_stream(blocks, x)
    assert np.allclose(out, [8.0, 0.0, 0.0, 0.0])


def test_knockout_ranking():
    blocks = make_blocks()
    x = np.zeros(D)
    x[0] = 1.0
    r = knockout_rank(blocks, x, score_fn)
    assert r["baseline_score"] == pytest.approx(8.0)
    by_index = {row["index"]: row for row in r["ranking"]}
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
    assert fragile_blocks(r, fraction=0.0) == [2, 0, 1]


def test_negative_damage_means_parasite():
    blocks = [
        lambda y: CONST,
        lambda y: -np.ones(D),
    ]
    x = np.zeros(D)
    r = knockout_rank(blocks, x, score_fn)
    assert r["baseline_score"] == pytest.approx(4.0)
    by_index = {row["index"]: row for row in r["ranking"]}
    assert by_index[0]["damage"] == pytest.approx(5.0)
    assert by_index[1]["damage"] == pytest.approx(-1.0)
    assert [row["index"] for row in r["ranking"]] == [0, 1]


def test_empty_blocks_error():
    with pytest.raises(ValueError):
        knockout_rank([], np.zeros(D), score_fn)
    with pytest.raises(ValueError):
        fragile_blocks({"not": "a report"})
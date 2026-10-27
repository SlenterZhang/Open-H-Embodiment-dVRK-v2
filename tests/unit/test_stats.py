"""Dataset statistics and aggregation."""

import numpy as np

from dvrk_lerobot.dataset.stats import aggregate, numeric_stats, to_jsonable


def test_numeric_stats_shapes() -> None:
    stats = numeric_stats(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert stats["min"].tolist() == [1.0, 2.0]
    assert stats["max"].tolist() == [3.0, 4.0]
    assert stats["mean"].tolist() == [2.0, 3.0]
    assert stats["count"].tolist() == [2]

    scalar = numeric_stats(np.array([1.0, 2.0, 3.0]))
    assert scalar["mean"].reshape(-1).tolist() == [2.0]
    assert scalar["count"].tolist() == [3]


def test_aggregate_matches_concatenated_data() -> None:
    first = np.array([[1.0, 10.0], [2.0, 20.0]])
    second = np.array([[3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
    combined = aggregate(
        [
            {"state": numeric_stats(first)},
            {"state": numeric_stats(second)},
        ]
    )["state"]
    expected = np.vstack((first, second))
    assert np.allclose(combined["mean"], expected.mean(axis=0))
    assert np.allclose(combined["std"], expected.std(axis=0))
    assert combined["count"].tolist() == [5]


def test_aggregate_keeps_single_count_shape() -> None:
    combined = aggregate(
        [
            {"x": numeric_stats(np.array([1.0, 2.0]))},
            {"x": numeric_stats(np.array([3.0]))},
        ]
    )
    assert combined["x"]["count"].shape == (1,)


def test_to_jsonable_rejects_non_finite_values() -> None:
    stats = {"x": numeric_stats(np.array([1.0, 2.0]))}
    assert to_jsonable(stats)["x"]["count"] == [2]
    stats["x"]["mean"][0] = np.nan
    try:
        to_jsonable(stats)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite stats were accepted")

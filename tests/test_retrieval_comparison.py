import pytest

from llm_benchmark.retrieval.comparison import compare_hit_outcomes


def test_all_fout_outcome_groups():
    baseline = {
        "q1": True,
        "q2": True,
        "q3": False,
        "q4": False,
    }
    candidate = {
        "q1": True,
        "q2": False,
        "q3": True,
        "q4": False,
    }

    report = compare_hit_outcomes(baseline, candidate)

    assert report["query_count"] == 4
    assert report["counts"] == {
        "both_hit": 1,
        "baseline_only": 1,
        "candidate_only": 1,
        "both_miss": 1,
    }
    assert report["query_ids"] == {
        "both_hit": ["q1"],
        "baseline_only": ["q2"],
        "candidate_only": ["q3"],
        "both_miss": ["q4"],
    }
    assert report["baseline_hit_rate"] == 0.5
    assert report["candidate_hit_rate"] == 0.5
    assert report["hit_rate_delta"] == 0.0
    assert report["candidate_union_hit_rate"] == 0.75


def test_positive_hit_rate_delta():
    report = compare_hit_outcomes(
        {"q1": False, "q2": True},
        {"q1": True, "q2": True},
    )

    assert report["hit_rate_delta"] == 0.5


def test_input_order_does_not_change_report():
    first = compare_hit_outcomes(
        {"q2": True, "q1": True},
        {"q1": True, "q2": True},
    )
    second = compare_hit_outcomes(
        {"q1": True, "q2": True},
        {"q2": True, "q1": True},
    )

    assert first == second
    assert first["query_ids"]["both_hit"] == ["q1", "q2"]


def test_mismatched_query_ids_are_rejected():
    with pytest.raises(ValueError, match="same query IDs"):
        compare_hit_outcomes({"q1": True}, {"q2": True})


@pytest.mark.parametrize(
    "outcomes",
    [
        {},
        {"": True},
        {" ": True},
        {1: True},
        {"q1": 1},
        {"q1": 0},
        {"q1": None},
        {"q1": "true"},
    ],
)
def test_invalid_outcomes_are_rejected(outcomes):
    with pytest.raises(ValueError):
        compare_hit_outcomes(outcomes, outcomes)
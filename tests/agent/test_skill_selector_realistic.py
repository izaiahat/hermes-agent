from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
CORPUS_PATH = FIXTURE_DIR / "selector_active_skills_2026-04-26.json"
CASES_PATH = FIXTURE_DIR / "selector_live_eval_cases_2026-04-26.json"
REFERENCE_PATH = FIXTURE_DIR / "selector_live_eval_reference_2026-04-26.json"

MIN_PRECISION = 0.80
MIN_RECALL = 0.90
MAX_CASES = 50
MIN_CASES = 30
MAX_RETURNED_SKILLS = 3
MAX_UNIVERSAL_LOAD_RATE = 0.40


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compute_metrics(reference: dict) -> dict:
    tp = fp = fn = 0
    load_counter: Counter[str] = Counter()
    for row in reference["results"]:
        expected = set(row["expected"])
        loaded = set(row["loaded"])
        tp += len(expected & loaded)
        fp += len(loaded - expected)
        fn += len(expected - loaded)
        for skill in loaded:
            load_counter[skill] += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "load_counter": load_counter,
    }


def test_fixture_counts_match_realistic_selector_audit_scope() -> None:
    corpus = _load_json(CORPUS_PATH)
    cases = _load_json(CASES_PATH)
    assert len(corpus) == 237
    assert MIN_CASES <= len(cases) <= MAX_CASES


def test_recorded_live_selector_scores_meet_thresholds() -> None:
    reference = _load_json(REFERENCE_PATH)
    metrics = _compute_metrics(reference)
    assert metrics["precision"] >= MIN_PRECISION, metrics
    assert metrics["recall"] >= MIN_RECALL, metrics


def test_no_case_in_reference_loads_more_than_three_skills() -> None:
    reference = _load_json(REFERENCE_PATH)
    for row in reference["results"]:
        assert len(row["loaded"]) <= MAX_RETURNED_SKILLS, row


def test_no_skill_behaves_like_a_universal_trigger_in_reference() -> None:
    reference = _load_json(REFERENCE_PATH)
    metrics = _compute_metrics(reference)
    case_count = len(reference["results"])
    if not metrics["load_counter"]:
        return
    most_common_skill, count = metrics["load_counter"].most_common(1)[0]
    assert (count / case_count) <= MAX_UNIVERSAL_LOAD_RATE, {
        "skill": most_common_skill,
        "count": count,
        "case_count": case_count,
    }


def test_recorded_live_selector_has_no_false_negatives_on_curated_cases() -> None:
    reference = _load_json(REFERENCE_PATH)
    failures = [row for row in reference["results"] if set(row["expected"]) - set(row["loaded"])]
    assert failures == [], failures

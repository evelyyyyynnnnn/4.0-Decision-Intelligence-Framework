"""Tests for the real MIMIC-IV triage track.

These skip cleanly when the PhysioNet demo cache is not present, so the suite
still passes in an environment without the data. When it IS present they check
the honesty-critical properties: a subject-level split with no leakage, finite
and reasonable calibration, and a fitted policy that beats the worst constant
floor on realised cost.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import real_tasks as rt

pytestmark = pytest.mark.skipif(
    not rt.available(),
    reason="PhysioNet MIMIC-IV demo cache not present; set MIMIC_DATA_ROOT")


@pytest.fixture(scope="module")
def sample():
    s, _ = rt.build_sample()
    return s


def test_sample_is_real_and_well_formed(sample):
    assert len(sample) > 200
    assert sample.X.shape[0] == sample.y.shape[0] == sample.subject.shape[0]
    assert sample.X.shape[1] == len(sample.feature_names)
    assert np.isfinite(sample.X).all()
    assert set(np.unique(sample.y).tolist()) <= {0, 1}
    assert 0.0 < sample.y.mean() < 1.0        # the label is non-degenerate


def test_subject_split_has_no_patient_in_both_sides(sample):
    tr, te = rt.subject_split(sample, frac=0.5, seed=0)
    train_subs = set(sample.subject[tr].tolist())
    test_subs = set(sample.subject[te].tolist())
    assert train_subs and test_subs
    assert train_subs.isdisjoint(test_subs)   # no leakage across the split


def test_calibrated_policy_is_finite_and_reasonably_calibrated(sample):
    from src.metrics import calibration
    tr, te = rt.subject_split(sample, frac=0.5, seed=0)
    train, test = rt._sub(sample, tr), rt._sub(sample, te)
    proba, _ = rt.fit_calibrated_logistic(train)
    p = proba(test.X)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    ece = calibration(p, test.y.astype(float))["ece"]
    assert 0.0 <= ece < 0.25                   # loose: a demo cohort, not a study


def test_fitted_policy_beats_the_worst_constant_floor(sample):
    tr, te = rt.subject_split(sample, frac=0.5, seed=0)
    train, test = rt._sub(sample, tr), rt._sub(sample, te)
    _, decide = rt.fit_calibrated_logistic(train)
    fitted_cost = rt.realised_cost(decide(test.X), test.y)
    never = rt.realised_cost(np.zeros(len(test), int), test.y)
    always = rt.realised_cost(np.ones(len(test), int), test.y)
    assert fitted_cost < max(never, always)


def test_fitted_policy_is_near_the_hindsight_floor(sample):
    """It cannot beat the in-hindsight best threshold; it should be close."""
    tr, te = rt.subject_split(sample, frac=0.5, seed=0)
    train, test = rt._sub(sample, tr), rt._sub(sample, te)
    proba, decide = rt.fit_calibrated_logistic(train)
    fitted_cost = rt.realised_cost(decide(test.X), test.y)
    floor = rt.best_hindsight_threshold_cost(proba(test.X), test.y)["cost"]
    assert fitted_cost >= floor - 1e-9         # floor is a lower bound
    assert fitted_cost - floor < 0.15          # and the policy is close to it

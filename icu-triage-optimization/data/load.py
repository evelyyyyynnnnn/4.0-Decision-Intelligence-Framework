"""Build a real MIMIC-IV cohort for the triage frontier.

This project keeps its own copy of the data rather than reaching into the
sibling ICU project's cache. The two are meant to be independently copyable,
and a project that silently reads another repository's download directory is
not. Only the modelling code is borrowed, by file path, under its own module
namespace.
"""
from __future__ import annotations

import pathlib

import numpy as np

from .mimicvitals import EVENT_DEFINITIONS, events_from_vitals, stay_series

ROOT = pathlib.Path(__file__).resolve().parent


def load_patients(cohort_module, root=ROOT, min_hours: float = 12.0):
    """Return (patients, provenance) as the sibling's Patient objects."""
    stays, prov = stay_series(root, min_hours=min_hours)
    patients = [
        cohort_module.Patient(
            pid=s["stay_id"], times=s["times"], age=s["age"],
            baseline_map=float(np.median(s["vitals"]["map_mmhg"])),
            vitals=s["vitals"], events=events_from_vitals(s["vitals"]))
        for s in stays
    ]
    prov = dict(prov)
    prov["event_definitions"] = EVENT_DEFINITIONS
    return patients, prov

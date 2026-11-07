# Method — Multi-Objective ICU Triage Optimization

> **Not yet written.** This file must state the method before the project can
> be cited as evidence.

## Problem

Multi-objective ICU triage and alert-threshold optimization.

## Why the approach is non-obvious

Triage policy and alert threshold are normally optimised separately, but the threshold determines the arrival process the triage policy then faces. Optimising either one alone yields a policy that degrades as soon as the other moves, so the two have to be solved together to mean anything.

## Method

_State the approach, the modelling choices, and the assumptions. A reviewer with
domain knowledge but no context should be able to follow it end to end._

## Baseline

_Name the baseline this is measured against. Without one, the result is a number
with nothing to compare it to._

Suggested baseline sources: 2.0-Healthcare-Ai-Systems — icu-early-warning

## Evaluation protocol

_State the split, the held-out set, and how out-of-sample performance is
measured. Report calibration and uncertainty where the domain calls for it._

## Reproduction

```
# commands that take a reader from a clean checkout to the reported numbers
```

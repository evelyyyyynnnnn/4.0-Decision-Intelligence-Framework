# Method — Optimization-Under-Uncertainty Library

> **Not yet written.** This file must state the method before the project can
> be cited as evidence.

## Problem

Stochastic and robust optimization applied to portfolio allocation and hospital resource allocation.

## Why the approach is non-obvious

The usual comparison reports each formulation's objective value under its own distribution, which flatters whichever one generated the scenarios. The informative test is cross-evaluation: score the stochastic solution under the robust uncertainty set and the robust solution under the sampled distribution, and report both.

## Method

_State the approach, the modelling choices, and the assumptions. A reviewer with
domain knowledge but no context should be able to follow it end to end._

## Baseline

_Name the baseline this is measured against. Without one, the result is a number
with nothing to compare it to._

Suggested baseline sources: to be identified

## Evaluation protocol

_State the split, the held-out set, and how out-of-sample performance is
measured. Report calibration and uncertainty where the domain calls for it._

## Reproduction

```
# commands that take a reader from a clean checkout to the reported numbers
```

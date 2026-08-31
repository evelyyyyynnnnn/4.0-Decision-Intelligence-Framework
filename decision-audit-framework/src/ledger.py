"""A decision ledger you can replay.

The question this exists to answer is the one asked after a decision goes wrong:
*why did the system do that, and what would it have done if the inputs had been
different?* Answering it needs three things that ordinary logging does not
provide -- the inputs as they were at the time, the policy version that ran, and
a guarantee that neither has been edited since.

Replay is exact rather than approximate: the policy is a pure function of the
recorded inputs, so re-running it reproduces the decision bit for bit. That
constraint is what makes the whole thing auditable, and it is a real restriction
on how policies may be written.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

GENESIS = "0" * 64


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _digest(prev: str, payload: dict) -> str:
    return hashlib.sha256((prev + _canon(payload)).encode()).hexdigest()


@dataclass
class Decision:
    seq: int
    case_id: str
    inputs: dict
    action: object
    score: float
    policy: str
    policy_version: str
    ts: float = field(default_factory=time.time)
    prev_hash: str = GENESIS
    hash: str = ""
    context: dict = field(default_factory=dict)

    def payload(self) -> dict:
        return {"seq": self.seq, "case_id": self.case_id, "inputs": self.inputs,
                "action": self.action, "score": round(float(self.score), 9),
                "policy": self.policy, "policy_version": self.policy_version,
                "context": self.context, "ts": round(self.ts, 6)}


@dataclass
class Policy:
    """A named, versioned pure function from inputs to (action, score)."""
    name: str
    version: str
    fn: Callable

    def __call__(self, inputs: dict):
        return self.fn(inputs)

    def fingerprint(self) -> str:
        return hashlib.sha256(f"{self.name}@{self.version}".encode()).hexdigest()[:12]


class DecisionLedger:
    def __init__(self):
        self.records: list = []
        self.policies: dict = {}

    def register(self, policy: Policy) -> None:
        self.policies[f"{policy.name}@{policy.version}"] = policy

    def decide(self, case_id: str, inputs: dict, policy: Policy,
               context: dict | None = None) -> Decision:
        action, score = policy(inputs)
        self.register(policy)
        prev = self.records[-1].hash if self.records else GENESIS
        rec = Decision(seq=len(self.records), case_id=case_id, inputs=dict(inputs),
                       action=action, score=float(score), policy=policy.name,
                       policy_version=policy.version, prev_hash=prev,
                       context=dict(context or {}))
        rec.hash = _digest(prev, rec.payload())
        self.records.append(rec)
        return rec

    # --- integrity --------------------------------------------------------

    def verify(self) -> tuple:
        prev = GENESIS
        for i, r in enumerate(self.records):
            if r.prev_hash != prev or r.hash != _digest(prev, r.payload()):
                return False, i
            prev = r.hash
        return True, -1

    # --- replay -----------------------------------------------------------

    def replay(self, seq: int) -> dict:
        """Re-run the recorded decision and check it still reproduces."""
        rec = self.records[seq]
        key = f"{rec.policy}@{rec.policy_version}"
        if key not in self.policies:
            return {"seq": seq, "reproduced": False,
                    "reason": f"policy {key} not registered"}
        action, score = self.policies[key](rec.inputs)
        same = action == rec.action and abs(float(score) - rec.score) < 1e-9
        return {"seq": seq, "reproduced": same,
                "recorded_action": rec.action, "replayed_action": action,
                "recorded_score": round(rec.score, 6),
                "replayed_score": round(float(score), 6),
                "reason": "" if same else "policy output changed"}

    def replay_all(self) -> dict:
        outs = [self.replay(i) for i in range(len(self.records))]
        bad = [o for o in outs if not o["reproduced"]]
        return {"n": len(outs), "reproduced": len(outs) - len(bad),
                "failures": bad[:10]}

    # --- counterfactual ---------------------------------------------------

    def counterfactual(self, seq: int, overrides: dict) -> dict:
        """What would this decision have been with different inputs?

        Runs the ORIGINAL policy version, not the current one. Answering "what
        would we do now" is a different and usually less useful question than
        "what would we have done then", and conflating them is how a review
        concludes the system behaved reasonably when it did not.
        """
        rec = self.records[seq]
        key = f"{rec.policy}@{rec.policy_version}"
        if key not in self.policies:
            return {"seq": seq, "available": False,
                    "reason": f"policy {key} not registered"}
        new_inputs = {**rec.inputs, **overrides}
        action, score = self.policies[key](new_inputs)
        return {"seq": seq, "available": True, "overrides": overrides,
                "original_action": rec.action, "counterfactual_action": action,
                "original_score": round(rec.score, 6),
                "counterfactual_score": round(float(score), 6),
                "changed": action != rec.action,
                "score_delta": round(float(score) - rec.score, 6)}

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(asdict(r), sort_keys=True, default=str)
                         for r in self.records)

    def stats(self) -> dict:
        intact, broken = self.verify()
        by_action: dict = {}
        for r in self.records:
            by_action[str(r.action)] = by_action.get(str(r.action), 0) + 1
        return {"n_decisions": len(self.records), "chain_intact": intact,
                "first_broken_index": broken,
                "policies": sorted(self.policies),
                "action_counts": by_action}

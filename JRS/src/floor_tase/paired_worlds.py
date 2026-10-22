"""One-factor interventions on a shared latent world, never regenerated tasks.

A horizon change leaves absolute deadlines unchanged. This separates an operating
budget intervention from a deadline intervention. The same keyed uniform draws
are thresholded at different public/actual failure probabilities. Layout changes
are NOT paired here: they alter geometry and require a separate stratum.
"""
from dataclasses import asdict, replace
import hashlib
import json
from .model import ScenarioConfig, make_scenario

DEFAULT_INTERVENTIONS = {
    "reference": {},
    "collateral_off": {"collateral_probability": 0., "assumed_collateral_probability": 0.},
    "collateral_high": {"collateral_probability": .4, "assumed_collateral_probability": .4},
    "failure_high": {"repair_failure_prob": .2, "assumed_repair_failure_prob": .2},
    "passage_off": {"shared_passage": False},
    "budget_short": {"horizon": 75.},
    "budget_long": {"horizon": 200.},
}
INTERVENTION_FIELDS = frozenset({"collateral_probability", "assumed_collateral_probability",
    "repair_failure_prob", "assumed_repair_failure_prob", "shared_passage", "horizon"})


def invariant_fingerprint(scenario):
    """Hash geometry, initial defects, tasks/deadlines and primitive noise IDs."""
    payload = {k: asdict(scenario)[k] for k in ("seed", "measurement_seed", "cells",
               "robots", "jobs", "views", "tasks", "initial_quality")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def paired_worlds(seed, base_config=None, interventions=None):
    """Generate once and apply declared rate/resource/budget interventions.

    Parameters are configuration *assumptions*, not estimated physical facts.
    To study model misspecification alter actual rate without its assumed rate.
    """
    base = make_scenario(seed, base_config or ScenarioConfig())
    conditions = DEFAULT_INTERVENTIONS if interventions is None else interventions
    result = {}
    expected = invariant_fingerprint(base)
    for name, changes in conditions.items():
        unknown = set(changes) - INTERVENTION_FIELDS
        if unknown:
            raise ValueError(f"Not an isolated supported intervention: {sorted(unknown)}")
        cfg = replace(base.config, **changes)
        world = replace(base, id=f"{base.id}:{name}", config=cfg,
                        description=f"Paired intervention {name}; absolute tasks/deadlines unchanged")
        if invariant_fingerprint(world) != expected:
            raise AssertionError("paired latent identity changed")
        result[name] = world
    return result


def paired_conditions(seed, base_config=None):
    return paired_worlds(seed, base_config)

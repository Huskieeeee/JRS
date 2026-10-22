"""Handoff-oriented recovery: local dependencies, dual-purpose inspection and rollout.

Inputs are public Observation and advertised actions. No live environment enters
the controller. Candidate forecasts use NEW synthetic public-belief worlds and
the SAME frontier policy as their continuation. Repeated online rollout is a
receding-horizon improvement heuristic, not an exact value of itself.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time

from .model import Observation
from .planners import _akey, _dist, _fixes, _impact, _pack
from .prediction import public_belief_environment


@dataclass(frozen=True)
class HandoffOptions:
    diagnostic_checks: bool = True
    local_dependencies: bool = True
    adaptive_scope: bool = True
    prediction_mode: str = "collateral"
    max_candidates: int = 8
    samples: int = 3
    max_rollout_events: int = 2000

    def __post_init__(self):
        if self.prediction_mode not in ("optimistic", "collateral", "cascade"):
            raise ValueError("Unknown prediction mode")
        if min(self.max_candidates, self.samples, self.max_rollout_events) < 1:
            raise ValueError("Positive planning budgets required")


class HandoffFrontier:
    """Work-conserving policy and deterministic first-action candidate generator.

    Final handoffs wait only for relevant known interventions. Diagnostic views
    may precede future interventions when they contain unknown quality that can
    reveal a new repair. No extra physical action type or relaxed quality gate
    is introduced. All admission decisions use the original legal-action API.
    """

    def __init__(self, options=None):
        self.options = options or HandoffOptions()

    @staticmethod
    def _incoming(obs):
        versions = {c.id: c.version for c in obs.cells}
        covered = set()
        for packet in obs.pending_inspections:
            if packet.delivery_time - packet.captured_at <= obs.freshness + 1e-9:
                covered.update(q for q, v in packet.captured_versions if versions.get(q) == v)
        for action in obs.active:
            if action.kind == "inspect" and obs.measurement_delay <= obs.freshness + 1e-9:
                covered.update(action.cells)
        return covered

    def features(self, obs, allowed):
        if not isinstance(obs, Observation):
            raise TypeError("Controller requires public Observation")
        cells = {c.id: c for c in obs.cells}
        jobs = {j.id: j for j in obs.jobs}
        views = {v.id: v for v in obs.views}
        tasks = {t.id: t for t in obs.tasks}
        robots = {r.id: r for r in obs.robots}
        pending = [j for j in obs.jobs if j.status in ("pending", "running")]
        allcells = set(cells)
        # Running impacts use the ACTUAL public selected scope, not a local proxy.
        running = {a.target_id: set(a.cells) for a in obs.active if a.kind == "repair"}
        impacts = {j.id: (allcells if obs.evidence_policy == "whole_reset" else
                         running.get(j.id, set(j.local_impact))) for j in pending}
        blocked = set().union(*impacts.values()) if impacts else set()
        known_targets = set().union(*(set(j.targets) for j in pending)) if pending else set()
        incoming = self._incoming(obs)
        live_tasks = [t for t in obs.tasks if t.status == "pending"]
        needs = set().union(*(set(t.requires) for t in live_tasks)) if live_tasks else set()
        # Done certificates need no refresh merely due to age; future repairs
        # that invalidate done tasks are charged explicitly below.
        refresh = set()
        for task in live_tasks:
            earliest = min((max(obs.time, r.busy_until) + _dist(r.position, task.position) / obs.travel_speed
                            + task.duration for r in obs.robots if r.skill == task.skill), default=math.inf)
            refresh.update(q for q in task.requires if cells[q].last_observed is not None and
                           earliest - cells[q].last_observed > obs.freshness + 1e-9)
        exhausted = {q for j in obs.jobs if j.status == "exhausted" for q in j.targets
                     if cells[q].status == "invalid"}
        # The shared environment deliberately keeps target evidence INVALID
        # after a repair completes until it is measured. With no outstanding
        # repair, INVALID is a recertification obligation, not proof of failure.
        unknown = {q for q in needs if (not cells[q].certified or q in refresh) and
                   q not in known_targets and q not in exhausted and q not in incoming}
        rows = []
        for action in sorted(allowed, key=_akey):
            robot = robots[action.robot_id]
            row = {"action": action, "diagnostic_cells": (), "acceptance_cells": (),
                   "unlock": 0., "gain": 0., "externality": 0., "role": action.kind}
            if action.kind == "repair":
                if not self.options.adaptive_scope and action.mode != "local":
                    continue
                job = jobs[action.target_id]
                fixed = _fixes(job, action.mode)
                impact = allcells if obs.evidence_policy == "whole_reset" else _impact(job, action.mode)
                duration = job.expanded_duration if action.mode == "expanded" else job.duration
                move = _dist(robot.position, job.position) / obs.travel_speed
                # Public one-generation expected recovery work; not hidden truth.
                new_risk = obs.assumed_collateral_probability * len(impact - fixed)
                certificate_work = sum(cells[q].certified for q in impact)
                redo = sum(t.duration for t in obs.tasks
                           if t.status == "done" and set(t.requires) & impact)
                checks = sum(min((v.duration for v in obs.views if q in v.covers), default=0.)
                             for q in impact if cells[q].certified)
                direct = len(fixed & known_targets)
                uncertainty_gain = obs.assumed_collateral_probability * len(fixed & unknown)
                row.update(gain=direct + uncertainty_gain,
                           externality=redo + new_risk * obs.repair_duration + checks,
                           duration=duration + move,
                           cost=duration + move + redo + new_risk * obs.repair_duration + checks,
                           unlock=sum(bool(set(t.requires) & fixed) for t in live_tasks),
                           certificate_revocations=certificate_work,
                           deadline=min((t.deadline for t in live_tasks if set(t.requires) & fixed),
                                        default=obs.horizon),
                           fixes=tuple(sorted(fixed)), impact=tuple(sorted(impact)))
            elif action.kind == "inspect":
                view = views[action.target_id]
                useful = set(view.covers) & unknown
                if not self.options.local_dependencies and pending:
                    continue
                diagnostic = useful & blocked
                acceptance = useful - blocked
                if not self.options.diagnostic_checks:
                    useful = acceptance
                    diagnostic = set()
                if not useful:
                    continue
                move = _dist(robot.position, view.position) / obs.travel_speed
                # At least one useful measurement must arrive while still fresh.
                if obs.measurement_delay > obs.freshness + 1e-9:
                    continue
                unlock = sum(bool(set(t.requires) & acceptance) and
                             not (set(t.requires) & (blocked | exhausted | known_targets)) and
                             all(cells[q].certified or q in view.covers or q in incoming for q in t.requires)
                             for t in live_tasks)
                row.update(role="mixed" if diagnostic and acceptance else "diagnostic" if diagnostic else "acceptance",
                           diagnostic_cells=tuple(sorted(diagnostic)), acceptance_cells=tuple(sorted(acceptance)),
                           gain=len(acceptance) + obs.assumed_collateral_probability * len(diagnostic),
                           unlock=unlock, duration=view.duration + move,
                           cost=view.duration + move + obs.measurement_delay,
                           deadline=min((t.deadline for t in live_tasks if set(t.requires) & useful), default=obs.horizon))
            elif action.kind == "successor":
                task = tasks[action.target_id]
                if (set(task.requires) & blocked) or (not self.options.local_dependencies and pending):
                    continue
                duration = task.duration + _dist(robot.position, task.position) / obs.travel_speed
                finish = obs.time + duration
                if any(cells[q].last_observed is None or
                       finish - cells[q].last_observed > obs.freshness + 1e-9 for q in task.requires):
                    continue
                row.update(gain=1., unlock=1., duration=duration, cost=duration,
                           deadline=task.deadline, protected_cells=task.requires)
            else:
                continue
            row["slack"] = row["deadline"] - obs.time - row["duration"]
            rows.append(row)
        return rows

    @staticmethod
    def _rank(row, profile):
        kind = row["action"].kind
        # Alternative first-step allocations are judged by full public rollouts.
        # The continuation uses the fixed balanced profile, making its meaning explicit.
        if profile == "repair":
            priority = {"repair": 0, "successor": 1, "inspect": 2}[kind]
        elif profile == "diagnose":
            priority = 0 if row["diagnostic_cells"] else {"successor": 1, "inspect": 2, "repair": 3}[kind]
        elif profile == "deadline":
            priority = max(0., row["slack"]) / max(row["duration"], .1)
        else:
            priority = 0 if kind == "successor" else 1 if kind == "inspect" and row["unlock"] else 2 if kind == "repair" else 3
        efficiency = row["cost"] / max(.05, row["gain"] + .5 * row["unlock"])
        return (priority, efficiency, row["slack"], _akey(row["action"]))

    def candidates(self, obs, allowed):
        rows = self.features(obs, allowed)
        result, seen = [], set()
        def add(ordered):
            chosen = _pack([r["action"] for r in ordered], obs, allowed)
            key = tuple(sorted(_akey(a) for a in chosen))
            if chosen and key not in seen:
                seen.add(key)
                result.append((chosen, {"actions": [self.describe(r) for r in ordered if r["action"] in chosen]}))
        for profile in ("balanced", "repair", "diagnose", "deadline"):
            add(sorted(rows, key=lambda r: self._rank(r, profile)))
        balanced = sorted(rows, key=lambda r: self._rank(r, "balanced"))
        # Anchoring each action exposes local/expanded and skill-allocation choices.
        for anchor in balanced:
            add([anchor] + [r for r in balanced if r is not anchor])
            if len(result) >= self.options.max_candidates:
                break
        if (obs.active or obs.pending_packets) and len(result) < self.options.max_candidates:
            result.append(((), {"actions": [], "wait": "allow in-flight work or evidence delivery to finish"}))
        if not result:
            result.append(((), {"actions": [], "wait": "pending event or no productive action"}))
        return tuple(result[:self.options.max_candidates])

    @staticmethod
    def describe(row):
        return {k: asdict(v) if k == "action" else v for k, v in row.items()}

    def select(self, obs, allowed):
        rows = sorted(self.features(obs, allowed), key=lambda r: self._rank(r, "balanced"))
        return _pack([r["action"] for r in rows], obs, allowed)


class HandoffPlanner:
    def __init__(self, seed=0, options=None, rollout=True):
        self.seed = seed
        self.options = options or HandoffOptions()
        self.frontier = HandoffFrontier(self.options)
        self.rollout = rollout
        self.name = "handoff" if rollout else "handoff_rule"
        self.stats = {"calls": 0, "master_candidates": 0, "search_nodes": 0,
                      "rollout_events": 0, "rollout_worlds": 0, "total_wall_s": 0.,
                      "diagnostic_dispatches": 0, "acceptance_dispatches": 0,
                      "mixed_dispatches": 0, "public_information_only": True,
                      "exact_mission_solver": False}
        self.last_diagnostics = {}

    def diagnostics(self):
        return dict(self.stats, options=asdict(self.options), rollout=self.rollout)

    def _evaluate(self, obs, candidates):
        state = json.dumps(asdict(obs), sort_keys=True, separators=(",", ":"))
        seeds = [int(hashlib.sha256(f"handoff:{self.seed}:{state}:{i}".encode()).hexdigest()[:15], 16)
                 for i in range(self.options.samples)]
        predictions = []
        for actions, explanation in candidates:
            outcomes, steps = [], 0
            for seed in seeds:
                env = public_belief_environment(obs, seed, self.options.prediction_mode)
                env.step_joint(actions)
                count = 1
                while not env.done and count < self.options.max_rollout_events:
                    current = env.observe()
                    env.step_joint(self.frontier.select(current, env.allowed_actions()))
                    count += 1
                if not env.done:
                    raise RuntimeError("Handoff forecast exceeded event budget; no truncated return accepted")
                outcomes.append(env.result())
                steps += count
            average = lambda k: sum(float(r[k]) for r in outcomes) / len(outcomes)
            predictions.append({"actions": [asdict(a) for a in actions],
                                "completion_probability": average("quality_success"), "mean_J": average("J"),
                                "mean_qualified_tasks": average("qualified_tasks"),
                                "mean_finish": average("time"), "mean_inspections": average("inspections"),
                                "mean_revocations": average("successor_revocations"), "events": steps,
                                "explanation": explanation})
            self.stats["rollout_events"] += steps
            self.stats["rollout_worlds"] += len(outcomes)
        return predictions

    def select(self, obs, allowed_actions):
        start = time.perf_counter()
        self.stats["calls"] += 1
        if obs.done:
            return ()
        candidates = self.frontier.candidates(obs, allowed_actions)
        self.stats["master_candidates"] += len(candidates)
        if self.rollout and len(candidates) > 1:
            predictions = self._evaluate(obs, candidates)
            index = min(range(len(candidates)), key=lambda i: (
                -predictions[i]["completion_probability"], predictions[i]["mean_J"],
                tuple(_akey(a) for a in candidates[i][0])))
        else:
            predictions, index = [], 0
        chosen, reason = candidates[index]
        for row in reason["actions"]:
            key = row["role"] + "_dispatches"
            if key in self.stats:
                self.stats[key] += 1
        self.last_diagnostics = {"method": self.name, "selected_index": index,
                                 "selected": reason, "candidate_predictions": predictions,
                                 "candidate_count": len(candidates),
                                 "continuation": "handoff_frontier_same_options_balanced",
                                 "selection_objective": "max predicted qualified completion; min mean unconditional J",
                                 "prediction_mode": self.options.prediction_mode}
        self.stats["total_wall_s"] += time.perf_counter() - start
        return tuple(chosen)

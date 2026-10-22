"""Typed interface for the recovery-frontier benchmark.

This is an event-level scheduling abstraction, not a physics simulator of four
complete construction processes.  Planner input is Observation only.  In
particular Scenario and ScenarioConfig also contain evaluator-only quantities.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional
import hashlib


def keyed_uniform(seed: int, family: str, *keys: object) -> float:
    """Order-independent deterministic random number in [0,1)."""
    raw = "|".join(map(str, (seed, family, *keys))).encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")
    return (value >> 11) / float(1 << 53)


@dataclass(frozen=True)
class ScenarioConfig:
    horizon: float = 100.0
    n_cells: int = 12
    n_regions: int = 3
    robots_per_skill: int = 1
    repair_fraction: float = 0.35
    impact_radius: int = 1                 # evaluator-only true radius
    assumed_radius: int = 1                # public conservative radius
    sensor_error: float = 0.05             # evaluator-only actual bounded noise
    assumed_sensor_error: float = 0.05     # public certified noise bound
    sensor_bias: float = 0.0               # evaluator-only stress perturbation
    measurement_delay: float = 1.0
    inspection_duration: float = 2.0
    successor_duration: float = 3.0
    repair_duration: float = 4.0
    deadline_slack: float = 1.0
    travel_speed: float = 2.0
    layout: str = "irregular"
    repair_failure_prob: float = 0.0
    collateral_probability: float = 0.20
    repair_overlap: float = 0.5
    evidence_policy: str = "robust_local"
    quality_threshold: float = 0.5
    freshness: float = 200.0
    max_repairs_per_cell: int = 3
    view_radius: int = 2
    inspection_repeat: int = 1
    # Public model assumptions are distinct from evaluator-only actual rates.
    assumed_collateral_probability: float = 0.20
    assumed_repair_failure_prob: float = 0.0
    shared_passage: bool = True

    def __post_init__(self):
        if self.horizon <= 0 or self.travel_speed <= 0:
            raise ValueError("horizon and travel_speed must be positive")
        if self.n_cells < 4 or self.n_regions < 1 or self.robots_per_skill < 1:
            raise ValueError("at least four cells, one region, and one robot per skill")
        if min(self.impact_radius, self.assumed_radius) < 0:
            raise ValueError("impact radii must be nonnegative")
        if self.evidence_policy not in {"robust_local", "whole_reset", "optimistic_local"}:
            raise ValueError("unknown evidence policy")
        if not 0 <= self.repair_fraction <= 1 or not 0 <= self.repair_overlap <= 1:
            raise ValueError("fractions must be in [0,1]")
        for value in (self.repair_failure_prob, self.collateral_probability,
                      self.assumed_collateral_probability, self.assumed_repair_failure_prob):
            if not 0 <= value <= 1:
                raise ValueError("event probabilities must be in [0,1]")
        if self.inspection_repeat < 1 or self.max_repairs_per_cell < 1:
            raise ValueError("positive repeat/repair limits required")


@dataclass(frozen=True)
class Cell:
    id: str
    x: int
    y: int
    region: int
    skill: int = 0
    @property
    def position(self): return (self.x, self.y)


@dataclass(frozen=True)
class Robot:
    id: str
    skill: int
    start: Tuple[float, float]


@dataclass(frozen=True)
class RepairJob:
    id: str
    skill: int
    targets: Tuple[str, ...]
    local_impact: Tuple[str, ...]
    expanded_impact: Tuple[str, ...]
    duration: float
    expanded_duration: float
    expanded_targets: Tuple[str, ...] = ()
    position: Tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class InspectionView:
    id: str
    position: Tuple[float, float]
    covers: Tuple[str, ...]
    duration: float
    skill: int = 0


@dataclass(frozen=True)
class SuccessorTask:
    id: str
    skill: int
    requires: Tuple[str, ...]
    duration: float
    deadline: float
    position: Tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class Scenario:
    id: str
    seed: int
    config: ScenarioConfig
    cells: Tuple[Cell, ...]
    robots: Tuple[Robot, ...]
    jobs: Tuple[RepairJob, ...]
    views: Tuple[InspectionView, ...]
    tasks: Tuple[SuccessorTask, ...]
    initial_quality: Tuple[Tuple[str, float], ...] = ()
    measurement_seed: Optional[int] = None
    description: str = "Synthetic construction recovery-frontier instance"


@dataclass(frozen=True)
class RobotState:
    id: str
    skill: int
    position: Tuple[float, float]
    busy_until: float


@dataclass(frozen=True)
class CellState:
    id: str
    position: Tuple[float, float]
    region: int
    skill: int
    status: str
    version: int
    certified: bool
    mean: Optional[float]
    upper: Optional[float]
    last_observed: Optional[float]
    lower: Optional[float] = None
    batch_id: Optional[str] = None
    samples: int = 0


@dataclass(frozen=True)
class JobState:
    id: str
    skill: int
    targets: Tuple[str, ...]
    local_impact: Tuple[str, ...]
    expanded_impact: Tuple[str, ...]
    duration: float
    expanded_duration: float
    expanded_targets: Tuple[str, ...]
    position: Tuple[float, float]
    status: str
    attempts: int = 0


@dataclass(frozen=True)
class TaskState:
    id: str
    skill: int
    requires: Tuple[str, ...]
    duration: float
    deadline: float
    position: Tuple[float, float]
    status: str


@dataclass(frozen=True)
class ActiveAction:
    kind: str
    robot_id: str
    target_id: str
    mode: str
    start: float
    arrival: float
    finish: float
    cells: Tuple[str, ...]
    channel: Optional[str]
    # Public versions used when authorization actually occurred. They can
    # differ from current versions after expiry during an executing action.
    authorization_versions: Tuple[Tuple[str, int], ...] = ()
    authorization_capture_times: Tuple[Tuple[str, float], ...] = ()
    authorization_order: Optional[int] = None
    completion_order: Optional[int] = None


@dataclass(frozen=True)
class Action:
    kind: str
    robot_id: str
    target_id: str
    mode: str = "local"


@dataclass(frozen=True)
class PendingInspection:
    """Transport metadata known to the team; no undelivered pixel/quality value."""
    view_id: str
    batch_id: str
    captured_at: float
    delivery_time: float
    captured_versions: Tuple[Tuple[str, int], ...]
    queue_order: int = 0


@dataclass(frozen=True)
class Observation:
    time: float
    horizon: float
    robots: Tuple[RobotState, ...]
    cells: Tuple[CellState, ...]
    jobs: Tuple[JobState, ...]
    views: Tuple[InspectionView, ...]
    tasks: Tuple[TaskState, ...]
    active: Tuple[ActiveAction, ...]
    remaining_budget: float
    quality_threshold: float
    assumed_sensor_error: float
    travel_speed: float
    measurement_delay: float
    evidence_policy: str
    pending_packets: int
    done: bool
    freshness: float
    assumed_radius: int = 1
    assumed_collateral_probability: float = 0.20
    assumed_repair_failure_prob: float = 0.0
    shared_passage: bool = True
    repair_duration: float = 4.0
    max_repairs_per_cell: int = 3
    inspection_repeat: int = 1
    pending_inspections: Tuple[PendingInspection, ...] = ()
    accrued_travel: float = 0.0
    accrued_repairs: int = 0


def _footprint(cells, target_ids, radius):
    by_id = {c.id: c for c in cells}
    targets = [by_id[i] for i in target_ids]
    return tuple(c.id for c in cells if any(max(abs(c.x-t.x), abs(c.y-t.y)) <= radius for t in targets))


def make_scenario(seed: int, config: Optional[ScenarioConfig] = None,
                  measurement_seed: Optional[int] = None) -> Scenario:
    """Generate one paired world. Hidden draws do not depend on algorithm calls."""
    cfg = config or ScenarioConfig()
    positions = []
    width = max(4, int(cfg.n_cells ** .5) + 1)
    for y in range(cfg.n_cells * 2):
        for x in range(width):
            if cfg.layout == "l_room" and y >= 2 and x >= 2: continue
            if cfg.layout == "hall" and y > 0 and x not in (0, width-1) and y % 3 == 0: continue
            if cfg.layout == "irregular" and (x + 2*y) % 7 == 6: continue
            positions.append((x,y))
            if len(positions) == cfg.n_cells: break
        if len(positions) == cfg.n_cells: break
    cells = tuple(Cell(f"c{i}", x, y, min(cfg.n_regions-1, i*cfg.n_regions//cfg.n_cells), i % 4)
                  for i,(x,y) in enumerate(positions))
    count = max(1, round(cfg.n_cells*cfg.repair_fraction)) if cfg.repair_fraction else 0
    # Controlled clustering changes job placement; cell geometry is unchanged.
    center = cells[int(keyed_uniform(seed,"cluster_center")*len(cells))]
    ranked = sorted(cells, key=lambda c: (cfg.repair_overlap * max(abs(c.x-center.x),abs(c.y-center.y))
                                          + (1-cfg.repair_overlap)*8*keyed_uniform(seed,"defect",c.id),c.id))
    selected = {c.id for c in ranked[:count]}
    quality = tuple((c.id, .82 if c.id in selected else .16) for c in cells)
    jobs = []
    for c in cells:
        if c.id not in selected: continue
        expanded = tuple(d.id for d in cells if d.skill == c.skill and max(abs(c.x-d.x),abs(c.y-d.y)) <= 2)
        jobs.append(RepairJob("j_"+c.id,c.skill,(c.id,), _footprint(cells,(c.id,),cfg.assumed_radius),
                              _footprint(cells,expanded,cfg.assumed_radius),cfg.repair_duration,
                              cfg.repair_duration*(1+.35*max(1,len(expanded)-1)),expanded,c.position))
    robots = tuple(Robot(f"r{s}_{k}",s,next(c.position for c in cells if c.skill==s))
                   for s in range(4) for k in range(cfg.robots_per_skill))
    views = tuple(InspectionView("v_"+c.id,c.position,
                  tuple(d.id for d in cells if d.skill==c.skill and max(abs(c.x-d.x),abs(c.y-d.y)) <= cfg.view_radius),
                  cfg.inspection_duration,c.skill) for c in cells)
    tasks=[]
    # Certificates of a current operation permit a successor of another skill.
    for region in range(cfg.n_regions):
        for stage in range(4):
            required=tuple(c.id for c in cells if c.region==region and c.skill==stage)
            if not required: continue
            p=next(c.position for c in cells if c.id==required[0])
            tasks.append(SuccessorTask(f"t{region}_{stage}",(stage+1)%4,required,cfg.successor_duration,
                                       min(cfg.horizon,cfg.horizon*cfg.deadline_slack),p))
    return Scenario(f"{cfg.layout}-{seed}",seed,cfg,cells,robots,tuple(jobs),views,tuple(tasks),quality,measurement_seed)


def example_overlap(seed: int=101) -> Scenario:
    """Two interventions share an inspection footprint; four skills remain real."""
    cfg=ScenarioConfig(horizon=60,n_cells=8,n_regions=2,repair_fraction=.25,
                       impact_radius=1,assumed_radius=1,collateral_probability=0,
                       repair_overlap=1,measurement_delay=.5)
    cells=tuple(Cell(f"c{i}",i//2,i%2,i//4,i%4) for i in range(8))
    targets=("c0","c4")
    jobs=tuple(RepairJob(f"j_{t}",0,(t,),_footprint(cells,(t,),1),_footprint(cells,targets,1),4,6,targets,
                         next(c.position for c in cells if c.id==t)) for t in targets)
    robots=tuple(Robot(f"r{s}_0",s,next(c.position for c in cells if c.skill==s)) for s in range(4))
    views=tuple(InspectionView(f"v{s}",(1.5,.5),tuple(c.id for c in cells if c.skill==s),2,s) for s in range(4))
    tasks=tuple(SuccessorTask(f"t{s}",(s+1)%4,tuple(c.id for c in cells if c.skill==s),3,55,(1.5,.5)) for s in range(4))
    quality=tuple((c.id,.82 if c.id in targets else .16) for c in cells)
    return Scenario("hand-overlap",seed,cfg,cells,robots,jobs,views,tasks,quality,description="Shared-view overlapping repair instance")


def example_deadline(seed: int=102) -> Scenario:
    """Waiting for a distant repair before recertifying the early task is late."""
    cfg=ScenarioConfig(horizon=50,n_cells=8,n_regions=2,repair_fraction=.25,
                       impact_radius=0,assumed_radius=0,collateral_probability=0,measurement_delay=.5)
    cells=tuple(Cell(f"c{i}",0 if i<4 else 12,i%4,i//4,i%4) for i in range(8))
    jobs=tuple(RepairJob(f"j_c{i}",0,(f"c{i}",),(f"c{i}",),(f"c{i}",),4,6,(f"c{i}",),cells[i].position) for i in (0,4))
    robots=tuple(Robot(f"r{s}_0",s,cells[s].position) for s in range(4))
    views=tuple(InspectionView(f"v{i}",cells[i].position,(f"c{i}",),2,cells[i].skill) for i in range(8))
    tasks=tuple(SuccessorTask(f"t{i}",(cells[i].skill+1)%4,(f"c{i}",),2,12 if i==0 else 48,cells[i].position) for i in range(8))
    quality=tuple((c.id,.82 if c.id in ("c0","c4") else .16) for c in cells)
    return Scenario("hand-deadline",seed,cfg,cells,robots,jobs,views,tasks,quality,description="Early handoff versus distant intervention")

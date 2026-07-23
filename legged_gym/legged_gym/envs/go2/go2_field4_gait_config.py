""" Gait-quality fine-tune of the Jul21 4-skill oracle.

Measured on the Jul21 run (probe_gait.py, 128 envs, uniform level spawn): on stairsup the
policy touches down 4.5 times/s per foot with 0.131 s of air time -- more than twice the
2 Hz trot the flat specialist was explicitly shaped to produce (air time target 0.25 s).
Joint speed is 25% higher on stairs than on flat at the SAME step frequency, which is why
the legs look frantic there: same cycle time, much larger swing amplitude.

Cause: the May13 recipe carries no gait shaping at all (that machinery was removed because
`alive` + gait terms are what let the robot freeze at a descent edge). Nothing in the
objective prefers a slower, longer stride, so 20k extra iterations changed nothing
(20k/30k/40k checkpoints: 4.54 / 4.35 / 4.50 steps per second).

This run adds ONE term -- feet_air_time -- and nothing else. Deliberately not added:
  - gait_clock observation: would change the obs layout (279 -> 281) and break both the
    warm start and the mutex padding in scripts pad_gait_clock.py
  - gait_phase / feet_clearance: the flat specialist's other two gait terms, but both score
    against `gait_phase_buf`, an internal clock that is randomized at reset and is NOT
    observable without the gait_clock input. Rewarding alignment to an unobservable phase
    is not learnable here, so they would only inject noise.
  - alive: the exact term that produced the freeze-at-edge failure
feet_air_time is the one gait term that needs no clock: it compares measured swing time
against a target derived from the commanded speed (gait_period_range [0.55, 0.35]).

Scale history -- 1.0 (the flat specialist's value) was measured to be too weak. After 2000
iterations at 1.0, stairsup air time moved 0.131 -> 0.138 s against a ~0.22 s target and
cadence did not drop (4.50 -> 4.36 steps/s/foot); much of the shrinking penalty came from
taking fewer steps rather than longer swings, since the term sums over touchdowns. Stair
curriculum meanwhile sat at or above the Jul21 baseline (stairsup 3.1, stairsdown 3.9 vs
3.1 / 3.7), i.e. there was headroom to push harder, so this run resumes that checkpoint at
3.0 (~10% of the tracking reward per step).

Success = air time up toward ~0.2 s and step frequency down toward ~3/s WITHOUT
terrain_level_stairsup/stairsdown dropping below the Jul21 baseline (3.1 / 3.7).
"""
import numpy as np
from os import path as osp

from legged_gym.envs.go2.go2_field4_config import Go2Field4Cfg, Go2Field4CfgPPO

class Go2Field4GaitCfg( Go2Field4Cfg ):
    class rewards( Go2Field4Cfg.rewards ):
        class scales( Go2Field4Cfg.rewards.scales ):
            # air_time_target comes from gait_period_range [0.55, 0.35] via _get_gait_period(),
            # i.e. it adapts with commanded speed -- no gait_clock observation required.
            feet_air_time = 3.0

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go2Field4GaitCfgPPO( Go2Field4CfgPPO ):
    class runner( Go2Field4CfgPPO.runner ):
        resume = True
        # Continues the scale-1.0 attempt rather than restarting from Jul21 model_40000, so its
        # 2000 iterations of adaptation are kept. Both are unpadded (no gait_clock), matching
        # this config's 279-dim observation space; pad only the final checkpoint, for the mutex.
        load_run = osp.join(logs_root, "field_go2",
            "Jul23_08-32-17_Go2Field4Gait_down-jump-stairsup-stairsdown_rAirTime1.0_fromJul21_03-46-49",
        )
        checkpoint = 42000

        run_name = "".join(["Go2Field4Gait_",
            "-".join(Go2Field4GaitCfg.terrain.BarrierTrack_kwargs["options"]),
            ("_rAirTime{:.1f}".format(Go2Field4GaitCfg.rewards.scales.feet_air_time)),
            "_fromJul23_08-32-17",
        ])

        max_iterations = 2000
        save_interval = 500

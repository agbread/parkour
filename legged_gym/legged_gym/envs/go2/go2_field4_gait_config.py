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

Scale 1.0 = the flat specialist's value. At the current cadence this contributes roughly
3% of the tracking reward per step (touchdowns 18/s * dt * (0.131 - 0.208) * scale), so it
biases without overwhelming; 0.3 was tried first and works out to ~1%, too weak to move a
policy that sat at the same cadence for 20k iterations.

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
            feet_air_time = 1.0

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go2Field4GaitCfgPPO( Go2Field4CfgPPO ):
    class runner( Go2Field4CfgPPO.runner ):
        resume = True
        # the unpadded original: this config has no gait_clock, so the obs space matches
        load_run = osp.join(logs_root, "field_go2",
            "Jul21_03-46-49_Go2Field4_down-jump-stairsup-stairsdown_zScale0.03_stairLen0.20_May13recipe_from260511",
        )
        checkpoint = 40000

        run_name = "".join(["Go2Field4Gait_",
            "-".join(Go2Field4GaitCfg.terrain.BarrierTrack_kwargs["options"]),
            ("_rAirTime{:.1f}".format(Go2Field4GaitCfg.rewards.scales.feet_air_time)),
            "_fromJul21_03-46-49",
        ])

        max_iterations = 3000
        save_interval = 500

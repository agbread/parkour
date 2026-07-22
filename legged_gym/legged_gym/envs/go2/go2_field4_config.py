""" 4-skill oracle retrain on the original (260511 -> May13) recipe.
Intended deviations from the proven May13_15-02-33_Go2_4skills run — exactly three:
  1. perlin zScale 0.07 -> 0.03 (milder roughness; 0.07 induces the short-quick-step habit)
  2. options: slope -> jump (jump<->stairsup transfer pair, mirroring down<->stairsdown)
  3. warm start from rough_go2/260511 directly (the May11 10-skills intermediate is deleted)
Everything else (no alive bonus, no gait shaping, wide init randomization, goal-based
commands, fixed 0.2m stair treads, reward scales) matches the May13 config.json verbatim.
Gait-quality machinery (gait_clock obs, feet_air_time) is flat-policy-only by design:
the mutex teacher uses this policy solely on obstacle blocks. """
import numpy as np
from os import path as osp

from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.go2.go2_field_config import Go2FieldCfg, Go2FieldCfgPPO

class Go2Field4Cfg( Go2FieldCfg ):
    class env( Go2FieldCfg.env ):
        # original obs space (no gait_clock) — required to warm start from rough_go2/260511
        obs_components = [
            "lin_vel",
            "ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos",
            "dof_vel",
            "last_actions",
            "height_measurements",
        ]

    class terrain( Go2FieldCfg.terrain ):
        BarrierTrack_kwargs = merge_dict(Go2FieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "down",
                "jump",
                "stairsup",
                "stairsdown",
            ],
            stairsup= dict(
                height= [0.1, 0.3],
                length= [0.2, 0.2], # fixed narrow tread (May13); [0.2, 0.4] diluted the hard case
                residual_distance= 0.05,
                num_steps= [3, 19],
                num_steps_curriculum= True,
            ),
            stairsdown= dict(
                height= [0.1, 0.3],
                length= [0.2, 0.2],
                num_steps= [3, 19],
                num_steps_curriculum= True,
            ),
            add_perlin_noise= True,
            border_perlin_noise= True,
        ))

        TerrainPerlin_kwargs = dict(
            zScale= 0.03,
            frequency= 10,
        )

    class control( Go2FieldCfg.control ):
        computer_clip_torque = False # May13 trained without computer-side clip (motor_clip_torque stays True)

    class rewards( Go2FieldCfg.rewards ):
        class scales:
            # May13 scale set verbatim — notably NO alive (makes edge-freezing profitable)
            # and NO feet_air_time (long-swing shaping fights stair stepping)
            tracking_lin_vel = 1.
            tracking_ang_vel = 1.
            energy_substeps = -2e-7
            torques = -1e-7
            stand_still = -1.
            dof_error_named = -1.
            dof_error = -0.005
            collision = -0.05
            lazy_stop = -3.
            exceed_dof_pos_limits = -0.1
            exceed_torque_limits_l1norm = -0.1
            penetrate_depth = -0.05
        clip_reward_min = None # May13 had no per-step reward floor

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go2Field4CfgPPO( Go2FieldCfgPPO ):
    class runner( Go2FieldCfgPPO.runner ):
        resume = True
        # base rough-walk model, the same starting point the original chain used
        load_run = osp.join(logs_root, "rough_go2", "260511")
        checkpoint = -1 # model_10000

        run_name = "".join(["Go2Field4_",
            "-".join(Go2Field4Cfg.terrain.BarrierTrack_kwargs["options"]),
            ("_zScale{:.2f}".format(Go2Field4Cfg.terrain.TerrainPerlin_kwargs["zScale"])),
            ("_stairLen{:.2f}".format(Go2Field4Cfg.terrain.BarrierTrack_kwargs["stairsup"]["length"][0])),
            "_May13recipe_from260511",
        ])

        max_iterations = 30000 # original field stage was ~48k over 10 skills; extend via resume if curves still rise
        save_interval = 1000
        log_interval = 50

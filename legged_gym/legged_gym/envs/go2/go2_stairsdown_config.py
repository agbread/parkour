""" Stairsdown-only specialist, warm-started from the 4-skill (jump/hurdle/stairs) run.
Descending plateaued at curriculum level ~0.5 in both A/B runs (robots balk at the
edge and time out) while ascending reached 6+; the paper likewise used a separate
descend policy (ClimbMutex jump-down). This becomes mutex sub-policy 2
(obstacle_id_mapping {9:1, 10:2}). """
import numpy as np
from os import path as osp

from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.go2.go2_field_config import Go2FieldCfg
from legged_gym.envs.go2.go2_stairs_config import Go2StairsCfg, Go2StairsCfgPPO

class Go2StairsDownCfg( Go2StairsCfg ):
    class terrain( Go2StairsCfg.terrain ):
        BarrierTrack_kwargs = merge_dict(Go2FieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "stairsdown",
            ],
        ))

    class rewards( Go2StairsCfg.rewards ):
        class scales( Go2StairsCfg.rewards.scales ):
            # standing at the edge nets alive(+2.0) safely while descending only adds
            # tracking(+1.0) against a termination risk -> freeze-at-edge local optimum
            # (both 25k-iter runs plateaued at level ~0.3-0.5 with num_timeouts==num_terminated).
            # go1_down recipe: forward tracking must dominate the reward.
            tracking_lin_vel = 5.0
            # go1_down descend shaping: reward pitching down 0.2 rad while engaging the
            # descend obstacle (mask extended to stairsdown in legged_robot_field.py)
            down_cond = 0.05

    class commands( Go2StairsCfg.commands ):
        class ranges( Go2StairsCfg.commands.ranges ):
            lin_vel_x = [0.3, 0.8] # descending is harder at speed; still within the shared cmd range

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go2StairsDownCfgPPO( Go2StairsCfgPPO ):
    class runner( Go2StairsCfgPPO.runner ):
        resume = True
        # warm start from the 4-skill winner (already climbs stairs, jump/hurdle transfer)
        load_run = osp.join(logs_root, "field_go2",
            "Jul19_17-24-21_Go2Skills4_up0.10-0.30_jump0.05-0.50_pEnergy2.e-07_cmdX0.3-1.0_rAlive2.0_noGoal_fromJul08_11-53-50",
        )
        checkpoint = -1

        run_name = "".join(["Go2StairsDown_",
            ("down{:.2f}-{:.2f}".format(*Go2StairsDownCfg.terrain.BarrierTrack_kwargs["stairsdown"]["height"])),
            ("_cmdX{:.1f}-{:.1f}".format(*Go2StairsDownCfg.commands.ranges.lin_vel_x)),
            ("_rTrackLin{:.1f}".format(Go2StairsDownCfg.rewards.scales.tracking_lin_vel)),
            ("_rDownCond" + np.format_float_scientific(Go2StairsDownCfg.rewards.scales.down_cond, precision=1)),
            ("_rAlive{:.1f}".format(Go2StairsDownCfg.rewards.scales.alive)),
            ("_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])

        max_iterations = 10000

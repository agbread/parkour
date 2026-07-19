""" B-variant of the stairs specialist: co-train jump/hurdle for skill transfer to tall stairs
(A/B against go2_stairs which trains stairs only) """
import numpy as np
from os import path as osp

from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.go2.go2_field_config import Go2FieldCfg
from legged_gym.envs.go2.go2_stairs_config import Go2StairsCfg, Go2StairsCfgPPO

class Go2Skills4Cfg( Go2StairsCfg ):
    class terrain( Go2StairsCfg.terrain ):
        BarrierTrack_kwargs = merge_dict(Go2FieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "jump",
                "hurdle",
                "stairsup",
                "stairsdown",
            ], # jump/hurdle teach the body-lift maneuver that transfers to tall stairs
        ))

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go2Skills4CfgPPO( Go2StairsCfgPPO ):
    class runner( Go2StairsCfgPPO.runner ):
        run_name = "".join(["Go2Skills4_",
            ("up{:.2f}-{:.2f}".format(*Go2Skills4Cfg.terrain.BarrierTrack_kwargs["stairsup"]["height"])),
            ("_jump{:.2f}-{:.2f}".format(*Go2Skills4Cfg.terrain.BarrierTrack_kwargs["jump"]["height"])),
            ("_pEnergy" + np.format_float_scientific(-Go2Skills4Cfg.rewards.scales.energy_substeps, precision=2)),
            ("_cmdX{:.1f}-{:.1f}".format(*Go2Skills4Cfg.commands.ranges.lin_vel_x)),
            ("_rAlive{:.1f}".format(Go2Skills4Cfg.rewards.scales.alive)),
            ("_noGoal" if not Go2Skills4Cfg.commands.is_goal_based else ""),
            ("_from" + "_".join(Go2StairsCfgPPO.runner.load_run.split("/")[-1].split("_")[:2])),
        ])

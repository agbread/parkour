""" Config to train the stairs specialist (stairsup + stairsdown) for the mutex teacher """
import numpy as np
from os import path as osp

from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.go2.go2_field_config import Go2FieldCfg, Go2FieldCfgPPO

class Go2StairsCfg( Go2FieldCfg ):
    class terrain( Go2FieldCfg.terrain ):
        BarrierTrack_kwargs = merge_dict(Go2FieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "stairsup",
                "stairsdown",
            ],
        ))

    class commands( Go2FieldCfg.commands ):
        class ranges( Go2FieldCfg.commands.ranges ):
            # stay within the flat specialist's training range (cmd cap 1.2, gait vref 1.2)
            # so the walk sub-policy and the stairs sub-policy share one command distribution
            lin_vel_x = [0.3, 1.0]

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go2StairsCfgPPO( Go2FieldCfgPPO ):
    class runner( Go2FieldCfgPPO.runner ):
        experiment_name = "field_go2"

        resume = True
        # must be a gait_clock-era flat run (281-dim obs); older flat runs (279-dim) fail to load
        load_run = osp.join(logs_root, "flat_go2",
            "Jul08_11-53-50_Go2Flat_computerClip_pEnergy-1e-05_pDofErr-1e-02_pDofErrN-2e+00_pStand-2e+00_rTrackLin1.5_adaptGait_noResume",
        )
        checkpoint = -1 # latest (8000 as of Jul15); switch load_run to the Jul15 flat run once it finishes

        run_name = "".join(["Go2Stairs_",
            ("up{:.2f}-{:.2f}".format(*Go2StairsCfg.terrain.BarrierTrack_kwargs["stairsup"]["height"])),
            ("_down{:.2f}-{:.2f}".format(*Go2StairsCfg.terrain.BarrierTrack_kwargs["stairsdown"]["height"])),
            ("_pEnergy" + np.format_float_scientific(-Go2StairsCfg.rewards.scales.energy_substeps, precision=2)),
            ("_pPenD" + np.format_float_scientific(-Go2StairsCfg.rewards.scales.penetrate_depth, precision=2)),
            ("_cmdX{:.1f}-{:.1f}".format(*Go2StairsCfg.commands.ranges.lin_vel_x)),
            ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])

        max_iterations = 10000
        save_interval = 500
        log_interval = 50

""" Config to train the stairs specialist (stairsup + stairsdown) for the mutex teacher """
import numpy as np
from os import path as osp

from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.go2.go2_field_config import Go2FieldCfg, Go2FieldCfgPPO

class Go2StairsCfg( Go2FieldCfg ):
    class control( Go2FieldCfg.control ):
        # the flat policy was trained in LeggedRobot, which IGNORES computer_clip_torque
        # (only the noisy field env implements it). With the clip active the flat-initialized
        # policy collapses on flat ground within ~40 steps (verified by ablation probe).
        # motor_clip_torque stays True == the effective flat-training dynamics.
        computer_clip_torque = False

    class init_state( Go2FieldCfg.init_state ):
        pos = [0.0, 0.0, 0.55] # small drop; 0.7 + wild tilt caused instant roll/pitch termination

    class domain_rand( Go2FieldCfg.domain_rand ):
        # tame spawn randomization (upstream field recipe): the flat config's aggressive
        # spawn tilt (+-0.75 rad) combined with the field 1.4/1.6 rad kill thresholds
        # produced 0.7s episodes (spawn -> terminate loop) and blocked all learning
        init_base_rot_range = dict(
            roll= [-0.1, 0.1],
            pitch= [-0.1, 0.1],
        )
        init_base_vel_range = dict(
            x= [-0.1, 0.3],
            y= [-0.1, 0.1],
            z= [-0.1, 0.1],
            roll= [-0.2, 0.2],
            pitch= [-0.2, 0.2],
            yaw= [-0.2, 0.2],
        )
        init_dof_pos_ratio_range = [0.9, 1.1]
        init_dof_vel_range = [-1., 1.]

    class terrain( Go2FieldCfg.terrain ):
        BarrierTrack_kwargs = merge_dict(Go2FieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "stairsup",
                "stairsdown",
            ],
        ))

    class rewards( Go2FieldCfg.rewards ):
        class scales( Go2FieldCfg.rewards.scales ):
            # net per-step reward must stay positive for a surviving robot, or PPO
            # converges to instant-suicide (die fast -> less accumulated penalty).
            # Round-4 failure mode: ep length 500->64/13, "reward" -12 -> -0.3.
            # Same cure as upstream skill configs (a1_crawl etc.): alive bonus.
            alive = 2.0
        clip_reward_min = -10. # physics-spike guard, same as the flat runs

    class commands( Go2FieldCfg.commands ):
        # single-skill training uses plain forward velocity commands (a1/go1 skill recipe).
        # goal-based steering (x_stop_by_yaw_threshold) kept zeroing the x command and
        # blocked learning entirely (tracking reward ~0 for the whole run)
        is_goal_based = False
        class ranges( Go2FieldCfg.commands.ranges ):
            # stay within the flat specialist's training range (cmd cap 1.2, gait vref 1.2)
            # so the walk sub-policy and the stairs sub-policy share one command distribution
            lin_vel_x = [0.3, 1.0]
            lin_vel_y = [0., 0.]
            ang_vel_yaw = [0., 0.]

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
            ("_rAlive{:.1f}".format(Go2StairsCfg.rewards.scales.alive)),
            ("_noGoal" if not Go2StairsCfg.commands.is_goal_based else ""),
            ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])

        max_iterations = 10000
        save_interval = 500
        log_interval = 50

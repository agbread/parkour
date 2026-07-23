""" Flat-walk specialist, re-trained in the field (BarrierTrack) env.

The Jul08 flat run walks well but collapses in the field env (0.13 m/s vs 0.61 m/s at
home, 40% shin/body contact) because it was trained on TerrainPerlin with zScale=0.0:
its height_measurements input was effectively constant (observed min +0.249), while the
field track feeds it a varying map that saturates at -5.0. The obstacle specialist walks
the very same flat blocks at 0.69 m/s, so this is the policy's input distribution, not
the environment.

Fix: keep the Jul08 reward set verbatim (that is what produced the good gait -- gait_clock
observation, feet_air_time/gait_phase shaping, adaptive trot period) and only move the
training terrain to the field track the mutex teacher actually runs on.
This becomes mutex sub-policy 0 (flat blocks); obstacle blocks stay with the Jul21 run.
"""
import numpy as np
from os import path as osp

from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.go2.go2_field_config import Go2FieldCfg, Go2FieldCfgPPO

class Go2WalkFieldCfg( Go2FieldCfg ):
    class terrain( Go2FieldCfg.terrain ):
        # the exact track the teacher/distillation runs on, so the upcoming stairs enter the
        # height map the same way they will at deployment
        BarrierTrack_kwargs = merge_dict(Go2FieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "stairsup",
                "stairsdown",
            ],
        ))

    class control( Go2FieldCfg.control ):
        computer_clip_torque = False # match go2_distill (the specialists share one env there)

    class commands( Go2FieldCfg.commands ):
        class ranges( Go2FieldCfg.commands.ranges ):
            lin_vel_x = [0.3, 1.0] # the distill env's range

    class domain_rand( Go2FieldCfg.domain_rand ):
        # legged_robot.py:813 defaults yaw to [-pi, pi] when the key is absent, so field runs
        # spawn facing a random direction and goal-based commands zero the forward command until
        # the robot turns back (17% of flat steps). Spawn aligned, as go2_distill now does.
        init_base_rot_range = dict(
            roll= Go2FieldCfg.domain_rand.init_base_rot_range["roll"],
            pitch= Go2FieldCfg.domain_rand.init_base_rot_range["pitch"],
            yaw= [-0.3, 0.3],
        )

    class rewards( Go2FieldCfg.rewards ):
        # Jul08 flat run's scales, verbatim: changing any of these risks the gait we are
        # keeping this policy for. The reward *parameters* (base_height_target 0.3,
        # feet_air_time_target 0.25, feet_clearance_target 0.08, gait_period_range
        # [0.55, 0.35], dof_error_names = hips) already match via Go2RoughCfg.
        class scales:
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.
            feet_air_time = 1.
            gait_phase = 0.5
            feet_clearance = -30.
            feet_slip = -0.2
            base_height = -20.
            orientation = -2.
            lin_vel_z = -1.
            ang_vel_xy = -0.05
            action_rate = -0.05
            dof_acc = -2.5e-7
            dof_error = -0.01
            dof_error_named = -2.
            dof_vel_limits = -0.4
            energy_substeps = -1e-5
            collision = -1.
            stand_still = -2.
            stop_lin_vel = -0.5
            exceed_dof_pos_limits = -0.4
            exceed_torque_limits_l1norm = -0.4

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go2WalkFieldCfgPPO( Go2FieldCfgPPO ):
    class runner( Go2FieldCfgPPO.runner ):
        resume = True
        load_run = osp.join(logs_root, "flat_go2",
            "Jul08_11-53-50_Go2Flat_computerClip_pEnergy-1e-05_pDofErr-1e-02_pDofErrN-2e+00_pStand-2e+00_rTrackLin1.5_adaptGait_noResume",
        )
        checkpoint = 15000

        run_name = "".join(["Go2WalkField_",
            "-".join(Go2WalkFieldCfg.terrain.BarrierTrack_kwargs["options"]),
            ("_cmdX{:.1f}-{:.1f}".format(*Go2WalkFieldCfg.commands.ranges.lin_vel_x)),
            ("_rTrackLin{:.1f}".format(Go2WalkFieldCfg.rewards.scales.tracking_lin_vel)),
            ("_rAirTime{:.1f}".format(Go2WalkFieldCfg.rewards.scales.feet_air_time)),
            "_spawnYaw0.3_flatRewards",
            ("_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])

        max_iterations = 5000
        save_interval = 500

"""Low-learning-rate speed-tracking fine-tune for the mutex walk teacher."""

from os import path as osp

from legged_gym.envs.go2.go2_walkfield_config import (
    Go2WalkFieldCfg,
    Go2WalkFieldCfgPPO,
)


class Go2WalkFieldSpeedCfg(Go2WalkFieldCfg):
    class env(Go2WalkFieldCfg.env):
        # The walk specialist is selected only on the flat approach block. End the
        # rollout before it is asked to solve stairs and receives misleading gradients.
        episode_length_s = 2.0

    class commands(Go2WalkFieldCfg.commands):
        # Train the command-to-speed mapping directly. Goal-based steering can rewrite
        # x commands near the obstacle and obscures the tracking error we are fixing.
        is_goal_based = False
        resampling_time = int(1e16)

    class domain_rand(Go2WalkFieldCfg.domain_rand):
        push_robots = False
        init_base_pos_range = dict(x=[0.05, 0.20], y=[-0.05, 0.05])
        init_base_rot_range = dict(
            roll=[-0.05, 0.05],
            pitch=[-0.05, 0.05],
            yaw=[-0.05, 0.05],
        )
        init_base_vel_range = dict(
            x=[0.0, 0.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            roll=[0.0, 0.0],
            pitch=[0.0, 0.0],
            yaw=[0.0, 0.0],
        )
        init_dof_vel_range = [-1.0, 1.0]

    class rewards(Go2WalkFieldCfg.rewards):
        class scales(Go2WalkFieldCfg.rewards.scales):
            # A small increase avoids the critic shock and gait collapse seen at 3.0.
            tracking_lin_vel = 2.0


logs_root = osp.join(
    osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))),
    "logs",
)


class Go2WalkFieldSpeedCfgPPO(Go2WalkFieldCfgPPO):
    class algorithm(Go2WalkFieldCfgPPO.algorithm):
        learning_rate = 1e-5

    class runner(Go2WalkFieldCfgPPO.runner):
        experiment_name = "field_go2"
        resume = True
        load_optimizer = False
        load_run = osp.join(
            logs_root,
            "field_go2",
            "Jul23_08-02-08_Go2WalkField_stairsup-stairsdown_cmdX0.3-1.0_"
            "rTrackLin1.5_rAirTime1.0_spawnYaw0.3_flatRewards_fromJul08_11-53-50",
        )
        checkpoint = 20000
        run_name = "Go2WalkFieldSpeed_directCmd_shortFlat_rTrackLin2_from20000"
        max_iterations = 4000
        save_interval = 250
        log_interval = 50

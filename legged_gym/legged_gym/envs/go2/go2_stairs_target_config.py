"""Fine-tune the current mutex obstacle teacher on the 15 cm x 20 cm stairs."""

from os import path as osp

from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.go2.go2_field4_gait_config import (
    Go2Field4GaitCfg,
    Go2Field4GaitCfgPPO,
)


class Go2StairsTargetCfg(Go2Field4GaitCfg):
    class init_state(Go2Field4GaitCfg.init_state):
        pos = [0.0, 0.0, 0.55]

    class env(Go2Field4GaitCfg.env):
        episode_length_s = 45
        # The distillation mutex uses this padded 281-d observation layout.
        obs_components = [
            "lin_vel",
            "ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos",
            "dof_vel",
            "last_actions",
            "gait_clock",
            "height_measurements",
        ]

    class terrain(Go2Field4GaitCfg.terrain):
        num_rows = 1
        num_cols = 1
        max_init_terrain_level = 0
        curriculum = False
        BarrierTrack_kwargs = merge_dict(
            Go2Field4GaitCfg.terrain.BarrierTrack_kwargs,
            dict(
                options=["stairsup", "stairsdown"],
                stairsup=dict(
                    height=0.15,
                    length=0.20,
                    residual_distance=0.05,
                    num_steps=12,
                    num_steps_curriculum=False,
                ),
                stairsdown=dict(
                    height=0.15,
                    length=0.20,
                    residual_distance=0.05,
                    num_steps=12,
                    num_steps_curriculum=False,
                ),
                track_block_length=2.6,
                n_obstacles_per_track=2,
                randomize_obstacle_order=False,
                add_perlin_noise=False,
                border_perlin_noise=False,
                draw_virtual_terrain=False,
            ),
        )

    class commands(Go2Field4GaitCfg.commands):
        lin_cmd_cutoff = 0.2

        class ranges(Go2Field4GaitCfg.commands.ranges):
            # Match the command distribution seen by the teacher during distillation.
            lin_vel_x = [0.3, 1.0]
            lin_vel_y = [0., 0.]
            ang_vel_yaw = [0., 0.]

    class domain_rand(Go2Field4GaitCfg.domain_rand):
        # Contact termination makes the old Field4 +/-0.75 rad spawn tilt invalid.
        # Keep dynamics randomization, but begin each target-stair rollout upright.
        push_robots = False
        init_base_pos_range = dict(x=[0.10, 0.40], y=[-0.10, 0.10])
        init_base_rot_range = dict(
            roll=[-0.10, 0.10],
            pitch=[-0.10, 0.10],
            yaw=[-0.10, 0.10],
        )
        init_base_vel_range = dict(
            x=[-0.10, 0.30],
            y=[-0.10, 0.10],
            z=[-0.10, 0.10],
            roll=[-0.20, 0.20],
            pitch=[-0.20, 0.20],
            yaw=[-0.20, 0.20],
        )
        init_dof_pos_ratio_range = [0.90, 1.10]
        init_dof_vel_range = [-1.0, 1.0]

    class asset(Go2Field4GaitCfg.asset):
        penalize_contacts_on = ["Head", "base", "thigh", "calf"]
        terminate_after_contacts_on = ["Head", "base"]

    class rewards(Go2Field4GaitCfg.rewards):
        class scales(Go2Field4GaitCfg.rewards.scales):
            tracking_lin_vel = 2.0
            collision = -0.50
            termination = -50.0

        clip_reward_min = -10.

    class sim(Go2Field4GaitCfg.sim):
        class physx(Go2Field4GaitCfg.sim.physx):
            # 1024 articulated robots on stair meshes need a larger aggregate-pair
            # pool; otherwise PhysX silently drops contacts used by rewards/resets.
            max_gpu_contact_pairs = 2**25
            default_buffer_size_multiplier = 20


logs_root = osp.join(
    osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))),
    "logs",
)


class Go2StairsTargetCfgPPO(Go2Field4GaitCfgPPO):
    class algorithm(Go2Field4GaitCfgPPO.algorithm):
        learning_rate = 5e-5

    class runner(Go2Field4GaitCfgPPO.runner):
        experiment_name = "field_go2"
        resume = True
        # pad_gait_clock preserves the old optimizer tensors, whose shapes are stale.
        load_optimizer = False
        load_run = osp.join(
            logs_root,
            "field_go2",
            "Jul23_10-36-50_Go2Field4Gait_down-jump-stairsup-stairsdown_"
            "rAirTime3.0_fromJul23_08-32-17_padGaitClock",
        )
        checkpoint = 44000
        run_name = "Go2StairsTarget_updown_h0.15_d0.20_cmdX0.3-1.0_from44000"
        max_iterations = 4000
        save_interval = 250
        log_interval = 50

""" Config to train the whole parkour oracle policy """
import numpy as np
from os import path as osp
from collections import OrderedDict
from datetime import datetime

from legged_gym.utils.helpers import merge_dict 
from legged_gym.envs.go2.go2_field_config import Go2FieldCfg, Go2FieldCfgPPO, Go2RoughCfgPPO

multi_process_ = True
class Go2DistillCfg( Go2FieldCfg ):
    class env( Go2FieldCfg.env ):
        num_envs = 256
        obs_components = [
            "lin_vel",
            "ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos",
            "dof_vel",
            "last_actions",
            "gait_clock", # student also observes the clock to imitate the clock-driven walk teacher
            "forward_depth",
        ]

        # must exactly reproduce the mutex teacher's input:
        # [shared specialist layout (281) | engaging_block (203, selection only)]
        privileged_obs_components = [
            "lin_vel",
            "ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos",
            "dof_vel",
            "last_actions",
            "gait_clock",
            "height_measurements",
            "engaging_block",
        ]

    class terrain( Go2FieldCfg.terrain ):
        if multi_process_:
            num_rows = 4
            num_cols = 1
        else:
            num_rows = 10
            num_cols = 20
        curriculum = False

        BarrierTrack_kwargs = merge_dict(Go2FieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "stairsup",
                "stairsdown",
            ], # flat-walk segments come from the flat start/run blocks between obstacles
        ))

    class sensor( Go2FieldCfg.sensor ):
        class forward_camera:
            obs_components = ["forward_depth"]
            resolution = [int(480/4), int(640/4)]
            position = dict(
                mean= [0.24, -0.0175, 0.12],
                std= [0.01, 0.0025, 0.03],
            )
            rotation = dict(
                lower= [-0.1, 0.37, -0.1],
                upper= [0.1, 0.43, 0.1],
            )
            resized_resolution = [48, 64]
            output_resolution = [48, 64]
            horizontal_fov = [86, 90]
            crop_top_bottom = [int(48/4), 0]
            crop_left_right = [int(28/4), int(36/4)]
            near_plane = 0.05
            depth_range = [0.0, 3.0]

            latency_range = [0.08, 0.142]
            latency_resampling_time = 5.0
            refresh_duration = 1/10 # [s]

    class commands( Go2FieldCfg.commands ):
        # a mixture of command sampling and goal_based command update allows only high speed range
        # in x-axis but no limits on y-axis and yaw-axis
        lin_cmd_cutoff = 0.2
        class ranges( Go2FieldCfg.commands.ranges ):
            # stay within the flat specialist's training range (cmd cap 1.2, gait vref 1.2)
            lin_vel_x = [0.3, 1.0]
        
        is_goal_based = True
        class goal_based:
            # the ratios are related to the goal position in robot frame
            x_ratio = None # sample from lin_vel_x range
            y_ratio = 1.2
            yaw_ratio = 0.8
            follow_cmd_cutoff = True
            x_stop_by_yaw_threshold = 1. # stop when yaw is over this threshold [rad]

    class normalization( Go2FieldCfg.normalization ):
        class obs_scales( Go2FieldCfg.normalization.obs_scales ):
            forward_depth = 1.0

    class noise( Go2FieldCfg.noise ):
        add_noise = False
        class noise_scales( Go2FieldCfg.noise.noise_scales ):
            forward_depth = 0.0
            ### noise for simulating sensors
            commands = 0.1
            lin_vel = 0.1
            ang_vel = 0.1
            projected_gravity = 0.02
            dof_pos = 0.02
            dof_vel = 0.2
            last_actions = 0.
            ### noise for simulating sensors
        class forward_depth:
            stereo_min_distance = 0.3 # when using (480, 640) resolution
            stereo_far_distance = 2.0
            stereo_far_noise_std = 0.08 
            stereo_near_noise_std = 0.02
            stereo_full_block_artifacts_prob = 0.008
            stereo_full_block_values = [0.0, 0.25, 0.5, 1., 3.]
            stereo_full_block_height_mean_std = [62, 1.5]
            stereo_full_block_width_mean_std = [3, 0.01]
            stereo_half_block_spark_prob = 0.02
            stereo_half_block_value = 3000
            sky_artifacts_prob = 0.0001
            sky_artifacts_far_distance = 2.
            sky_artifacts_values = [0.6, 1., 1.2, 1.5, 1.8]
            sky_artifacts_height_mean_std = [2, 3.2]
            sky_artifacts_width_mean_std = [2, 3.2]

    class curriculum:
        no_moveup_when_fall = False

    class sim( Go2FieldCfg.sim ):
        no_camera = False
    
logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go2DistillCfgPPO( Go2FieldCfgPPO ):
    class algorithm( Go2FieldCfgPPO.algorithm ):
        entropy_coef = 0.0
        using_ppo = False
        num_learning_epochs = 8
        num_mini_batches = 2
        distill_target = "l1"
        learning_rate = 3e-4
        optimizer_class_name = "AdamW"
        teacher_act_prob = 0.
        distillation_loss_coef = 1.0
        # update_times_scale = 100
        action_labels_from_sample = False

        # mutex teacher: walk specialist (idx 0) + stairs specialist (idx 1),
        # switched by the engaging_block obstacle id (selection-only tail component)
        teacher_policy_class_name = "ActorCriticTailFieldMutex"
        teacher_ac_path = None # the mutex loads each sub policy's newest model_*.pt itself

        class teacher_policy:
            num_actor_obs = 48 + 2 + 21 * 11 + (1 + 200 + 2) # 484: shared layout 281 + engaging_block 203
            num_critic_obs = 48 + 2 + 21 * 11 + (1 + 200 + 2)
            num_actions = 12
            obs_segments = OrderedDict([
                ("lin_vel", (3,)),
                ("ang_vel", (3,)),
                ("projected_gravity", (3,)),
                ("commands", (3,)),
                ("dof_pos", (12,)),
                ("dof_vel", (12,)),
                ("last_actions", (12,)), # till here: 3+3+3+3+12+12+12 = 48
                ("gait_clock", (2,)),
                ("height_measurements", (1, 21, 11)),
                ("engaging_block", (1 + 200 + 2,)), # 1 + terrain.max_track_options + block_info_dim
            ])

            sub_policy_class_name = "EncoderStateAcRecurrent"
            sub_policy_paths = [ # must both be gait_clock-era runs (281-dim shared layout)
                osp.join(logs_root, "flat_go2",
                    "Jul08_11-53-50_Go2Flat_computerClip_pEnergy-1e-05_pDofErr-1e-02_pDofErrN-2e+00_pStand-2e+00_rTrackLin1.5_adaptGait_noResume"),
                osp.join(logs_root, "field_go2",
                    "REPLACE_WITH_GO2_STAIRS_RUN"), # fill in after training go2_stairs
            ]
            obstacle_id_mapping = {9: 1, 10: 1} # stairsup/stairsdown -> stairs policy; others -> walk
            env_action_scale = 0.5
            action_smoothing_buffer_len = 3
            reset_non_selected = "when_skill"
            cmd_vel_mapping = {} # keep the env's goal-based commands (both specialists trained on them)

    class policy( Go2RoughCfgPPO.policy ):
        # configs for estimator module
        estimator_obs_components = [
            "ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos",
            "dof_vel",
            "last_actions",
        ]
        estimator_target_components = ["lin_vel"]
        replace_state_prob = 1.0
        class estimator_kwargs:
            hidden_sizes = [128, 64]
            nonlinearity = "CELU"
        # configs for visual encoder
        encoder_component_names = ["forward_depth"]
        encoder_class_name = "Conv2dHeadModel"
        class encoder_kwargs:
            channels = [16, 32, 32]
            kernel_sizes = [5, 4, 3]
            strides = [2, 2, 1]
            hidden_sizes = [128]
            use_maxpool = True
            nonlinearity = "LeakyReLU"
        # configs for critic encoder
        critic_encoder_component_names = ["height_measurements"]
        critic_encoder_class_name = "MlpModel"
        class critic_encoder_kwargs:
            hidden_sizes = [128, 64]
            nonlinearity = "CELU"
        encoder_output_size = 32

        init_noise_std = 0.1

    if multi_process_:
        runner_class_name = "TwoStageRunner"
    class runner( Go2FieldCfgPPO.runner ):
        policy_class_name = "EncoderStateAcRecurrent"
        algorithm_class_name = "EstimatorTPPO"
        experiment_name = "distill_go2"
        num_steps_per_env = 32

        if multi_process_:
            pretrain_iterations = -1
            class pretrain_dataset:
                data_dir = osp.join(logs_root, "distill_go2_data")
                dataset_loops = -1
                random_shuffle_traj_order = True
                keep_latest_n_trajs = 1500
                starting_frame_range = [0, 50]

        resume = True
        # student warm start from the stairs specialist: actor side (memory_a/actor/estimator)
        # is shape-compatible; encoders.0 (depth) and the whole critic side are re-initialized
        load_run = osp.join(logs_root, "field_go2",
            "REPLACE_WITH_GO2_STAIRS_RUN", # fill in after training go2_stairs
        )
        ckpt_manipulator = "replace_encoder0_and_critic" if "field_go2" in load_run else None

        run_name = "".join(["Go2_",
            ("{:d}skills".format(len(Go2DistillCfg.terrain.BarrierTrack_kwargs["options"]))),
            ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])

        max_iterations = 60000
        log_interval = 100
        
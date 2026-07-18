""" Visual sanity-check of the (mutex) teacher policy inside the distill env.
Run this BEFORE collecting distillation data:
    python legged_gym/scripts/play_teacher.py --task go2_distill
Checks to make visually: trot on flat blocks, switch to the stairs policy ~0.8m
before the stairs (engaging_next_threshold), transition smoothness, up/down success.
"""
import isaacgym
from collections import OrderedDict
import torch
import numpy as np
np.float = float

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import class_to_dict

from rsl_rl.modules import build_actor_critic

def main(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs if args.num_envs else 4
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = False
    env_cfg.termination.timeout_at_border = False
    env_cfg.viewer.debug_viz = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    config = class_to_dict(train_cfg)
    config.update(class_to_dict(env_cfg))

    # create teacher policy (same construction path as collect.py)
    policy = build_actor_critic(
        env,
        config["algorithm"]["teacher_policy_class_name"],
        config["algorithm"]["teacher_policy"],
    ).to(env.device)
    if config["algorithm"]["teacher_ac_path"] is not None:
        teacher_ac_path = config["algorithm"]["teacher_ac_path"]
        if "{LEGGED_GYM_ROOT_DIR}" in teacher_ac_path:
            teacher_ac_path = teacher_ac_path.format(LEGGED_GYM_ROOT_DIR= LEGGED_GYM_ROOT_DIR)
        state_dict = torch.load(teacher_ac_path, map_location= "cpu")
        policy.load_state_dict(state_dict["model_state_dict"])
    policy.eval()

    env.reset()
    obs = env.get_observations()
    critic_obs = env.get_privileged_observations()
    assert critic_obs is not None, "The teacher runs on privileged obs; the task must define privileged_obs_components"

    step_count = 0
    while True:
        with torch.no_grad():
            actions = policy.act_inference(critic_obs)
        obs, critic_obs, rews, dones, infos = env.step(actions)
        if dones.any():
            policy.reset(dones)
        step_count += 1
        if hasattr(policy, "get_policy_selection") and step_count % 50 == 0:
            selection = policy.get_policy_selection(critic_obs)
            fractions = selection.float().mean(dim= 0)
            print("step {:6d} | sub-policy selection fractions: {}".format(
                step_count,
                " ".join(["[{}] {:.2f}".format(i, f.item()) for i, f in enumerate(fractions)]),
            ))

if __name__ == "__main__":
    args = get_args()
    main(args)

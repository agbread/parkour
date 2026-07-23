"""Two questions the speed averages cannot answer:

1. Flat tracks 0.52 against a 0.80 command. Is the robot uniformly slow, or does it walk
   near 0.80 and lose it in bursts? -> speed percentiles per block.
2. Does the teacher actually CLEAR the stairs, and up to what difficulty?
   -> per skill, per terrain level: finished the track / fell / stuck until timeout.

Uniform spawn across all levels, curriculum off, fixed command, camera disabled.
"""
import os
import numpy as np
np.float = np.float32
import isaacgym  # noqa: F401
import torch
from collections import defaultdict

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args
from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.go2.go2_distill_config import Go2DistillCfgPPO
from rsl_rl.modules import ActorCriticTailFieldMutex

CMD = float(os.environ.get("CMD", 0.8))


def main():
    args = get_args()
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.obs_components = list(env_cfg.env.privileged_obs_components)
    env_cfg.env.num_envs = 256
    env_cfg.sim.no_camera = True
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 20
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = int(os.environ.get("MAXLVL", env_cfg.terrain.num_rows - 1))
    env_cfg.commands.resampling_time = int(1e16)
    env_cfg.commands.ranges.lin_vel_x = [CMD, CMD]
    env_cfg.termination.timeout_at_finished = True
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    tp = Go2DistillCfgPPO.algorithm.teacher_policy
    kwargs = {k: getattr(tp, k) for k in dir(tp) if not k.startswith("__")}
    policy = ActorCriticTailFieldMutex(**kwargs).to(env.device)
    policy.eval()

    id2name = {v: k for k, v in env.terrain.track_options_id_dict.items()}
    speeds = defaultdict(list)                    # block -> per-step body vx
    outcome = defaultdict(lambda: defaultdict(int))  # (skill, level) -> {finished/fell/stuck}

    track_len = env.terrain.env_block_length * env.terrain.n_blocks_per_track
    print(f"track length {track_len:.2f} m = {env.terrain.n_blocks_per_track} blocks "
          f"(1 flat start + 1 obstacle) x {env.terrain.env_block_length} m; "
          f"'finished' means clearing the obstacle block")
    skill_of_env = env.terrain.get_terrain_type_names(env.terrain_types)
    levels = env.terrain_levels.clone()

    # step() resets finished envs internally, so root_states read afterwards is the NEW spawn.
    # Track progress with our own buffers updated BEFORE each step.
    max_x = torch.zeros(env.num_envs, device=env.device)
    ep_len = torch.zeros(env.num_envs, device=env.device)

    env.reset()
    obs = env.get_observations()
    with torch.no_grad():
        for i in range(3000):
            x_now = env.root_states[:, 0] - env.env_origins[:, 0]
            max_x = torch.maximum(max_x, x_now)
            ep_len = env.episode_length_buf.clone().float()

            actions = policy.act_inference(obs.detach())
            obs, _, _, dones, _ = env.step(actions.detach())

            env.refresh_volume_sample_points()
            types = env.terrain.get_engaging_block_types(
                env.root_states[:, :3],
                env.volume_sample_points - env.root_states[:, :3].unsqueeze(-2),
            )
            bx = env.base_lin_vel[:, 0]
            for t in types.unique():
                m = types == t
                speeds[id2name.get(int(t), "flat")].append(bx[m].cpu().numpy())

            done_idx = dones.nonzero(as_tuple=False).flatten()
            if len(done_idx) > 0:
                for e in done_idx.tolist():
                    skill = skill_of_env[e] if skill_of_env is not None else "?"
                    lvl = int(levels[e])
                    reached = float(max_x[e])
                    if reached > track_len - 0.10:
                        outcome[(skill, lvl)]["finished"] += 1
                    elif ep_len[e] >= env.max_episode_length - 2:
                        outcome[(skill, lvl)]["stuck"] += 1
                    else:
                        outcome[(skill, lvl)]["fell"] += 1
                    outcome[(skill, lvl)]["_sum_x"] += reached
                max_x[done_idx] = 0.

            if (i + 1) % 1000 == 0:
                print(f"step {i+1}/3000", flush=True)
            if i == 1500:
                xs = (env.root_states[:, 0] - env.env_origins[:, 0]).cpu().numpy()
                print(f"[debug] track_len={track_len:.2f} n_blocks={env.terrain.n_blocks_per_track} "
                      f"block_len={env.terrain.env_block_length}")
                print(f"[debug] live x percentiles: p50={np.percentile(xs,50):.2f} "
                      f"p90={np.percentile(xs,90):.2f} max={xs.max():.2f}")
                print(f"[debug] max_x ever percentiles: p50={np.percentile(max_x.cpu().numpy(),50):.2f} "
                      f"p90={np.percentile(max_x.cpu().numpy(),90):.2f} max={max_x.max().item():.2f}")
                print(f"[debug] episode_length max={env.episode_length_buf.max().item()} "
                      f"limit={env.max_episode_length}")

    print(f"\n===== 1. forward speed distribution, command {CMD} m/s (body frame) =====")
    print(f"{'block':12s} {'mean':>6s} {'p10':>6s} {'p25':>6s} {'median':>7s} {'p75':>6s} {'p90':>6s} "
          f"{'>=80% of cmd':>13s}")
    for name in sorted(speeds, key=lambda k: -sum(len(a) for a in speeds[k])):
        v = np.concatenate(speeds[name])
        if len(v) < 5000:
            continue
        near = 100.0 * np.mean(v >= 0.8 * CMD)
        print(f"{name:12s} {v.mean():6.2f} {np.percentile(v,10):6.2f} {np.percentile(v,25):6.2f} "
              f"{np.percentile(v,50):7.2f} {np.percentile(v,75):6.2f} {np.percentile(v,90):6.2f} "
              f"{near:12.0f}%")

    kw = env_cfg.terrain.BarrierTrack_kwargs
    print(f"\n===== 2. per-level outcome (uniform spawn, {env_cfg.terrain.num_rows} levels) =====")
    for skill in sorted({s for s, _ in outcome}):
        h = kw.get(skill, {}).get("height", [0, 0])
        ns = kw.get(skill, {}).get("num_steps", [0, 0])
        ln = kw.get(skill, {}).get("length", [0, 0])
        print(f"\n--- {skill} ---")
        print(f"{'lvl':>4s} {'riser':>7s} {'tread':>7s} {'steps':>6s} "
              f"{'finished':>9s} {'fell':>6s} {'stuck':>6s} {'success':>8s} {'mean x':>7s}")
        for lvl in range(env_cfg.terrain.num_rows):
            o = outcome.get((skill, lvl))
            if not o:
                continue
            d = lvl / (env_cfg.terrain.num_rows - 1)
            n = sum(v for k, v in o.items() if not k.startswith("_"))
            rise = h[0] + (h[1] - h[0]) * d if isinstance(h, (list, tuple)) else h
            tread = ln[0] + (ln[1] - ln[0]) * d if isinstance(ln, (list, tuple)) else ln
            steps = ns[0] + (ns[1] - ns[0]) * d if isinstance(ns, (list, tuple)) else ns
            print(f"{lvl:4d} {rise:6.2f}m {tread:6.2f}m {steps:6.0f} "
                  f"{o['finished']:9d} {o['fell']:6d} {o['stuck']:6d} "
                  f"{100.0*o['finished']/n:7.0f}% {o['_sum_x']/n:6.2f}m")


if __name__ == "__main__":
    main()

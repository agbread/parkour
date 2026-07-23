""" Make a pre-gait_clock checkpoint usable as a mutex sub-policy.

The mutex teacher (ActorCriticMutex) instantiates every sub-policy with ONE observation
layout and then load_state_dict()s each checkpoint into it, so all sub-policies must share
the same input space. The walk specialist observes gait_clock (sin/cos of the trot phase);
the obstacle specialist was trained on the original May13 recipe, which has no gait_clock.

Their checkpoints differ in exactly two tensors -- memory_a/memory_c GRU input weights,
(768, 80) vs (768, 82). Inserting two ZERO columns at index 48 (right after the 48
proprioception values, where gait_clock sits) makes the policy ignore the clock, so it
behaves identically to the original while fitting the shared 281-dim layout.

Layout: [lin_vel 3 | ang_vel 3 | gravity 3 | commands 3 | dof_pos 12 | dof_vel 12 |
         last_actions 12] = 48, then gait_clock 2, then height_measurements 231.

Writes a sibling directory suffixed `_padGaitClock` containing the padded model and a
config.json with gait_clock recorded in obs_components. The source run is never modified.

Usage:
    python legged_gym/scripts/pad_gait_clock.py <run_dir> [model_file] [--reference <ckpt>]

The reference checkpoint (default: the flat specialist) is only used to assert that every
tensor shape matches afterwards.
"""
import argparse
import json
import os
import os.path as osp

import torch

INSERT_AT = 48                      # gait_clock slot in the proprioception block
PAD_WIDTH = 2                       # sin, cos
PAD_KEYS = ["memory_a.rnn.weight_ih_l0", "memory_c.rnn.weight_ih_l0"]


def newest_model(run_dir):
    models = [f for f in os.listdir(run_dir) if "model" in f and f.endswith(".pt")]
    if not models:
        raise SystemExit("no model_*.pt in " + run_dir)
    models.sort(key=lambda m: "{0:0>15}".format(m))   # same ordering the mutex uses
    return models[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="source run directory (not modified)")
    p.add_argument("model", nargs="?", default=None, help="model file (default: newest)")
    p.add_argument("--reference", default=None,
                   help="checkpoint whose tensor shapes the result must match")
    p.add_argument("--out", default=None, help="output dir (default: <run_dir>_padGaitClock)")
    args = p.parse_args()

    run_dir = args.run_dir.rstrip("/")
    model = args.model or newest_model(run_dir)
    out_dir = args.out or (run_dir + "_padGaitClock")
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(osp.join(run_dir, model), map_location="cpu")
    state_dict = ckpt["model_state_dict"]

    expected_in = None
    for key in PAD_KEYS:
        w = state_dict[key]
        if expected_in is None:
            expected_in = w.shape[1]
        elif w.shape[1] != expected_in:
            raise SystemExit("{}: input width {} disagrees with {}".format(key, w.shape[1], expected_in))
        state_dict[key] = torch.cat([
            w[:, :INSERT_AT],
            torch.zeros(w.shape[0], PAD_WIDTH, dtype=w.dtype),
            w[:, INSERT_AT:],
        ], dim=1)
        print("padded {}: {} -> {}".format(key, tuple(w.shape), tuple(state_dict[key].shape)))

    if args.reference:
        ref = torch.load(args.reference, map_location="cpu")["model_state_dict"]
        bad = [k for k, v in state_dict.items() if k not in ref or ref[k].shape != v.shape]
        bad += ["only in reference: " + k for k in ref if k not in state_dict]
        if bad:
            raise SystemExit("SHAPE MISMATCH vs reference:\n  " + "\n  ".join(map(str, bad)))
        print("shape check vs reference: all {} tensors match".format(len(state_dict)))

    torch.save(ckpt, osp.join(out_dir, model))

    # the mutex reads policy kwargs / obs_scales / action_scale out of this file
    with open(osp.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    obs = cfg["env"]["obs_components"]
    if "gait_clock" not in obs:
        obs.insert(obs.index("last_actions") + 1, "gait_clock")
    with open(osp.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=4)

    print("saved:", out_dir)


if __name__ == "__main__":
    main()

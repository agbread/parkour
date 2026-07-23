# Go2 distillation handoff

Everything up to and including the mutex **teacher** was built and verified on a machine
where Isaac Gym cannot render depth (see "Why this is a handoff"). This document is for the
session that runs the **camera-dependent** half: depth collection, distillation, deployment.

---

## Where the pipeline stands

```
per-skill specialists  ->  ActorCriticTailFieldMutex teacher  ->  DAgger distillation  ->  depth student
        DONE                        DONE + verified                   <- YOU ARE HERE          -> real Go2
```

The end product is a **single** policy that runs on the robot from an onboard depth camera
plus proprioception. The teacher is scaffolding: it sees a privileged 231-dim height map the
robot will never have, and exists only to generate action labels for the student.

## Why this is a handoff

The origin machine cannot create Isaac Gym camera tensors. Every compute/graphics GPU
combination available there fails at `cudaExternalMemoryGetMappedBuffer ... error 101`
(invalid device): Isaac Gym renders with Vulkan and shares that memory with CUDA, but Vulkan
enumerates GPUs independently and ignores `CUDA_VISIBLE_DEVICES`, so the two do not line up.
Running inside docker with a single GPU exposed (`--gpus '"device=0"'`) fixes it, which the
target machine can do. Nothing else about the project is machine-specific.

Verify the camera works before anything else:

```bash
python -c "
from isaacgym import gymapi, gymtorch
gym = gymapi.acquire_gym(); sp = gymapi.SimParams()
sp.use_gpu_pipeline = True; sp.physx.use_gpu = True
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
env = gym.create_env(sim, gymapi.Vec3(-1,-1,0), gymapi.Vec3(1,1,1), 1)
props = gymapi.CameraProperties(); props.width, props.height = 64, 48; props.enable_tensors = True
cam = gym.create_camera_sensor(env, props); gym.prepare_sim(sim)
t = gymtorch.wrap_tensor(gym.get_camera_image_gpu_tensor(sim, env, cam, gymapi.IMAGE_DEPTH))
print('OK' if t is not None else 'FAIL', None if t is None else t.shape)"
```

---

## What must be copied alongside the repo

`git pull` brings the code. The two teacher sub-policy checkpoints are **not** in git and
must be copied to the same paths, since `go2_distill_config.py` resolves them relative to the
repo root:

| Slot | Path under `legged_gym/logs/` | Needed files |
|---|---|---|
| 0 — walk (flat blocks) | `field_go2/Jul23_08-02-08_Go2WalkField_stairsup-stairsdown_cmdX0.3-1.0_rTrackLin1.5_rAirTime1.0_spawnYaw0.3_flatRewards_fromJul08_11-53-50/` | `model_20000.pt` + `config.json` |
| 1 — obstacle (stairs) | `field_go2/Jul23_10-36-50_Go2Field4Gait_down-jump-stairsup-stairsdown_rAirTime3.0_fromJul23_08-32-17_padGaitClock/` | `model_44000.pt` + `config.json` |

~18 MB each. `ActorCriticMutex` reads only the newest `model_*.pt` (string-sorted) and
`config.json` from each directory, so intermediate checkpoints do not need to travel.

**Do not substitute an unpadded directory for slot 1.** See below.

---

## Two decisions already made that are easy to undo by accident

### 1. Slot 1 must be the `_padGaitClock` copy

The mutex builds every sub-policy with one observation layout and then loads each checkpoint
into it, so both sub-policies must share the 281-dim layout
`[proprio 48 | gait_clock 2 | height_measurements 231]`.

The obstacle specialist was trained on the original May13 recipe, which has no `gait_clock`
(279 dims). `legged_gym/scripts/pad_gait_clock.py` inserts two **zero** columns at index 48
into the two GRU input weight tensors, so the policy ignores the clock and behaves exactly as
before. This was checked numerically, not just by shape: over 5 inference steps with
deliberately large clock values, `max |action difference| == 0.0`.

To regenerate:

```bash
python legged_gym/scripts/pad_gait_clock.py \
  legged_gym/logs/field_go2/Jul23_10-36-50_Go2Field4Gait_down-jump-stairsup-stairsdown_rAirTime3.0_fromJul23_08-32-17 \
  --reference legged_gym/logs/field_go2/Jul23_08-02-08_Go2WalkField_.../model_20000.pt
```

The padded checkpoint is used **twice**: as mutex slot 1, and as the student's warm start
(`runner.load_run`, with `ckpt_manipulator = "replace_encoder0_and_critic"` re-initializing
the depth encoder and the whole critic). An unpadded checkpoint fails both.

### 2. Slot 0 is NOT `flat_go2/Jul08_11-53-50_...`

That flat specialist walks beautifully in its own environment (0.61 m/s body-frame at a 0.8
command) but **collapses on the field track**: 0.13 m/s and shin/body contact 40% of steps,
on the very flat blocks where the obstacle policy walks at 0.69 m/s. Cause: it was trained on
`TerrainPerlin` with `zScale=0.0`, so its `height_measurements` input was effectively
constant (observed min +0.249, mean 0.679); the BarrierTrack map is out of that distribution
(min -5.0, saturated) and the height encoder's latent is garbage.

Ruled out by individual experiment, so do not re-investigate these: `computer_clip_torque`
mismatch, spawn height 0.5 vs 0.7, proprioception latency 5-45 ms, and goal-based command
zeroing. Observation scales, height grid, action scale, decimation and sim dt are identical
between the two environments.

The fix was to re-train that policy on the field track with **its reward set copied verbatim**
(`go2_walkfield_config.py`) — only the terrain, spawn yaw, torque-clip flag and command range
changed. That run is slot 0.

---

## One environment fix that matters for the student

`legged_robot.py` falls back to `yaw: [-pi, pi]` when `domain_rand.init_base_rot_range` has no
`yaw` key, which every field config inherited — so robots spawned facing a random direction.
With goal-based commands, `x_stop_by_yaw_threshold = 1.0` then forces the forward command to
zero until the robot turns back onto the track: **17% of flat-block steps** with a healthy
policy, 44% with a struggling one.

`Go2DistillCfg.domain_rand.init_base_rot_range` now sets `yaw: [-0.3, 0.3]`. Measured effect:
command zeroing 17.1% -> 0.0%, flat-block tracking 0.54 -> 0.66 m/s at a 0.80 command. It is
deliberately a small range rather than 0.0 so the student still sees heading corrections.

Two earlier attempts to fix this in `_update_command_by_terrain_goal` (requiring the goal to
be ahead in world +x; requiring a minimum goal distance) changed nothing — both measured
17.1% — and were reverted. The bug was never in the goal geometry.

---

## Teacher verification results (reproduce these first)

`ActorCriticTailFieldMutex` rolled out in the distill env with the camera disabled
(`obs_components := privileged_obs_components`, `sim.no_camera = True`), 128 envs, uniform
terrain-level spawn, fixed 0.8 m/s command, 1500 steps:

| Block | Sub-policy used | world vx | body vx | stalled | steps/s/foot | air time |
|---|---|---|---|---|---|---|
| flat | walk 100% | 0.52 | 0.52 | 20.9% | 2.39 | 0.241 s |
| stairsup | obstacle 100% | 0.68 | 0.72 | 7.9% | 3.93 | 0.164 s |
| stairsdown | obstacle 100% | 0.60 | 0.66 | 13.9% | 3.86 | 0.165 s |

`n_obstacle_passed` 0.647, `num_terminated` 1.318 per reset.

Selection is exact — every stairs step uses the obstacle policy, every flat step the walk
policy — and there is no dead zone at the handover. "stalled" = instantaneous forward speed
below 30% of the command.

Flat still tracks 0.52 against a 0.80 command with a fifth of steps stalled. That is the
weakest number in the table and the obvious thing to improve if the student inherits it.

**Caveat when measuring:** use body-frame `base_lin_vel[:, 0]`, or world `root_states[:, 7]`
only where robots are aligned with +x. Averaging world-frame x over randomly-yawed robots
cancels to near zero and looks like a failure that is not there. Also measure gait *inside*
the mutex rollout — running the walk policy alone on this track traps it in front of stairs it
cannot climb (0.08 m/s), which is not a gait result.

---

## Gait history behind slot 1 (why the run has "Gait" in its name)

The obstacle specialist originally stepped at ~4.5 touchdowns/s/foot with 0.13 s of air time
— visibly frantic legs next to the walk policy's 2.4/0.24, and the user noticed. Its May13
recipe carries no gait shaping at all, so cadence is simply not in the objective and did not
budge over 20k iterations (4.54 / 4.35 / 4.50 steps/s at checkpoints 20k / 30k / 40k). Those
terms were removed on purpose: `alive` plus gait shaping is what made an earlier run freeze at
descent edges and never learn to go down stairs.

`go2_field4_gait_config.py` adds `feet_air_time` and nothing else, in two stages:

| | stairsup steps/s/foot | stairsup air time |
|---|---|---|
| Jul21 baseline | 4.50 | 0.131 s |
| scale 1.0, +2000 iters | 4.36 | 0.138 s |
| scale 3.0, +2000 iters | 3.78 | 0.156 s |

Scale 1.0 barely moved it — and much of its shrinking penalty came from taking fewer steps
rather than longer swings, since the term sums over touchdowns. Stair curriculum had headroom
at that point, so the second stage resumed from it at 3.0. Final stair levels stayed at the
baseline (stairsup ~3.2, stairsdown ~4.0 vs 3.1 / 3.7).

Only `feet_air_time` was used, because it is the one gait term that needs no clock: the other
two (`gait_phase`, `feet_clearance`) score against `gait_phase_buf`, which is randomized at
reset and unobservable without the `gait_clock` input — rewarding alignment to it would be
unlearnable, and adding that input would change the observation layout and break both the warm
start and the padding.

Still worth watching on hardware: the policy folds the calf near its hard limit (-2.7227 rad)
on ~37% of descent steps. Functional for clearing edges, but peak torque already reaches
45.3 Nm against a 45.43 Nm motor limit.

---

## Running the distillation

`multi_process_ = True` in `go2_distill_config.py` selects `TwoStageRunner`: one process
collects depth trajectories, another trains on them from `logs/distill_go2_data`.

```bash
# 1. sanity-check the teacher visually (needs the camera; the distill env includes it)
python legged_gym/scripts/play_teacher.py --task go2_distill

# 2. collect  (writes logs/distill_go2_data)
python legged_gym/scripts/collect.py --task go2_distill --headless

# 3. distill
python legged_gym/scripts/train.py --task go2_distill --headless
```

`clear_dataset.py` prunes the collected set. `max_iterations = 60000`,
`teacher_act_prob = 0.` (pure student rollouts), `distill_target = "l1"`.

## Deployment trap

`go2_run.py:88` — `computer_clip_torque`. The specialists here were trained with it **off**
(`Go2Field4Cfg.control.computer_clip_torque = False`, mirrored in `Go2DistillCfg`), so the
onboard code must match or the torques the policy expects will be silently clipped.

---

## Conventions on the origin machine (may not apply here)

- Only GPUs 0 and 1 were usable; 2 and 3 belonged to other users. Confirm the policy here.
- Scripts importing `legged_gym.envs` from outside the repo need `np.float = np.float32`
  before importing isaacgym (numpy 1.24 removed `np.float`; the repo's own scripts patch it).
- `isaacgym` must be imported before `torch`.
- A segfault/core dump when an Isaac Gym process exits is harmless teardown noise.
- `play.py` forces `max_init_terrain_level = 0` and `lin_vel_x = [1.2, 1.2]`; with difficulty 0
  the stairs are ~3 x 10 cm and the robot looks like it is just walking on flat ground. Raise
  the level and pin the command when inspecting a policy.
- Command speed under `goal_based` with `x_ratio = None` is **not** derived from the goal: it
  is resampled from `ranges.lin_vel_x` on every reset. Pin it to a scalar for playback, or
  speed appears to surge and stall at random.

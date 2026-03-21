# Reward System — Duck Hunt GRPO Training

The total reward the model receives for each generation is a weighted combination of two independent reward functions:

```
total_reward = accuracy_weight * accuracy_reward + format_weight * format_reward
```

Default weights: `accuracy_weight = 1.0`, `format_weight = 0.3`.

## 1. Accuracy Reward

Computed in `src/reward.py`. This is the core game reward — did the shot hit a duck?

### Decision flow

```
Model output
    │
    ├─ Can't parse tool call?          → -1.0  (invalid_action)
    │
    ├─ No ducks flying at obs time?    → -0.5  (shoot_nothing)
    │
    ├─ Ducks escaped during wait?      → -0.7  (shoot_dead_duck)
    │   (had_target at obs, not at shot)
    │
    ├─ Hit both ducks?                 → +2.5 - horizon_penalty  (double_kill)
    │
    ├─ Hit one duck?                   → +1.0 - horizon_penalty  (hit)
    │
    └─ Missed?                         → -0.3 - horizon_penalty_miss + proximity_bonus  (miss)
```

### 1.1 Invalid action: `-1.0`

The model produced output that couldn't be parsed into `shoot(x, y, horizon)`. This is the worst possible reward. Examples:

- `"The ducks move in a pattern..."` (text instead of tool call)
- `"[Shoot emoji]"` (no parseable parameters)

**Why -1.0**: must be strictly worse than any valid shot (even a miss at -0.3) so the model always prefers producing a parseable tool call.

### 1.2 Shoot nothing: `-0.5`

The model fired when no ducks were flying at observation time. The model should learn to recognize "no target" states.

### 1.3 Shoot dead duck: `-0.7`

Ducks were flying when the model saw the frames, but by the time the shot lands (`latency_frames + horizon` frames later), all ducks have escaped or fallen. This means the model chose too large a horizon.

**Why -0.7**: worse than a miss (-0.3) because the model wasted a shot on something that was guaranteed to fail. Penalizes excessive horizons. But not as bad as invalid output (-1.0) because the model at least produced a valid tool call.

```
Example:
  - Duck is near top edge, moving up
  - Model predicts horizon=25
  - After 25 frames, duck has escaped off screen
  - Result: shoot_dead_duck = -0.7

  If model had used horizon=5, duck would still be on screen → chance to hit
```

### 1.4 Hit rewards: `+1.0` and `+2.5`

Base reward for hitting one duck (+1.0) or both ducks simultaneously (+2.5). The double kill bonus is 2.5x a single hit because it's much harder — both ducks must be at the same pixel location.

### 1.5 Horizon penalty

**Applied on both hits and misses** with different strengths. Penalizes large horizons to encourage the model to shoot sooner rather than using high horizon as a lottery ticket.

#### On hits:

```
penalty = lambda_horizon * (horizon / max_horizon)
       = 0.4 * (horizon / 30)
```

| Horizon | Penalty | Net single hit | Net double kill |
|---------|---------|---------------|-----------------|
| 0 | 0.000 | +1.000 | +2.500 |
| 5 | 0.067 | +0.933 | +2.433 |
| 10 | 0.133 | +0.867 | +2.367 |
| 15 | 0.200 | +0.800 | +2.300 |
| 20 | 0.267 | +0.733 | +2.233 |
| 30 | 0.400 | +0.600 | +2.100 |

**Why 0.4**: the old value (0.1) was too weak — a hit with h=30 scored +0.9, giving almost no incentive to reduce horizon. At 0.4, a hit with h=30 scores +0.6, creating real pressure to aim with shorter horizons while still keeping any hit better than a miss.

#### On misses:

```
penalty = lambda_horizon_miss * (horizon / max_horizon)
       = 0.2 * (horizon / 30)
```

| Horizon | Penalty | Net miss (before proximity) |
|---------|---------|---------------------------|
| 0 | 0.000 | -0.300 |
| 5 | 0.033 | -0.333 |
| 10 | 0.067 | -0.367 |
| 15 | 0.100 | -0.400 |
| 20 | 0.133 | -0.433 |
| 30 | 0.200 | -0.500 |

**Why penalize horizon on misses**: without this, all misses with the same proximity get identical rewards regardless of horizon. A miss with h=24 is objectively worse than h=2 — the model waited longer and still missed, making the prediction harder. This creates gradient signal to reduce horizon even when the model is mostly missing (e.g. 15% hit rate), which is when horizon optimization matters most.

**Why 0.2 (lower than hits)**: misses are already negative, so we don't need as strong a penalty. The miss with h=30 goes from -0.3 to -0.5, which is meaningful but doesn't approach `shoot_dead_duck` (-0.7) territory.

### 1.6 Miss: `-0.3 - horizon_penalty_miss`

The shot was valid, ducks were flying, but the shot didn't hit any duck's hitbox. The base miss reward is -0.3, further penalized by horizon (see section 1.5).

**Why -0.3 base (not -1.0)**: a miss with a valid tool call is much better than invalid output. The model at least produced the right format and attempted to aim. The proximity bonus (below) provides gradient signal to improve aim.

**Effective miss range**: from -0.3 (h=0) to -0.5 (h=30), before proximity bonus.

### 1.7 Proximity bonus (target-aware)

**Only on misses. Only measures distance to FLYING ducks.**

If one duck is falling/dead and the other is still flying, the proximity bonus only rewards aiming near the flying duck. Aiming at a dead duck gives no proximity signal — the model learns to ignore it and target the alive one.

```
proximity = proximity_bonus * exp(-proximity_decay * min_distance_to_flying_duck)
          = 0.3 * exp(-5.0 * d)
```

| Distance (norm) | Approx pixels | Proximity bonus | Total miss reward |
|----------------|---------------|----------------|-------------------|
| 0.00 | 0 px | +0.300 | +0.000 |
| 0.05 | ~40 px | +0.234 | -0.066 |
| 0.10 | ~80 px | +0.182 | -0.118 |
| 0.20 | ~160 px | +0.110 | -0.190 |
| 0.30 | ~240 px | +0.067 | -0.233 |
| 0.50 | ~400 px | +0.025 | -0.275 |
| 1.00 | ~800 px | +0.002 | -0.298 |

**Target-aware behavior by duck state:**

| Duck A state | Duck B state | Proximity measures distance to |
|---|---|---|
| flying | flying | nearest of A or B |
| flying | falling/escaped | A only |
| falling/escaped | flying | B only |
| falling/escaped | falling/escaped | no proximity bonus (both dead) |

**Why this matters**: without proximity bonus, all misses get the same -0.3 regardless of whether the shot was 10px or 400px from the duck. GRPO needs reward variance across generations to compute advantages. If 5 out of 6 generations miss by different amounts, the proximity bonus creates different rewards for each, giving the optimizer a gradient toward the closer miss.

**The decay rate (5.0)** is tuned so that:
- Within ~50px (one duck-width): meaningful bonus (+0.2)
- Beyond ~200px: negligible bonus (<0.1)
- At max distance (diagonal): effectively zero

## 2. Format Strictness (per model family)

Each model family has a strict parser in `src/formats.py`. Outputs that don't match the model's native format are rejected as **invalid** (accuracy = -1.0, format = 0.0).

**LiquidAI** accepts only pythonic format:
- `<|tool_call_start|>[shoot(x=0.5, y=0.3, horizon=8)]<|tool_call_end|>` — primary
- `shoot(x=0.5, y=0.3, horizon=8)` — plain pythonic (no special tokens)

Mistral-style JSON (`[{"name": "shoot", "arguments": {...}}]`) is **intentionally rejected** even though it contains valid coordinates. Without this, the model gets rewarded for using the wrong format, removing the learning signal to produce its native tool call syntax.

**Mistral** accepts its native format plus JSON/KV fallbacks (Mistral models occasionally vary their JSON structure).

## 3. Format Reward

Computed in `src/dataset.py → make_format_reward_function()`. Scores the structural quality of the output.

### Decision flow

```
Model output
    │
    ├─ Can't parse tool call?          → 0.0
    │
    ├─ Valid call, clean (≤5 extra chars)?  → 1.0
    │
    └─ Valid call + extra text?        → max(0.3, 1.0 - extra_chars * 0.01)
```

### 3.1 No tool call: `0.0`

The model didn't produce any parseable function call. Different from accuracy's -1.0 because format reward is scaled by `format_weight = 0.3`, so the total contribution is `0.3 * 0.0 = 0.0`.

### 3.2 Clean tool call: `1.0`

The model produced a valid tool call with no extra text:
```
<|tool_call_start|>[shoot(x=0.5, y=0.3, horizon=8)]<|tool_call_end|>
```

This is the maximum format reward. Special tokens (`<|pad|>`, `<|im_end|>`) don't count as extra text.

### 3.3 Verbose tool call: `0.3–0.9`

The model produced a valid tool call but also generated explanatory text:
```
The ducks are moving to the right. <|tool_call_start|>[shoot(x=0.5, y=0.3, horizon=8)]<|tool_call_end|>
```

Penalty scales linearly with extra character count:

| Extra chars | Format reward | Example |
|------------|--------------|---------|
| 0–5 | 1.0 | Just the tool call |
| 20 | 0.8 | Short prefix |
| 50 | 0.5 | One sentence of explanation |
| 70+ | 0.3 (floor) | Paragraph of reasoning |

**Why penalize verbosity**: extra tokens waste compute and slow inference. The model should learn to respond with just the tool call. The floor at 0.3 ensures that a verbose-but-valid response is still better than no tool call (0.0).

## 4. Hotspot Penalty (anti-exploit)

Implemented in `src/trainer.py`. Prevents the model from exploiting a fixed position (e.g. always shooting screen center).

The trainer tracks the last 50 shot positions. For each hit (reward > 0), it calculates the **concentration** — what fraction of recent shots are within 0.10 normalized distance (~80px) of the current shot. If concentration exceeds 20%, the reward is scaled down linearly, eventually going **negative** at high concentration.

```
concentration = nearby_shots / total_tracked_shots

if concentration > 0.2:
    scale = 1.0 - (concentration - 0.2) * 5.0
    reward = reward * max(scale, -1.5)
```

| Concentration | Scale | Hit (+1.0) becomes | Effect |
|---|---|---|---|
| <20% | 1.0 | +1.0 | No penalty |
| 20% | 1.0 | +1.0 | Threshold |
| 30% | 0.5 | +0.5 | Reduced |
| 40% | 0.0 | **0.0** | Hit is worthless |
| 50% | -0.5 | **-0.5** | Hit is punished |
| 60% | -1.0 | **-1.0** | Same as invalid output |
| 80%+ | -1.5 (floor) | **-1.5** | Worse than anything |

**Why scaling goes negative**: a flat 0.5x penalty (the old approach) left hotspot hits at +0.5, still much better than any miss. The model rationally preferred spamming the hotspot. With scaling, at 40%+ concentration a hotspot hit is worth 0.0 or less — the only way to get positive reward is to aim somewhere new.

**Why only on hits**: misses are already negative. The penalty specifically makes repeated-position hits unrewarding.

**Parameters** (hardcoded in trainer):
- `hotspot_window`: 50 (track last N shots)
- `hotspot_radius`: 0.10 (~80px zone)
- Scale: linear from 1.0 at 20% to -1.5 floor at 50%+

## 5. Combined Reward

The total reward per generation is:

```
total = 1.0 * accuracy_reward + 0.3 * format_reward
```

Then the hotspot penalty may scale it (only on positive rewards):

```
if concentration > 0.2 AND total > 0:
    scale = 1.0 - (concentration - 0.2) * 5.0
    total = total * max(scale, -1.5)
```

### Example scenarios

| Scenario | Accuracy | Format | Hotspot | Total |
|----------|---------|--------|---------|-------|
| Hit duck, clean output, h=5 | +0.933 | 1.0 | none | +1.233 |
| Hit duck, clean, h=5, **30% conc.** | +0.933 | 1.0 | scale=0.5 | **+0.617** |
| Hit duck, clean, h=5, **50% conc.** | +0.933 | 1.0 | scale=-0.5 | **-0.617** |
| Hit duck, verbose output, h=5 | +0.933 | 0.5 | none | +1.083 |
| Close miss to flying duck (d=0.05), h=5, clean | -0.099 | 1.0 | — | +0.201 |
| Close miss to flying duck (d=0.05), h=20, clean | -0.199 | 1.0 | — | +0.101 |
| Close miss to dead duck (d=0.05), h=5, clean | -0.333 | 1.0 | — | -0.033 |
| Far miss (d=0.5), h=0, clean | -0.275 | 1.0 | — | +0.025 |
| Far miss (d=0.5), h=20, clean | -0.408 | 1.0 | — | -0.108 |
| Miss, no tool call | -1.0 | 0.0 | — | -1.000 |
| Miss, verbose but parseable, h=15 | -0.400 | 0.5 | — | -0.250 |
| Duck escaped during horizon | -0.700 | 1.0 | — | -0.400 |
| Double kill, h=0 | +2.500 | 1.0 | none | +2.800 |
| Double kill, h=0, **40% conc.** | +2.500 | 1.0 | scale=0.0 | **0.000** |

### Key invariants

These must always hold for the reward system to provide correct learning signal:

```
double_kill > hit > close_miss > far_miss > shoot_nothing > shoot_dead > invalid
  +2.5      +1.0    ~-0.1       ~-0.3...-0.5  -0.5           -0.7        -1.0

hit(h=30) > miss                   →  lowest hit (0.6) still better than best miss (0.0)
novel_hit > hotspot_hit(30%)       →  diverse aiming rewarded over position spam
hotspot_hit(50%) < miss            →  severe spam is worse than missing
clean > verbose                    →  format reward always higher for concise output
invalid < miss                     →  always better to produce a parseable tool call
miss_near_flying > miss_near_dead  →  proximity only to alive ducks
```

## 6. How Rewards Drive GRPO Learning

GRPO samples G generations (default 6) for each game state and computes group-normalized advantages:

```
advantages = (rewards - mean(rewards)) / std(rewards)
```

**What matters is reward VARIANCE across generations, not absolute values.**

If all 6 generations get the same reward (e.g. all miss by the same distance), `std = 0` and advantages are zero — no learning happens. The model needs different generations to produce different rewards.

### Sources of reward variance

| Source | How it creates variance |
|--------|----------------------|
| Hit vs miss | +1.0 vs -0.3 = huge signal |
| Different horizons on hits | h=3 (+0.96) vs h=20 (+0.73) = strong signal |
| Different horizons on misses | h=3 (-0.32) vs h=20 (-0.43) = useful variance |
| Proximity bonus on misses (flying only) | close miss (-0.1) vs far miss (-0.28) = moderate signal |
| Aiming at flying vs dead duck | proximity bonus (-0.1) vs no bonus (-0.3) = teaches target selection |
| Format quality | clean (1.0) vs verbose (0.5) = moderate signal |
| Dead duck penalty | miss (-0.3) vs escaped (-0.7) = useful signal for horizon tuning |
| Hotspot penalty | novel hit (+1.0) vs hotspot hit at 40% (0.0) = strong anti-exploit |

### Common failure mode

If the model collapses to one output (e.g. always `shoot(x=0.5, y=0.5, horizon=8)`), all 6 generations get identical rewards. `std = 0`, advantages = 0, no gradient. Multiple mechanisms prevent this:
- Entropy bonus/floor keeps the policy stochastic
- Hotspot penalty makes repeated-position hits negative at >40% concentration
- New match every step ensures diverse duck positions (see section 7)

## 7. Environment Diversity

The trainer forces **a new match every training step**. This means every step has fresh ducks with new random spawn positions, speeds, and directions. Without this, the same two ducks would persist for 15-60 steps while bouncing around, causing the model to see similar frames repeatedly.

Each step:
1. Advance to next match (new pair of ducks spawned off-screen)
2. Advance 8-25 random frames (ducks enter the visible area)
3. Verify at least 1 duck is flying
4. Capture frames and train

Additional diversity mechanisms:
- Ducks spawn from left, right (40% each) or top (20%) edges
- Per-duck speed with +/-20% individual variation
- Random starting round (1-5) on reset — varied speed ranges
- Mid-flight jitter: 3% chance per frame of direction nudge
- Bounce speed jitter: +/-15% on wall bounce

## 8. Curriculum Training

Training is split into two phases to help the model learn incrementally:

### Phase 1: Learn to aim (steps 0 to `curriculum_phase2_step`)

- **Horizon clamped to 0**: whatever horizon the model outputs, the shot is simulated with `horizon=0`
- **Shorter completions**: `phase1_max_completion_length` (default 30 tokens)
- **Goal**: model learns to produce valid tool calls and aim at duck positions
- **Difficulty**: easier — shot lands after processing latency only, no extra prediction needed

### Phase 2: Learn horizon optimization (steps `curriculum_phase2_step` to end)

- **Horizon unlocked**: model's predicted horizon is used in simulation
- **Full completion length**: `max_completion_length` (default 40 tokens)
- **Goal**: model learns to optimize when to shoot
- **Difficulty**: harder — must predict future duck position

### Why curriculum helps

The model needs to learn three things simultaneously: tool call format, spatial aiming, and temporal prediction. Without curriculum, the model often gets stuck learning one (usually tool call format) while failing at the others, leading to collapse. Curriculum separates the easier task (aiming) from the harder one (timing).

```yaml
grpo:
  curriculum_phase2_step: 2000    # 0 = disabled
  phase1_max_completion_length: 30
  max_completion_length: 40       # used in phase 2
```

W&B logs `train/curriculum_phase` (1 or 2) so you can see the transition.

## 9. Early Training Stabilization

LoRA weights start at zero — the first gradient updates have outsized influence on the training direction. If the first few steps happen to have ducks clustered in one area, the model develops a bias that takes hundreds of steps to unlearn. This causes different hit rate trajectories across runs with different seeds.

Three mechanisms reduce this seed sensitivity:

### 9.1 LoRA freeze

LoRA parameters are frozen for the first `lora_freeze_steps` (default 50). The model runs forward passes and collects batches but no weight updates occur. This lets the optimizer/scheduler warm up and populates the hotspot tracker before any learning signal reaches the weights.

### 9.2 Stabilization (boosted gradient accumulation)

For the first `stabilization_steps` (default 500), `gradient_accumulation_steps` is temporarily increased to `stabilization_grad_accum` (default 8, effective batch = 16). Each weight update averages over more duck positions, reducing the influence of any single state.

After `stabilization_steps`, accumulation reverts to the normal value (default 4, effective batch = 8).

### 9.3 Warmup

`warmup_ratio: 0.10` gradually ramps the learning rate from ~0 to the target over 10% of total steps. Early gradients (which are high-variance due to random game states) produce smaller weight updates.

### Timeline

```
Step 0                50              500              2000          end
  |--- LoRA frozen ---|--- stabilization (grad_accum=8) ---|
  |---------------- warmup (LR ramp) ---------|
  |----------------------- Phase 1 (horizon=0) -----------|-- Phase 2 --|
```

### W&B tracking

| Metric | Values | Description |
|--------|--------|-------------|
| `train/lora_frozen` | 0 or 1 | Whether LoRA is currently frozen |
| `train/grad_accum` | 4 or 8 | Current gradient accumulation steps |
| `train/curriculum_phase` | 1 or 2 | Curriculum phase |

```yaml
grpo:
  lora_freeze_steps: 50          # 0 = disabled
  stabilization_steps: 500       # 0 = disabled
  stabilization_grad_accum: 8    # grad_accum during stabilization

training:
  warmup_ratio: 0.10
```

## 11. Config Reference

```yaml
reward:
  hit: 1.0                  # base reward for hitting one duck
  double_kill: 2.5           # base reward for hitting both ducks
  miss: -0.3                 # base reward for missing
  shoot_nothing: -0.5        # shooting when no ducks flying
  shoot_dead_duck: -0.7      # ducks escaped during latency+horizon wait
  invalid_action: -1.0       # unparseable model output
  lambda_horizon: 0.4        # horizon penalty coefficient (on hits)
  lambda_horizon_miss: 0.2   # horizon penalty coefficient (on misses)
  proximity_bonus: 0.3       # max proximity bonus (on misses only)
  proximity_decay: 5.0       # how fast proximity bonus decays with distance
  format_weight: 0.3         # weight for format reward in total
  accuracy_weight: 1.0       # weight for accuracy reward in total
  max_horizon: 30            # max horizon for penalty normalization
```

## 12. Tuning Guide

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Model never produces tool calls | `invalid_action` not negative enough, or format reward weight too low | Increase `format_weight` to 0.5 |
| Model always shoots center | Hitbox too large, or not enough game state diversity | Reduce hitbox, advance more frames between steps |
| Model uses horizon=0 always | `lambda_horizon` too high | Reduce to 0.05 |
| Model uses horizon=30 always | No penalty for dead ducks, or `lambda_horizon` too low | Check `shoot_dead_duck`, increase `lambda_horizon` and `lambda_horizon_miss` |
| All generations get same reward | Model collapsed, or proximity bonus too small | Increase `proximity_bonus`, check entropy config |
| Hit rate plateau | Proximity decay too slow (bonus too easy) | Increase `proximity_decay` to 8.0 |
| Model writes explanations | Format verbosity penalty too weak | Reduce format floor from 0.3 to 0.1 |
| Model always shoots center | Hotspot penalty threshold too high | Reduce concentration threshold or increase scale steepness |
| Model ignores dead ducks | Working correctly — proximity only targets flying ducks | No fix needed |
| Model shoots dead ducks | `shoot_dead_duck` penalty too weak | Increase to -0.9 |
| Same duck positions every step | Environment not advancing to new match | Verify `auto_advance_to_next_match()` is called each step |
| Hit rate stalls in phase 1 | Model learned aiming but needs horizon | Transition to phase 2 earlier (lower `curriculum_phase2_step`) |
| Hit rate drops at phase 2 start | Normal — model adjusting to harder task | Let it run 500+ steps, should recover |
| Different hit rates across runs (<1k steps) | Seed sensitivity — early duck positions bias LoRA direction | Increase `stabilization_steps` and `stabilization_grad_accum`, or increase `lora_freeze_steps` |
| Early training unstable / spiky loss | Gradients too noisy from random game states | Increase `warmup_ratio` to 0.15, increase `stabilization_grad_accum` to 16 |

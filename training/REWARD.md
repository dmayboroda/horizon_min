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
    └─ Missed?                         → -0.3 + proximity_bonus  (miss)
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

**Only applied on hits.** Penalizes large horizons to encourage the model to shoot sooner:

```
penalty = lambda_horizon * (horizon / max_horizon)
       = 0.1 * (horizon / 30)
```

| Horizon | Penalty | Net single hit | Net double kill |
|---------|---------|---------------|-----------------|
| 0 | 0.000 | +1.000 | +2.500 |
| 5 | 0.017 | +0.983 | +2.483 |
| 10 | 0.033 | +0.967 | +2.467 |
| 15 | 0.050 | +0.950 | +2.450 |
| 20 | 0.067 | +0.933 | +2.433 |
| 30 | 0.100 | +0.900 | +2.400 |

**Why only on hits**: on misses, the model needs to explore different horizons to find one that works. Penalizing horizon on misses would discourage this exploration. On hits, we want the model to discover that shorter horizons give higher rewards.

**Why it's small (0.1 max)**: the penalty must never make a hit worse than a miss. A hit with horizon=30 gives `1.0 - 0.1 = 0.9`, still much better than a miss at `-0.3`.

### 1.6 Miss: `-0.3`

The shot was valid, ducks were flying, but the shot didn't hit any duck's hitbox.

**Why -0.3 (not -1.0)**: a miss with a valid tool call is much better than invalid output. The model at least produced the right format and attempted to aim. The proximity bonus (below) provides gradient signal to improve aim.

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

## 2. Format Reward

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

### 2.1 No tool call: `0.0`

The model didn't produce any parseable function call. Different from accuracy's -1.0 because format reward is scaled by `format_weight = 0.3`, so the total contribution is `0.3 * 0.0 = 0.0`.

### 2.2 Clean tool call: `1.0`

The model produced a valid tool call with no extra text:
```
<|tool_call_start|>[shoot(x=0.5, y=0.3, horizon=8)]<|tool_call_end|>
```

This is the maximum format reward. Special tokens (`<|pad|>`, `<|im_end|>`) don't count as extra text.

### 2.3 Verbose tool call: `0.3–0.9`

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

## 3. Hotspot Penalty (anti-exploit)

Implemented in `src/trainer.py`. Prevents the model from exploiting a fixed position (e.g. always shooting screen center).

The trainer tracks the last 50 shot positions. If more than 30% of them are within 0.08 normalized distance (~64px) of the current shot, and the current shot is a **hit** (reward > 0), the reward is multiplied by 0.5.

```
if hit AND >30% of last 50 shots within 0.08 of (x, y):
    reward *= 0.5
```

| Scenario | Raw reward | After hotspot | Why |
|----------|-----------|--------------|-----|
| Hit at (0.5, 0.5), first time | +1.0 | +1.0 | No history of this position |
| Hit at (0.5, 0.5), 20th time in a row | +1.0 | **+0.5** | >30% of recent shots at center |
| Hit at (0.35, 0.6), diverse history | +1.0 | +1.0 | Not a hotspot |
| Miss at (0.5, 0.5), repeated | -0.2 | -0.2 | Only applies to positive rewards |

**Why only on hits**: misses are already negative. Reducing them further would just make the model avoid shooting entirely. The hotspot penalty specifically makes a repeated-position hit less rewarding than a novel-position hit.

**Why 30% threshold**: the model generates 6 shots per step, each tracked independently. If 3 out of 6 generations target the same spot step after step, that position accumulates quickly in the 50-shot window. The 30% threshold triggers after roughly 5 consecutive steps of spamming the same position.

**Parameters** (hardcoded in trainer):
- `hotspot_window`: 50 (track last N shots)
- `hotspot_penalty`: 0.5 (reward multiplier when triggered)
- `hotspot_radius`: 0.08 (~64px zone)

## 4. Combined Reward

The total reward per generation is:

```
total = 1.0 * accuracy_reward + 0.3 * format_reward
```

Then the hotspot penalty may reduce it:

```
if hotspot_detected AND total > 0:
    total *= 0.5
```

### Example scenarios

| Scenario | Accuracy | Format | Hotspot | Total |
|----------|---------|--------|---------|-------|
| Hit duck, clean output, h=5 | +0.983 | 1.0 | no | +1.283 |
| Hit duck, clean, h=5, **hotspot** | +0.983 | 1.0 | **yes** | **+0.642** |
| Hit duck, verbose output, h=5 | +0.983 | 0.5 | no | +1.133 |
| Close miss to flying duck (d=0.05), clean | -0.066 | 1.0 | — | +0.234 |
| Close miss to dead duck (d=0.05), clean | -0.300 | 1.0 | — | +0.000 |
| Far miss (d=0.5), clean | -0.275 | 1.0 | — | +0.025 |
| Miss, no tool call | -1.0 | 0.0 | — | -1.000 |
| Miss, verbose but parseable | -0.300 | 0.5 | — | -0.150 |
| Duck escaped during horizon | -0.700 | 1.0 | — | -0.400 |
| Double kill, h=0 | +2.500 | 1.0 | no | +2.800 |
| Double kill, h=0, **hotspot** | +2.500 | 1.0 | **yes** | **+1.400** |

### Key invariants

These must always hold for the reward system to provide correct learning signal:

```
double_kill > hit > close_miss > far_miss > shoot_nothing > shoot_dead > invalid
  +2.5      +1.0    ~-0.1       ~-0.3       -0.5           -0.7        -1.0

hit(h=30) > miss             →  lowest hit (0.9) still better than best miss (0.0)
novel_hit > hotspot_hit      →  diverse aiming rewarded over position spam
clean > verbose              →  format reward always higher for concise output
invalid < miss               →  always better to produce a parseable tool call
miss_near_flying > miss_near_dead  →  proximity only to alive ducks
```

## 5. How Rewards Drive GRPO Learning

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
| Different horizons on hits | h=3 (+0.99) vs h=20 (+0.93) = small but useful |
| Proximity bonus on misses (flying only) | close miss (-0.1) vs far miss (-0.28) = moderate signal |
| Aiming at flying vs dead duck | proximity bonus (-0.1) vs no bonus (-0.3) = teaches target selection |
| Format quality | clean (1.0) vs verbose (0.5) = moderate signal |
| Dead duck penalty | miss (-0.3) vs escaped (-0.7) = useful signal for horizon tuning |
| Hotspot penalty | novel hit (+1.0) vs repeated-position hit (+0.5) = discourages exploits |

### Common failure mode

If the model collapses to one output (e.g. always `shoot(x=0.5, y=0.5, horizon=8)`), all 6 generations get identical rewards. `std = 0`, advantages = 0, no gradient. The entropy mechanisms (see config) prevent this by encouraging diverse outputs. The hotspot penalty further discourages this by halving the reward for repeated positions.

## 6. Config Reference

```yaml
reward:
  hit: 1.0                  # base reward for hitting one duck
  double_kill: 2.5           # base reward for hitting both ducks
  miss: -0.3                 # base reward for missing
  shoot_nothing: -0.5        # shooting when no ducks flying
  shoot_dead_duck: -0.7      # ducks escaped during latency+horizon wait
  invalid_action: -1.0       # unparseable model output
  lambda_horizon: 0.1        # horizon penalty coefficient (on hits only)
  proximity_bonus: 0.3       # max proximity bonus (on misses only)
  proximity_decay: 5.0       # how fast proximity bonus decays with distance
  format_weight: 0.3         # weight for format reward in total
  accuracy_weight: 1.0       # weight for accuracy reward in total
  max_horizon: 30            # max horizon for penalty normalization
```

## 7. Tuning Guide

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Model never produces tool calls | `invalid_action` not negative enough, or format reward weight too low | Increase `format_weight` to 0.5 |
| Model always shoots center | Hitbox too large, or not enough game state diversity | Reduce hitbox, advance more frames between steps |
| Model uses horizon=0 always | `lambda_horizon` too high | Reduce to 0.05 |
| Model uses horizon=30 always | No penalty for dead ducks, or `lambda_horizon` too low | Check `shoot_dead_duck`, increase `lambda_horizon` to 0.15 |
| All generations get same reward | Model collapsed, or proximity bonus too small | Increase `proximity_bonus`, check entropy config |
| Hit rate plateau | Proximity decay too slow (bonus too easy) | Increase `proximity_decay` to 8.0 |
| Model writes explanations | Format verbosity penalty too weak | Reduce format floor from 0.3 to 0.1 |
| Model always shoots center | Hotspot penalty not triggered, or hitbox too large | Reduce `hotspot_radius`, reduce hitbox size |
| Model ignores dead ducks | Working correctly — proximity only targets flying ducks | No fix needed |
| Model shoots dead ducks | `shoot_dead_duck` penalty too weak | Increase to -0.9 |

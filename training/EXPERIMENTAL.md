# Experimental Reward Features

These features are implemented but experimental. Use them via CLI overrides.

---

## 1. Simplified Proximity (distance to hitbox center)

**What changed:** Previously there were 3 separate bonuses:
- `proximity_bonus` (misses only, distance to duck top-left)
- `edge_bonus` (hits only, distance from screen center)
- `hitbox_center_bonus` (hits only, distance to hitbox center)

**Now:** Single `proximity_bonus` that measures distance from shot to the **duck hitbox center**. Works for BOTH hits and misses. Closer to hitbox center = higher bonus.

```
duck_center = duck_position + (sprite_size / 2) in normalized coords
distance = euclidean_distance(shot, duck_center)
proximity_bonus = proximity_bonus_max × e^(-decay × distance)
```

**Why simpler is better:** One continuous gradient from anywhere on screen toward the duck center. No separate hit/miss logic. The model always gets signal pointing toward the duck.

**Config:**
```yaml
reward:
  proximity_bonus: 0.5       # max bonus at zero distance
  proximity_decay: 3.0       # how fast bonus drops with distance
```

Same config keys as before — just the computation changed internally.

---

## 2. Format Reward Decay

**Problem:** Once the model learns tool call format (~200 steps), `format_weight=0.3` adds a constant +0.3 to ALL generations. GRPO advantages are computed relative to group mean, so a constant adds zero signal. It's wasted.

**Solution:** Decay format_weight to 0 over N steps.

```
format_weight(step) = initial_weight × max(0, 1 - step / decay_steps)
```

| Step | format_weight (decay_steps=1000) |
|------|----------------------------------|
| 0 | 0.30 (full — model learning format) |
| 250 | 0.225 |
| 500 | 0.15 |
| 750 | 0.075 |
| 1000+ | 0.00 (pure accuracy signal) |

**Config:**
```yaml
reward:
  format_weight: 0.3
  format_decay_steps: 1000    # 0 = no decay (constant)
```

**CLI:**
```bash
--override reward.format_decay_steps=1000
```

**When to use:** When the model learns format quickly (LFM2-VL-3B, Qwen3-VL-8B) and you want training to focus entirely on aiming after the first ~1000 steps.

---

## 3. Reward Normalization Modes

Three modes for computing GRPO advantages:

### Mode: `group` (default)

Standard GRPO. Normalize rewards within the current step's generations only.

```
advantages = (rewards - mean(rewards)) / std(rewards)
```

**Pros:** Simple, proven.
**Cons:** No memory of past performance. Model at 50% hit rate gets same advantage scale as model at 5%.

### Mode: `moving_avg`

Compare rewards against an exponential moving average (EMA) of recent performance.

```
baseline = (1 - α) × baseline + α × mean(current_rewards)
advantages = (rewards - baseline) / std(rewards - baseline)
```

**How it works:**
- Baseline tracks the model's recent average reward
- A reward above baseline gets positive advantage
- As the model improves, baseline rises — the model must keep improving to get positive advantages
- Prevents plateau: the model can't rest on past performance

**Config:**
```yaml
reward:
  reward_normalization: "moving_avg"
  moving_avg_alpha: 0.01      # EMA smoothing (smaller = longer memory)
```

| alpha | Memory | Description |
|-------|--------|-------------|
| 0.1 | ~10 steps | Fast adaptation, noisy |
| 0.01 | ~100 steps | Smooth baseline, standard |
| 0.001 | ~1000 steps | Very stable, slow to adapt |

**When to use:** When hit rate plateaus at a fixed level. The moving average creates pressure to keep improving.

### Mode: `per_component`

Normalize accuracy and format rewards separately, then combine.

```
norm_accuracy = (accuracy - mean) / std    # across generations
norm_format = (format - mean) / std        # across generations

advantage = accuracy_weight × norm_accuracy + format_weight × norm_format
```

**How it works:**
- Each reward component is normalized to zero mean, unit variance independently
- A tiny proximity difference (0.01) gets the same importance as a big format difference (1.0)
- Prevents one component from dominating the gradient

**Example without per_component:**
```
Gen 0: accuracy=-0.10, format=1.0  →  total=0.90  →  advantage ≈ 0
Gen 1: accuracy=-0.15, format=1.0  →  total=0.85  →  advantage ≈ 0
(format dominates, accuracy difference invisible)
```

**Example with per_component:**
```
Gen 0: norm_acc=+1.0, norm_fmt=0.0  →  advantage = +1.0  (closer to duck!)
Gen 1: norm_acc=-1.0, norm_fmt=0.0  →  advantage = -1.0  (further from duck)
(accuracy difference fully visible)
```

**Config:**
```yaml
reward:
  reward_normalization: "per_component"
```

**When to use:** When format reward drowns out proximity signal. Especially useful early in training when some gens produce valid tool calls and some don't.

**Note:** Requires `reward_breakdowns` and `format_rewards` to be passed in the batch dict. Currently only works with per_component when these are available.

---

## CLI Examples

```bash
# Format decay: format reward fades to 0 over 1000 steps
--override reward.format_decay_steps=1000

# Moving average normalization
--override reward.reward_normalization=moving_avg
--override reward.moving_avg_alpha=0.01

# Per-component normalization
--override reward.reward_normalization=per_component

# Combine format decay with moving average
--override reward.format_decay_steps=1000 \
--override reward.reward_normalization=moving_avg

# Full example
accelerate launch --num_processes=2 --mixed_precision=bf16 \
    train.py --config configs/liquidai_3b_config.yaml --custom \
    --override training.max_steps=25000 \
    --override grpo.curriculum_phase2_step=15000 \
    --override reward.skip_no_target=true \
    --override reward.format_decay_steps=1000 \
    --override reward.reward_normalization=moving_avg \
    --override logging.wandb_run_name="lfm2-3b-moving-avg"
```

---

## What to Watch in W&B

| Metric | What it tells you |
|--------|-------------------|
| `train/mean_reward` | With moving_avg: should trend upward as baseline pushes performance |
| `train/std_reward` | With per_component: should be more consistent (no format dominance) |
| `train/advantages_std` | Higher = GRPO has more signal. Should be >0.5 for meaningful learning |
| `train/hit_rate` | The only metric that matters — are ducks being hit? |

---

## Status

| Feature | Implemented | Tested | Notes |
|---------|-------------|--------|-------|
| Simplified proximity (hitbox center) | Yes | No | Replaces edge_bonus + hitbox_center_bonus |
| Format decay | Yes | No | `format_decay_steps` config |
| Moving average normalization | Yes | No | `reward_normalization: moving_avg` |
| Per-component normalization | Yes | Partial | Needs format_rewards in batch |

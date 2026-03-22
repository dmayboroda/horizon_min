"""Duck Hunt OpenEnv Configuration"""

# SCREEN
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 500
FPS = 30

# TIMING
MATCH_DURATION_SECONDS = 30
MATCH_DURATION_FRAMES = MATCH_DURATION_SECONDS * FPS  # 900
ROUND_START_SECONDS = 2  # Auto-skip
DOG_SCENE_SECONDS = 2  # Auto-skip

# GAME STRUCTURE
MATCHES_PER_ROUND = 5
DUCKS_PER_MATCH = 2
BULLETS_PER_MATCH = 3
MAX_MISSES = 4  # Game over threshold

# DUCK
# Sprite dimensions (fixed — must match sprite sheet)
SPRITE_WIDTH = 81
SPRITE_HEIGHT = 75

# Hit detection box (centered on sprite)
HITBOX_WIDTH = 50   # ~6.25% of screen width
HITBOX_HEIGHT = 45  # ~9% of screen height
SPEED_BASE = 6  # Minimum speed component (was 4 — faster ducks)
SPEED_VARIANCE = 4  # Added to base (was 2 — wider speed range)
# Speed = BASE+round to BASE+VARIANCE+round
# Round 1: 7–11 px/frame (was 5–7)

# SPAWN (ducks always enter from off-screen edges)
SPAWN_Y_MIN_FRAC = 0.15  # normalised min Y for spawn height (higher spawn possible)
SPAWN_Y_MAX_FRAC = 0.85  # normalised max Y for spawn height (avoid dog area)

# MID-FLIGHT JITTER (makes trajectories less predictable)
JITTER_CHANCE = 0.03     # probability per frame of a direction nudge
JITTER_DX_RANGE = 1.0    # max dx perturbation (px/frame)
JITTER_DY_RANGE = 1.0    # max dy perturbation (px/frame)

# BOUNCE
BOUNCE_DY_MIN = -5       # was -4
BOUNCE_DY_MAX = 5        # was 4
BOUNCE_TOP_DY_MIN = 2
BOUNCE_TOP_DY_MAX = 5    # was 4
BOUNCE_BOTTOM_DY_MIN = -5  # was -4
BOUNCE_BOTTOM_DY_MAX = -2
BOUNCE_SPEED_JITTER = 0.15  # ±15% speed variation on bounce

# OPENENV SPECIFIC
FRAME_OUTPUT_SIZE = (512, 512)  # Resized for VLM
FRAMES_PER_OBSERVATION = 4  # Number of frames per observation (default)
FRAMES_PER_OBSERVATION_MIN = 2  # Minimum frames for random range
FRAMES_PER_OBSERVATION_MAX = 6  # Maximum frames for random range
FRAME_SKIP = 3  # Game frames to skip between observation frames
                 # e.g. skip=3 means capture every 3rd frame
                 # With 4 obs frames: spans 4*3=12 game frames of duck movement
MAX_HORIZON = 30  # Maximum horizon VLM can use
LATENCY_OPTIONS_MS = [50, 100, 200, 400, 600]

# REWARDS
REWARD_HIT = 1.0
REWARD_DOUBLE_KILL = 2.5
REWARD_MISS = -0.3
REWARD_SHOOT_NOTHING = -0.5  # Shot when no duck flying
LAMBDA_HORIZON = 0.1  # Horizon penalty weight
BONUS_PERFECT_MATCH = 0.5
BONUS_PERFECT_ROUND = 2.0

"""OpenAI Tool Schemas for Duck Hunt VLM Agent."""

# Tool schema for the shoot action
SHOOT_TOOL = {
    "type": "function",
    "function": {
        "name": "shoot",
        "description": "Fire at predicted duck position. Analyze the frames to identify flying ducks and predict where they will be after the specified horizon frames.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "integer",
                    "description": "Target x coordinate (0-800). Left edge is 0, right edge is 800.",
                    "minimum": 0,
                    "maximum": 800,
                },
                "y": {
                    "type": "integer",
                    "description": "Target y coordinate (0-500). Top edge is 0, bottom edge is 500. Ducks fly in the upper half (0-250).",
                    "minimum": 0,
                    "maximum": 500,
                },
                "horizon": {
                    "type": "integer",
                    "description": "Number of frames to wait before shooting (0-30). Use higher values if duck is moving fast and you need to lead the shot. Lower values reduce penalty but require more accurate current position.",
                    "minimum": 0,
                    "maximum": 30,
                },
                "confidence": {
                    "type": "string",
                    "description": "Your confidence level in this shot.",
                    "enum": ["high", "medium", "low"],
                },
            },
            "required": ["x", "y", "horizon"],
        },
    },
}

# List of all tools for the agent
DUCK_HUNT_TOOLS = [SHOOT_TOOL]

# System prompt for the VLM agent
SYSTEM_PROMPT = """You are a Duck Hunt game AI agent. Your goal is to shoot flying ducks.

GAME RULES:
- Two ducks fly around the screen at a time
- You have 3 bullets per match
- Ducks bounce off screen edges
- Ducks fly in the upper half of the screen (y: 0-250)
- Duck hitbox is approximately 81x75 pixels
- Match lasts 30 seconds

OBSERVATION:
- You receive 4 consecutive frames showing duck movement
- Use frame sequence to estimate duck velocity and direction
- Screen size: 800x500 pixels

STRATEGY:
- Identify duck positions in the frames
- Calculate duck velocity from frame-to-frame movement
- Predict future position based on horizon value
- Lead your shot ahead of the duck's current position
- Use lower horizon (0-5) for slow/stationary ducks
- Use higher horizon (10-20) for fast-moving ducks

ALWAYS call the shoot tool with your best prediction."""

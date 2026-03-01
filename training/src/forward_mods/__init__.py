"""Forward modification modules for VLA architecture changes."""

from .registry import apply_forward_mods, get_forward_mod

# Alias for backward compatibility
get_forward_mods = apply_forward_mods

__all__ = ["apply_forward_mods", "get_forward_mod", "get_forward_mods"]

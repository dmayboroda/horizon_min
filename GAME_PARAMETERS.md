# Game Parameters Reference

| Parameter | Value | File | Line |
|-----------|-------|------|------|
| Match duration | 10 seconds | `duckhunt/game/states.py` | 187 |
| Hitbox size | 81×75 | `duckhunt/game/duck.py` | 6 |
| Duck speed formula | `range(4+round, 6+round)` | `duckhunt/game/duck.py` | 123 |
| Bounce logic | Edge detection & direction change | `duckhunt/game/duck.py` | 141-175 |
| Ducks per match | 2 | `duckhunt/game/states.py` | 186, 291 |
| Bullets per match | 3 | `duckhunt/game/gun.py` | 7, 16 |
| Matches per round | 5 (10 hitDucks / 2 ducks each) | `duckhunt/game/states.py` | 46 |
| Game over condition | 4 misses | `duckhunt/game/states.py` | 311-312 |

## Detailed Code References

### Match Duration
```python
# duckhunt/game/states.py:187
self.roundTime = 30 # Seconds in a round
```

### Hitbox Size
```python
# duckhunt/game/duck.py:6
FRAME_SIZE = adjpos (81, 75)
```

### Duck Speed Formula
```python
# duckhunt/game/duck.py:123
speedRange = range(4+round, 6+round)
```

### Bounce Logic (Edge Behavior)
```python
# duckhunt/game/duck.py:141-175
# At the left side of the screen
if x <= 0:
    self.dx = random.choice(speedRange)
    self.dy = random.randint(-4, 4)

# At the right side of the screen
elif (x + frameWidth) > surface.get_width():
    self.dx = random.choice(speedRange) * -1
    self.dy = random.randint(-4, 4)

# At the top of the screen
elif y <= 0:
    self.dx = random.choice(speedRange) * coinToss
    self.dy = random.randint(2, 4)

# At the bottom of the screen
elif y > (surface.get_height() / 2):
    self.dx = random.choice(speedRange) * coinToss
    self.dy = random.randint(-4, -2)
```

### Ducks Per Match
```python
# duckhunt/game/states.py:186
self.ducks = [Duck(self.registry), Duck(self.registry)]

# duckhunt/game/states.py:291
self.ducks = [Duck(self.registry), Duck(self.registry)]
```

### Bullets Per Match
```python
# duckhunt/game/gun.py:7
self.rounds = 3

# duckhunt/game/gun.py:16 (reload)
self.rounds = 3
```

### Matches Per Round
```python
# duckhunt/game/states.py:46
self.hitDucks = [False for i in range(10)]
# 10 duck slots / 2 ducks per match = 5 matches per round
```

### Game Over Condition
```python
# duckhunt/game/states.py:311-312
# Miss 4 or more and you're done
if missedCount >= 4:
    self.isGameOver = True
```

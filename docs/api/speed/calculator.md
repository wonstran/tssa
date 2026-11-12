# Speed (`tssa.speed`)

`SpeedCalculator` turns smoothed ground positions into velocity and speed, at every frame or over blocks of N frames.

```python
from tssa.speed import SpeedCalculator

calc = SpeedCalculator(fps=10, xy_cols=("x_smooth", "y_smooth"))   # mph; units="metric" gives m/s
out = calc.calculate(smoothed)                                    # adds vx, vy, speed
```

::: tssa.speed.calculator

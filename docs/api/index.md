# API Reference

The pages in this section are rendered from module docstrings and object signatures.

| Section | Modules |
|---|---|
| [Vehicle Footprints](footprint/index.md) | `tssa.footprint` and its locators, detectors, masks and geometry |
| [Smoothing](smooth/index.md) | `tssa.smooth` |
| [Speed](speed/calculator.md) | `tssa.speed` |
| [Transform](transform/index.md) | `tssa.transform` |
| [Engine](engine/index.md) | `tssa.engine` |
| [Analyses](core/index.md) | `tssa.core` |
| [Annotation](annotation.md), [Track Files](io.md) | `tssa.annotation`, `tssa.io` |

Most users need only the classes exported at package level:

```python
from tssa.footprint import VehFootPrinter, FootprintConfig, M7Config, detect, zone_from_points
from tssa.smooth import TrackSmoother, SmoothConfig
from tssa.speed import SpeedCalculator
from tssa.transform import HomographyTransform, DistortedHomography, GroundModel, PinholeCamera, kmz_to_csv
```

# Detection

The detectors need the optional install `pip install tssa[detect]`. `run()` reads the video once and runs every
detector on the frames it has rows for. Its output tables go straight into `VehFootPrinter.estimate(tracks, **tables)`.

::: tssa.footprint.detect.runner

::: tssa.footprint.detect.detectors

::: tssa.footprint.match

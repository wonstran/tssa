# Analyses (`tssa.core`)

The earlier analysis modules of tssa: counting tracks across a line, assigning travel lanes, speed meters, and
scanning track pairs for interactions, near misses and yielding.

!!! note
    These modules predate `tssa.footprint`, `tssa.smooth` and `tssa.speed`, and are documented as they are. `speed`,
    `interact`, `nearmiss` and `yield` import from [dnt](https://wonstran.github.io/dnt/) (its tracker and
    labeler), and the last three from `shapelysmooth`. Neither is a dependency of tssa; install them to use those
    modules.
    `yield` is a Python keyword, so load that module with `importlib.import_module("tssa.core.yield")`.

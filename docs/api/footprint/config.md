# Configuration

`FootprintConfig` holds every setting of the printer; `M7Config` holds the settings of the M7 solver and sits in
`FootprintConfig.m7`. Both are frozen: change them with `with_overrides(...)`, or pass the overrides to
`VehFootPrinter(...)` as keyword arguments.

::: tssa.footprint.config

::: tssa.footprint.m7config

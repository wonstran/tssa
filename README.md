# tssa - Transportation Surrogate Safety Analysis Tools

TSSA is a set of tools designed for conducting transportation sorrogate safety studies, specifically focusing on pedestrian and bicycle safety. It leverages AI and machine learning to analyze and interpret transportation data, providing insights and recommendations for improving safety in urban environments.

**Documentation:** [tssa API Manual](https://wonstran.github.io/tssa/)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Modeling and Analysis](#modeling-and-analysis)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Data Collection**: Collect data from various sources like traffic cameras, sensors, and public datasets.
- **Data Analysis**: Use AI/ML algorithms to analyze trends and identify safety risks for pedestrians and cyclists.
- **Visualization**: Generate heatmaps and graphs for easy interpretation of data.
- **Predictive Modeling**: Predict future accident hotspots and suggest preventive measures.
- **Recommendations**: Provide actionable recommendations based on data analysis to improve pedestrian and bicycle safety.

## Installation

To get started with TSSA, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/tssa.git
    cd tssa
    ```

2. Install the package and required dependencies:
    ```bash
    pip install .
    ```

3. Set up your environment variables (e.g., API keys, database credentials) in a `.env` file.

4. Run the application:
    ```bash
    python app.py
    ```

## Usage

After installation, you can run the application locally to:

1. Collect data from transportation systems and traffic sensors.
2. Run AI models to analyze transportation safety data.
3. Visualize results in a web interface or save them for further reporting.

Sample usage:

```bash
python analysis.py --data=data/traffic_data.csv --output=output/safety_report.pdf
```

## Vehicle footprints (`tssa.footprint`)

`VehFootPrinter` turns tracked vehicle boxes into ground footprints: a reference point at the body centre before
smoothing, then an oriented polygon of the right size after smoothing. It carries the seven methods of the speed
study (M1 bbox bottom-centre ... M7 wheel-anchored rigid fit), per-class fallback chains, and a 2 x 2 measurement
covariance per row. Design: [docs/2026-11-10-footprint-design.md](docs/2026-11-10-footprint-design.md). API:
[VehFootPrinter reference](https://wonstran.github.io/tssa/api/footprint/).

```python
import pandas as pd
from tssa.footprint import VehFootPrinter, detect, zone_from_points
from tssa.transform import DistortedHomography

dh = DistortedHomography.load_json("model_a1a_nw.json")            # no OpenCV needed to load and project
tracks = pd.read_csv("veh_tracks.csv")                              # frame, track, x, y, w, h, cls (pixels)
gcps = pd.read_csv("gcps.csv")[["utm_e_m", "utm_n_m"]].to_numpy()
vfp = VehFootPrinter(dh, zone=zone_from_points(gcps, buffer_ft=15), workers=8, cache_dir="cache")

# optional: run the detectors (pip install tssa[detect]) for the rows the default chains need
tables = detect.run("clip.mp4", tracks,
                    [detect.KeypointDetector(), detect.CuboidDetector("uod.pt", "path/to/urbanomnidetect"),
                     detect.MaskSegmenter("sam2.1_b.pt")],
                    rows=vfp.required_rows(tracks), cache_dir="cache")

out = vfp.estimate(tracks, **tables)
gdf = vfp.to_geodataframe(out, crs="EPSG:32617")                    # one polygon per row, for PET
```

- Default chains: cars M6 then M2; buses M7, M6, M2; trucks M7, M6, M3, M2. Without detections they reduce to M2
  (M3 then M2 for trucks).
- `compare()` runs each method alone on one grid (native-only with `fallback=None`).
- Methods can also be named `bottom_centre`, `nominal_shift`, `nominal_box`, `wheel_keypoints`, `sam_outline`,
  `cuboid` and `wheel_fit` (M1 to M7), e.g. `chains={2: ("cuboid", "nominal_shift")}`; outputs keep the M names.
- M7 is the slow method: about a minute per bus or truck segment on one core. Use `workers` (each worker gets one
  BLAS thread; more workers than physical cores slows every solve) and `cache_dir` (solves are cached).
- Sizes and distance settings are in feet (`FootprintConfig`); positions stay in the homography's metres; speeds
  in mph unless you pass a metric `SpeedCalculator`.

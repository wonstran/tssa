# Transform (`tssa.transform`)

Mapping between image pixels and ground-plane world coordinates (metres, for example UTM).

| Class | What it is |
|---|---|
| `HomographyTransform` | A 3 x 3 ground-plane homography fitted from control points |
| `DistortedHomography` | Radial lens distortion followed by a homography; the usual model for roadside cameras |
| `GroundModel` | One interface over either, with explicit distortion and image size; used by `tssa.footprint` |
| `PinholeCamera` | An approximate pinhole camera fitted to a ground model, for methods that need heights (M4, M7) |

Fitting needs OpenCV. Loading a saved model (`load_json`, `load_matrix`) and projecting with it use NumPy only.

```python
from tssa.transform import DistortedHomography

dh = DistortedHomography.load_json("model.json")
world = dh.image_to_world(pixels)          # (N, 2) distorted pixels -> (N, 2) metres
pixels = dh.world_to_image(world)
```

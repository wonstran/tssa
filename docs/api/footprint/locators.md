# Locators

A locator turns rows (box, class and any detection tables) into reference points. The seven built-in locators are
registered under their M names and a descriptive alias; your own are added with `register_locator`.

| Method | Alias | Needs | Scope | Classes |
|---|---|---|---|---|
| M1 | `bottom_centre` | - | row (point only) | all |
| M2 | `nominal_shift` | - | row | all |
| M3 | `nominal_box` | - | row | all |
| M4 | `wheel_keypoints` | `keypoints`, camera | row | all |
| M5 | `sam_outline` | `masks` | row | all |
| M6 | `cuboid` | `cuboids` | row | all |
| M7 | `wheel_fit` | `masks`, camera | segment | 5 (bus), 7 (truck) |

::: tssa.footprint.locators

::: tssa.footprint.locators.m1_m2_m3

::: tssa.footprint.locators.m4

::: tssa.footprint.locators.m5

::: tssa.footprint.locators.m6

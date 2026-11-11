"""The speed study's affine stand-in for a ground model (identity distortion)."""

import numpy as np


class AffineGround:
    image_size = (640, 480)

    def __init__(self, A=((0.05, 0.01), (0.0, -0.08)), b=(580000.0, 2852000.0)):
        self.A, self.b = np.asarray(A, float), np.asarray(b, float)

    def image_to_world(self, uv):
        return np.asarray(uv, float) @ self.A.T + self.b

    def world_to_image(self, xy):
        return (np.asarray(xy, float) - self.b) @ np.linalg.inv(self.A).T

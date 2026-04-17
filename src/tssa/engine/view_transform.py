"""
This script is used to transform image coordinates to real-world coordinates and vice versa.
updates:
    2024-10-13
"""

import cv2
import numpy as np


class ViewTransform:
    """
    A class used to transform image coordinates to real-world coordinates and vice versa.
    """

    def __init__(self, source: np.ndarray, destination: np.ndarray):
        """
        Initializes the ViewTransform object. \n
        Args:
            source (numpy.ndarray): The source points (shape: (4, 2))
            destination (numpy.ndarray): The destination points (shape: (4, 2))
        """
        source = np.float32(source)
        destination = np.float32(destination)
        self.M = cv2.getPerspectiveTransform(source, destination)
        self.M_inv = cv2.getPerspectiveTransform(destination, source)

    def transform(self, image_points: np.ndarray) -> np.ndarray:
        """
        Transforms an image coordinate to a real-world coordinate.\n

        Args:\n
            image_points (numpy.ndarray): The image coordinates (shape: (2,))

        Returns:\n
            numpy.ndarray: The real-world coordinates (shape: (2,))
        """
        if image_points.ndim == 1:
            image_points = np.array([image_points])

        reshaped_points = image_points.reshape(-1, 1, 2).astype(np.float32)
        world_points = cv2.perspectiveTransform(reshaped_points, self.M)

        return world_points.reshape(-1, 2).astype(np.int32)

    def transform_inv(self, world_points: np.ndarray) -> np.ndarray:
        """
        Transforms a real-world coordinate to an image coordinate.\n

        Args:\n
            world_points (numpy.ndarray): The real-world coordinates (shape: (2,))

        Returns:\n
            numpy.ndarray: The image coordinates (shape: (2,))
        """
        if world_points.ndim == 1:
            world_points = np.array([world_points])

        reshaped_points = world_points.reshape(-1, 1, 2).astype(np.float32)
        image_points = cv2.perspectiveTransform(reshaped_points, self.M_inv)

        return image_points.reshape(-1, 2).astype(np.int32)

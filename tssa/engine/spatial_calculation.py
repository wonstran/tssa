'''
This script is used to calculate spatial relationships
updates: 
    2021-09-30
'''
import numpy as np

def point_relative_to_line(point:np.array, line_start:np.array, line_end:np.array) -> int:
    """
    Determines the position of a point relative to a line.

    Args:
        point (numpy.ndarray): The point in question (shape: (2,))
        line_start (numpy.ndarray): The starting point of the line (shape: (2,))
        line_end (numpy.ndarray): The ending point of the line (shape: (2,))

    Returns:
        str: 'left-up', 'right-down', or 'on'
    """

    # Calculate the cross product of the vectors from the line start to the point and the line end
    cross_product = np.cross(line_end - line_start, point - line_start)

    result = 'on'
    if cross_product > 0:
        result = 'right-down'
    elif cross_product < 0:
        result = 'left-up'

    return result

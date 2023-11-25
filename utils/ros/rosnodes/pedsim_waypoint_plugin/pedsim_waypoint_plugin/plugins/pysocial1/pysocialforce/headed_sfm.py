"""Filters the forces through implementation of Headed Social Force Model"""
import numpy as np

from pysocialforce.potentials import PedPedPotential, PedSpacePotential
from pysocialforce.fieldofview import FieldOfView
from pysocialforce.utils import Config, stateutils, logger


def heading(velocities: np.ndarray) -> np.ndarray:
    """
    Calculates the heading angle or direction in 
    relation to the global reference frame given 
    the y- and x-component of the velocity.
    """
    return stateutils.vector_angles(velocities)

def angular_vel(accelerations: np.ndarray) -> np.ndarray:
    """
    Calculates the angular velocity or torque 
    given the current force/acceleration.
    """
    return stateutils.vector_angles(accelerations)

def rotate_2d_vector(vector: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotates the vectors in the nx2 np array by 
    <theta>. theta should be given in radian
    format. The rotation is clock-wise
    """
    theta *= -1
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    return (rotation_matrix * vector.transpose()).transpose()


    

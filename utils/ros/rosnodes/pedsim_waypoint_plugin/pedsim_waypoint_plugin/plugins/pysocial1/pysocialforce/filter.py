"""Filters the forces through implementation of Walking Ahead (HSFM)"""
import re
from abc import ABC, abstractmethod

import numpy as np

from pysocialforce.potentials import PedPedPotential, PedSpacePotential
from pysocialforce.fieldofview import FieldOfView
from pysocialforce.utils import Config, stateutils, logg


def heading(v_y, v_x):
    """
    Calculates the heading angle or direction in 
    relation to the global reference frame given 
    the y- and x-component of the velocity.
    """
    return np.arctan2(v_y, v_x)

def angular_vel(a_y, a_x):
    """
    Calculates the angular velocity or torque 
    given the current force.
    """
    return np.arctan2(a_y, a_x)


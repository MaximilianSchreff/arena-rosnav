"""Filters the forces through implementation of Headed Social Force Model"""
import numpy as np
from typing import Tuple, Optional

from pysocialforce.utils import Config, stateutils, logger


def heading(velocities: np.ndarray) -> np.ndarray:
    """Calculates the heading angle or direction in 
    relation to the global reference frame.

    Args:
        velocities (np.ndarray): The velocities
            of each ped with x-component in 
            column 1 and y-component in columns
            2 of shape n x 2.

    Returns:
        np.ndarray: The heading angles in radian 
            format of shape n x 1.
    """
    return stateutils.vector_angles(velocities)[:, np.newaxis]


def torque_and_heading(accelerations: np.ndarray, velocities: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the angular velocity.

    Args:
        accelerations (np.ndarray): accelerations of
            each ped of shape n x 2.
        velocities (np.ndarray): velocities of each
            ped of shape n x 2.

    Returns:
        np.ndarray: The angular velocity of each
            ped of shape n x 1.
        np.ndarray: The heading of each ped of
            shape n x 1.
    """
    new_headings = heading(velocities)
    old_headings = heading(velocities - accelerations)
    torq = new_headings - old_headings
    return torq, new_headings


def rotation_matrix(theta: np.ndarray) -> np.ndarray:
    """Calculates the rotation matrix for each angle
    in theta. Angles should be in radian format.

    Args:
        theta (np.ndarray): Vector of angles per ped
            of shape n x 1.

    Returns:
        np.ndarray: Tensor of rotation matrices per
            ped of shape n x 2 x 2.
    """
    mat = np.array(
        [[
            [np.cos(theta[i]), -np.sin(theta[i])],
            [np.sin(theta[i]), np.cos(theta[i])]
        ] for i in range(theta.shape[0])]
    )
    return np.squeeze(mat, axis=3)

def rotate_2d_vectors(vectors: np.ndarray, theta: np.ndarray, 
                      rotation: Optional[np.ndarray] = None
                      ) -> np.ndarray:
    """Rotates the 2d vector in row i of the n x 2 np 
    array by degree theta_i. theta should be given
    in radian format. The rotation is counter clock-wise.

    Args:
        vectors (np.ndarray): vectors with row i
            being a 2d vector with [x, y] of shape
            n x 2.
        theta (np.ndarray): Vector where row i
            contains angle i by which 2d vector
            of row i should be rotated of shape
            n x 1.
        rotation (np.ndarray): Optional rotation
            matrix if already computed of shape 
            n x 2 x 2.

    Returns:
        np.ndarray: Rotated vectors with same 
            format as vectors.
    """
    if rotation is None:
        rotation = rotation_matrix(theta)
    rotated = rotation @ vectors[:, :, np.newaxis]
    rotated = np.squeeze(rotated, axis=2)
    
    return rotated


def filter_forces(velocities: np.ndarray,
                  old_force: np.ndarray,
                  desired_force: np.ndarray,
                  obstacle_force: np.ndarray,
                  social_force: np.ndarray,
                  group_force: np.ndarray,
                  factor_orth: float,
                  torq_lambda: float,
                  torq_alpha: float,
                  inertia: float
                  ) -> np.ndarray:
    """Filters the forces computed by the improved 
    SFM according to the Headed Social Force Model 
    introduced in 'Walking Ahead: The Headed Social
    Force Model' by Farina et al..
    This is supposed to improve the walking dynamic
    of a single ped by smoothing their acceleration
    direction and comprimising their range of 
    walking direction.

    Args:
        velocities (np.ndarray): Current linear part
            of the velocities per ped of shape n x 2.
        old_force (np.ndarray): Last active forces
            or acceleration per ped of shape n x 2.
        desired_force (np.ndarray): Goal attractive 
            force of shape n x 2.
        obstacle_force (np.ndarray): Obstacle 
            avoidant force of shape n x 2.
        social_force (np.ndarray): Social repulsive
            force of shape n x 2.
        group_force (np.ndarray): Group forces of
            shape n x 2.
        factor_orth (float): Factor for the force 
            along the orthogonal walking direction.
        torq_lambda (float): Factor >0 used to tune 
            the dominant time constant of the 
            torque.
        torq_alpha (float): Factor >1 for pole 
            creation for computation of torque.
        inertia (float): Value of moment of inertia 
            used for every ped.

    Returns:
        np.ndarray: Smoothened/Filtered force of 
            shape n x 2.
    """
    obs_force = obstacle_force + social_force
    # walking direaction of each ped and torque
    angular_vel, heads = torque_and_heading(old_force, velocities)
    # tensor of rotation matrices for each walking direction
    rotation = rotation_matrix(heads) 
    
    # project overall force along walking direction of ped
    force_forw = np.sum((obs_force + desired_force) * rotation[:, :, 0], axis=1, keepdims=True)
    # project obstacle and ped force along orthogonal walking direction
    force_orth = factor_orth * np.sum(obs_force * rotation[:, :, 1], axis=1, keepdims=True)

    # calculate torque
    force_magnitude = np.linalg.norm(desired_force, axis=1, keepdims=True)
    force_phase = heading(desired_force)
    factor_head = -inertia * torq_lambda * force_magnitude  # shape n x 1
    factor_torq = -inertia * (1 + torq_alpha) * np.sqrt((torq_lambda / torq_alpha) * force_magnitude)  # shape n x 1
    angular_force = factor_head * (heads - force_phase) + factor_torq * angular_vel
    print(angular_vel)
    print(angular_force)
    
    # translate body frame force to global reference frame
    force = rotate_2d_vectors(np.hstack([force_forw, force_orth]), heads, rotation)
    # apply angular force to overall force
    force = rotate_2d_vectors(force, angular_force)
    return force

from typing import Any, Callable, Dict
from warnings import warn

import numpy as np

from ..constants import REWARD_CONSTANTS, DEFAULTS
from ..reward_function import RewardFunction
from .base_reward_units import RewardUnit, GlobalplanRewardUnit
from .reward_unit_factory import RewardUnitFactory
from ..utils import check_params

# UPDATE WHEN ADDING A NEW UNIT
__all__ = [
    "RewardGoalReached",
    "RewardSafeDistance",
    "RewardNoMovement",
    "RewardApproachGoal",
    "RewardCollision",
    "RewardDistanceTravelled",
    "RewardApproachGlobalplan",
    "RewardFollowGlobalplan",
    "RewardReverseDrive",
    "RewardAbruptVelocityChange",
]


@RewardUnitFactory.register("goal_reached")
class RewardGoalReached(RewardUnit):
    DONE_INFO = {"is_done": True, "done_reason": 2, "is_success": True}
    NOT_DONE_INFO = {"is_done": False}

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.GOAL_REACHED.REWARD,
        _on_safe_dist_violation: bool = DEFAULTS.GOAL_REACHED._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward when the goal is reached.

        Args:
            reward_function (RewardFunction): The reward function object holding this unit.
            reward (float, optional): The reward value for reaching the goal. Defaults to DEFAULTS.GOAL_REACHED.REWARD.
            _on_safe_dist_violation (bool, optional): Flag to indicate if there is a violation of safe distance. Defaults to DEFAULTS.GOAL_REACHED._ON_SAFE_DIST_VIOLATION.
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._reward = reward
        self._goal_radius = self._reward_function.goal_radius

    def check_parameters(self, *args, **kwargs):
        if self._reward < 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Negative rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def __call__(self, distance_to_goal: float, *args: Any, **kwargs: Any) -> None:
        """Calculates the reward and updates the information when the goal is reached.

        Args:
            distance_to_goal (float): Distance to the goal in m.
        """
        if distance_to_goal < self._reward_function.goal_radius:
            self.add_reward(self._reward)
            self.add_info(RewardGoalReached.DONE_INFO)
        else:
            self.add_info(RewardGoalReached.NOT_DONE_INFO)

    def reset(self):
        self._goal_radius = self._reward_function.goal_radius


@RewardUnitFactory.register("safe_distance")
class RewardSafeDistance(RewardUnit):
    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.SAFE_DISTANCE.REWARD,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward when violating the safe distance.

        Args:
            reward_function (RewardFunction): The reward function object.
            reward (float, optional): The reward value for violating the safe distance. Defaults to DEFAULTS.SAFE_DISTANCE.REWARD.
        """
        super().__init__(reward_function, True, *args, **kwargs)
        self._reward = reward
        self._safe_dist = self._reward_function._safe_dist

    def check_parameters(self, *args, **kwargs):
        if self._reward > 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def __call__(self, laser_scan: np.ndarray, *args: Any, **kwargs: Any):
        violation_in_blind_spot = False
        if "full_laser_scan" in kwargs:
            violation_in_blind_spot = kwargs["full_laser_scan"].min() <= self._safe_dist

        if laser_scan.min() < self._safe_dist or violation_in_blind_spot:
            self.add_reward(self._reward)
            self.add_info({"safe_dist_violation": True})


@RewardUnitFactory.register("no_movement")
class RewardNoMovement(RewardUnit):
    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.NO_MOVEMENT.REWARD,
        _on_safe_dist_violation: bool = DEFAULTS.NO_MOVEMENT._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward when there is no movement.

        Args:
            reward_function (RewardFunction): The reward function object.
            reward (float, optional): The reward value for no movement. Defaults to DEFAULTS.NO_MOVEMENT.REWARD.
            _on_safe_dist_violation (bool, optional): Flag to indicate if there is a violation of safe distance. Defaults to DEFAULTS.NO_MOVEMENT._ON_SAFE_DIST_VIOLATION.
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._reward = reward

    def check_parameters(self, *args, **kwargs):
        if self._reward > 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def __call__(self, action: np.ndarray, *args: Any, **kwargs: Any):
        if (
            action is not None
            and abs(action[0]) <= REWARD_CONSTANTS.NO_MOVEMENT_TOLERANCE
        ):
            self.add_reward(self._reward)


@RewardUnitFactory.register("approach_goal")
class RewardApproachGoal(RewardUnit):
    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        pos_factor: float = DEFAULTS.APPROACH_GOAL.POS_FACTOR,
        neg_factor: float = DEFAULTS.APPROACH_GOAL.NEG_FACTOR,
        _on_safe_dist_violation: bool = DEFAULTS.APPROACH_GOAL._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward when approaching the goal.

        Args:
            reward_function (RewardFunction): The reward function object.
            pos_factor (float, optional): Positive factor for approaching the goal. Defaults to DEFAULTS.APPROACH_GOAL.POS_FACTOR.
            neg_factor (float, optional): Negative factor for distancing from the goal. Defaults to DEFAULTS.APPROACH_GOAL.NEG_FACTOR.
            _on_safe_dist_violation (bool, optional): Flag to indicate if there is a violation of safe distance. Defaults to DEFAULTS.APPROACH_GOAL._ON_SAFE_DIST_VIOLATION.
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._pos_factor = pos_factor
        self._neg_factor = neg_factor
        self.last_goal_dist = None

    def check_parameters(self, *args, **kwargs):
        if self._pos_factor < 0 or self._neg_factor < 0:
            warn_msg = (
                f"[{self.__class__.__name__}] Both factors should be positive. "
                f"Current values: [pos_factor={self._pos_factor}], [neg_factor={self._neg_factor}]"
            )
            warn(warn_msg)
        if self._pos_factor >= self._neg_factor:
            warn_msg = (
                "'pos_factor' should be smaller than 'neg_factor' otherwise rotary trajectories will get rewarded. "
                f"Current values: [pos_factor={self._pos_factor}], [neg_factor={self._neg_factor}]"
            )
            warn(warn_msg)

    def __call__(self, distance_to_goal, *args, **kwargs):
        if self.last_goal_dist is not None:
            w = (
                self._pos_factor
                if (self.last_goal_dist - distance_to_goal) > 0
                else self._neg_factor
            )
            self.add_reward(w * (self.last_goal_dist - distance_to_goal))
        self.last_goal_dist = distance_to_goal

    def reset(self):
        self.last_goal_dist = None


@RewardUnitFactory.register("collision")
class RewardCollision(RewardUnit):
    DONE_INFO = {"is_done": True, "done_reason": 1, "is_success": False}

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.COLLISION.REWARD,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward when a collision is detected.

        Args:
            reward_function (RewardFunction): The reward function object.
            reward (float, optional): The reward value for reaching the goal. Defaults to DEFAULTS.COLLISION.REWARD.
        """
        super().__init__(reward_function, True, *args, **kwargs)
        self._reward = reward

    def check_parameters(self, *args, **kwargs):
        if self._reward > 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def __call__(self, laser_scan: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        coll_in_blind_spots = False
        if "full_laser_scan" in kwargs:
            coll_in_blind_spots = kwargs["full_laser_scan"].min() <= self.robot_radius

        if laser_scan.min() <= self.robot_radius or coll_in_blind_spots:
            self.add_reward(self._reward)
            self.add_info(RewardCollision.DONE_INFO)


@RewardUnitFactory.register("distance_travelled")
class RewardDistanceTravelled(RewardUnit):
    def __init__(
        self,
        reward_function: RewardFunction,
        consumption_factor: float = DEFAULTS.DISTANCE_TRAVELLED.CONSUMPTION_FACTOR,
        lin_vel_scalar: float = DEFAULTS.DISTANCE_TRAVELLED.LIN_VEL_SCALAR,
        ang_vel_scalar: float = DEFAULTS.DISTANCE_TRAVELLED.ANG_VEL_SCALAR,
        _on_safe_dist_violation: bool = DEFAULTS.DISTANCE_TRAVELLED._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward for the distance travelled.

        Args:
            reward_function (RewardFunction): The reward function object.
            consumption_factor (float, optional): Negative consumption factor. Defaults to DEFAULTS.DISTANCE_TRAVELLED.CONSUMPTION_FACTOR.
            lin_vel_scalar (float, optional): Scalar for the linear velocity. Defaults to DEFAULTS.DISTANCE_TRAVELLED.LIN_VEL_SCALAR.
            ang_vel_scalar (float, optional): Scalar for the angular velocity. Defaults to DEFAULTS.DISTANCE_TRAVELLED.ANG_VEL_SCALAR.
            _on_safe_dist_violation (bool, optional): Flag to indicate if there is a violation of safe distance. Defaults to DEFAULTS.DISTANCE_TRAVELLED._ON_SAFE_DIST_VIOLATION.
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._factor = consumption_factor
        self._lin_vel_scalar = lin_vel_scalar
        self._ang_vel_scalar = ang_vel_scalar

    def __call__(self, action: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        if action is None:
            return
        lin_vel, ang_vel = action[0], action[-1]
        reward = (
            (lin_vel * self._lin_vel_scalar) + (ang_vel * self._ang_vel_scalar)
        ) * -self._factor
        self.add_reward(reward)


@RewardUnitFactory.register("approach_globalplan")
class RewardApproachGlobalplan(GlobalplanRewardUnit):
    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        pos_factor: float = DEFAULTS.APPROACH_GLOBALPLAN.POS_FACTOR,
        neg_factor: float = DEFAULTS.APPROACH_GLOBALPLAN.NEG_FACTOR,
        _on_safe_dist_violation: bool = DEFAULTS.APPROACH_GLOBALPLAN._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward for approaching the global plan.

        Args:
            reward_function (RewardFunction): The reward function object.
            pos_factor (float, optional): Positive factor for approaching the goal. Defaults to DEFAULTS.APPROACH_GLOBALPLAN.POS_FACTOR.
            neg_factor (float, optional): Negative factor for distancing from the goal. Defaults to DEFAULTS.APPROACH_GLOBALPLAN.NEG_FACTOR.
            _on_safe_dist_violation (bool, optional): Flag to indicate if there is a violation of safe distance. Defaults to DEFAULTS.APPROACH_GLOBALPLAN._ON_SAFE_DIST_VIOLATION.
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._pos_factor = pos_factor
        self._neg_factor = neg_factor

        self.last_dist_to_path = None
        self._kdtree = None

    def check_parameters(self, *args, **kwargs):
        if self._pos_factor < 0 or self._neg_factor < 0:
            warn_msg = (
                f"[{self.__class__.__name__}] Both factors should be positive. "
                f"Current values: [pos_factor={self._pos_factor}], [neg_factor={self._neg_factor}]"
            )
            warn(warn_msg)
        if self._pos_factor >= self._neg_factor:
            warn_msg = (
                "'pos_factor' should be smaller than 'neg_factor' otherwise rotary trajectories will get rewarded. "
                f"Current values: [pos_factor={self._pos_factor}], [neg_factor={self._neg_factor}]"
            )
            warn(warn_msg)

    def __call__(
        self, global_plan: np.ndarray, robot_pose, *args: Any, **kwargs: Any
    ) -> Any:
        super().__call__(global_plan=global_plan, robot_pose=robot_pose)

        if self.curr_dist_to_path and self.last_dist_to_path:
            self.add_reward(self._calc_reward())

        self.last_dist_to_path = self.curr_dist_to_path

    def _calc_reward(self) -> float:
        w = (
            self._pos_factor
            if self.curr_dist_to_path < self.last_dist_to_path
            else self._neg_factor
        )
        return w * (self.last_dist_to_path - self.curr_dist_to_path)

    def reset(self):
        super().reset()
        self.last_dist_to_path = None


@RewardUnitFactory.register("follow_globalplan")
class RewardFollowGlobalplan(GlobalplanRewardUnit):
    def __init__(
        self,
        reward_function: RewardFunction,
        min_dist_to_path: float = DEFAULTS.FOLLOW_GLOBALPLAN.MIN_DIST_TO_PATH,
        reward_factor: float = DEFAULTS.FOLLOW_GLOBALPLAN.REWARD_FACTOR,
        _on_safe_dist_violation: bool = DEFAULTS.FOLLOW_GLOBALPLAN._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        """Class for calculating the reward for following the global plan.

        Args:
            reward_function (RewardFunction): The reward function object.
            min_dist_to_path (float, optional): Minimum distance for reward application. Defaults to DEFAULTS.FOLLOW_GLOBALPLAN.MIN_DIST_TO_PATH.
            reward_factor (float, optional): Reward factor to be multiplied with the linear velocity. Defaults to DEFAULTS.FOLLOW_GLOBALPLAN.REWARD_FACTOR.
            _on_safe_dist_violation (bool, optional): Flag to indicate if there is a violation of safe distance. Defaults to DEFAULTS.FOLLOW_GLOBALPLAN._ON_SAFE_DIST_VIOLATION.
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._min_dist_to_path = min_dist_to_path
        self._reward_factor = reward_factor

    def __call__(
        self,
        action: np.ndarray,
        global_plan: np.ndarray,
        robot_pose,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        super().__call__(global_plan=global_plan, robot_pose=robot_pose)

        if (
            self.curr_dist_to_path
            and action is not None
            and self.curr_dist_to_path <= self._min_dist_to_path
        ):
            self.add_reward(self._reward_factor * action[0])


@RewardUnitFactory.register("reverse_drive")
class RewardReverseDrive(RewardUnit):
    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.REVERSE_DRIVE.REWARD,
        _on_safe_dist_violation: bool = DEFAULTS.REVERSE_DRIVE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        """Class for calculating the reward when reversing.

        Args:
            reward_function (RewardFunction): The reward function object.
            reward (float, optional): The reward value for reversing. Defaults to DEFAULTS.REVERSE_DRIVE.REWARD.
            _on_safe_dist_violation (bool, optional): Flag to indicate if there is a violation of safe distance. Defaults to DEFAULTS.REVERSE_DRIVE._ON_SAFE_DIST_VIOLATION.
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._reward = reward

    def check_parameters(self, *args, **kwargs):
        if self._reward > 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def __call__(self, action: np.ndarray, *args, **kwargs):
        if action is not None and action[0] < 0:
            self.add_reward(self._reward)


@RewardUnitFactory.register("abrupt_velocity_change")
class RewardAbruptVelocityChange(RewardUnit):
    def __init__(
        self,
        reward_function: RewardFunction,
        vel_factors: Dict[str, float] = DEFAULTS.ABRUPT_VEL_CHANGE.VEL_FACTORS,
        _on_safe_dist_violation: bool = DEFAULTS.ABRUPT_VEL_CHANGE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        """Class for calculating the reward for abrupt velocity change.

        Args:
            reward_function (RewardFunction): The reward function object.
            vel_factors (Dict[str, float], optional): Dictionary containing the reward scalars for each action dimension. Defaults to DEFAULTS.ABRUPT_VEL_CHANGE.VEL_FACTORS.
            _on_safe_dist_violation (bool, optional): Flag to indicate if there is a violation of safe distance. Defaults to DEFAULTS.ABRUPT_VEL_CHANGE._ON_SAFE_DIST_VIOLATION.
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._vel_factors = vel_factors
        self.last_action = None

        self._vel_change_fcts = self._get_vel_change_fcts()

    def _get_vel_change_fcts(self):
        return [
            self._prepare_reward_function(int(idx), factor)
            for idx, factor in self._vel_factors.items()
        ]

    def _prepare_reward_function(
        self, idx: int, factor: float
    ) -> Callable[[np.ndarray], None]:
        def vel_change_fct(action: np.ndarray):
            assert isinstance(self.last_action, np.ndarray)
            vel_diff = abs(action[idx] - self.last_action[idx])
            self.add_reward(-((vel_diff**4 / 100) * factor))

        return vel_change_fct

    def __call__(self, action: np.ndarray, *args, **kwargs):
        if self.last_action is not None:
            for rew_fct in self._vel_change_fcts:
                rew_fct(action)
        self.last_action = action

    def reset(self):
        self.last_action = None

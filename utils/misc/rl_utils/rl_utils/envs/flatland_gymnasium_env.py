#! /usr/bin/env python3
import math
import os
import time
from typing import Tuple

import gymnasium
import numpy as np
import rospy
from flatland_msgs.msg import StepWorld
from geometry_msgs.msg import Twist
from rosnav.rosnav_space_manager.rosnav_space_manager import RosnavSpaceManager
from stable_baselines3.common.env_checker import check_env
from std_srvs.srv import Empty
from task_generator.shared import Namespace
from task_generator.task_generator_node import TaskGenerator
from task_generator.tasks.base_task import BaseTask
from task_generator.utils import rosparam_get

# from ..utils.old_observation_collector import ObservationCollector
from rl_utils.utils.observation_collector.observation_manager import ObservationManager
from rl_utils.utils.rewards.reward_function import RewardFunction
from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS


def delay_node_init(ns):
    try:
        # given every environment enough time to initialize, if we dont put sleep,
        # the training script may crash.
        import re

        ns_int = int(re.search(r"\d+", ns)[0])
        time.sleep((ns_int + 1) * 2)
    except Exception:
        rospy.logwarn(
            "Can't not determinate the number of the environment, training script may crash!"
        )
        time.sleep(2)


class FlatlandEnv(gymnasium.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        ns: str,
        reward_fnc: str,
        max_steps_per_episode=100,
        verbose: bool = True,
        log_last_n_eps: int = 20,
        *args,
        **kwargs,
    ):
        """Default env
        Flatland yaml node check the entries in the yaml file, therefore other robot related parameters cound only be saved in an other file.


        Args:
            task (ABSTask): [description]
            reward_fnc (str): [description]
            train_mode (bool): bool to differ between train and eval env during training
            safe_dist (float, optional): [description]. Defaults to None.
            goal_radius (float, optional): [description]. Defaults to 0.1.
            extended_eval (bool): more episode info provided, no reset when crashing
        """
        super(FlatlandEnv, self).__init__()

        self.ns = Namespace(ns)

        delay_node_init(ns=self.ns.simulation_ns)

        if not rospy.get_param("/debug_mode", True):
            rospy.init_node("env_" + self.ns, anonymous=True)

        self._is_train_mode = rospy.get_param("/train_mode")
        self.model_space_encoder = RosnavSpaceManager()

        # observation collector
        self.observation_collector = ObservationManager(self.ns)

        self.action_space = self.model_space_encoder.get_action_space()
        self.observation_space = self.model_space_encoder.get_observation_space()

        # instantiate task manager
        task_generator = TaskGenerator(self.ns)
        self.task: BaseTask = task_generator._get_predefined_task(**kwargs)

        # reward calculator
        self.reward_calculator = RewardFunction(
            rew_func_name=reward_fnc,
            holonomic=self.model_space_encoder._is_holonomic,
            robot_radius=self.task.robot_managers[0]._robot_radius,
            safe_dist=self.task.robot_managers[0].safe_distance,
            goal_radius=rosparam_get(float, "goal_radius", 0.3),
        )

        # action agent publisher
        if self._is_train_mode:
            self.agent_action_pub = rospy.Publisher(
                self.ns("cmd_vel"), Twist, queue_size=1
            )
        else:
            self.agent_action_pub = rospy.Publisher(
                self.ns("cmd_vel_pub"), Twist, queue_size=1
            )

        # service clients
        if self._is_train_mode:
            self._service_name_step = self.ns.simulation_ns("step_world")
            # self._sim_step_client = rospy.ServiceProxy(self._service_name_step, StepWorld)
            self._step_world_publisher = rospy.Publisher(
                self._service_name_step, StepWorld, queue_size=10
            )
            self._step_world_srv = rospy.ServiceProxy(
                self._service_name_step, Empty, persistent=True
            )

        self._verbose = verbose
        self._log_last_n_eps = log_last_n_eps

        self._steps_curr_episode = 0
        self._episode = 0
        self._max_steps_per_episode = max_steps_per_episode
        self._last_action = np.array([0, 0, 0])  # linear x, linear y, angular z

        # for extended eval
        self._action_frequency = 1 / rospy.get_param("/robot_action_rate", 10)
        self._last_robot_pose = None
        self._distance_travelled = 0
        self._safe_dist_counter = 0
        self._collisions = 0
        self._in_crash = False

        self.last_mean_reward = 0
        self.mean_reward = [0, 0]
        self.step_count_hist = [0] * self._log_last_n_eps
        self.step_time = [0, 0]

        self._done_reasons = {
            "0": "Timeout",
            "1": "Crash",
            "2": "Success",
        }
        self._done_hist = 3 * [0]

    def _pub_action(self, action: np.ndarray) -> Twist:
        assert len(action) == 3

        action_msg = Twist()
        action_msg.linear.x = action[0]
        action_msg.linear.y = action[1]
        action_msg.angular.z = action[2]

        self.agent_action_pub.publish(action_msg)

    def step(self, action: np.ndarray):
        """
        done_reasons:   0   -   exceeded max steps
                        1   -   collision with obstacle
                        2   -   goal reached
        """

        start_time = time.time()

        decoded_action = self.model_space_encoder.decode_action(action)
        self._pub_action(decoded_action)

        if self._is_train_mode:
            self.call_service_takeSimStep()

        obs_dict = self.observation_collector.get_observations(
            last_action=self._last_action
        )
        self._last_action = decoded_action

        # calculate reward
        reward, reward_info = self.reward_calculator.get_reward(
            action=decoded_action,
            **obs_dict,
        )

        self.update_statistics(reward=reward)

        # info
        info, done = FlatlandEnv.determine_termination(
            reward_info=reward_info,
            curr_steps=self._steps_curr_episode,
            max_steps=self._max_steps_per_episode,
        )

        if done and self._verbose:
            self.step_count_hist[
                self._episode % self._log_last_n_eps
            ] = self._steps_curr_episode
            self._done_hist[int(info["done_reason"])] += 1
            if sum(self._done_hist) >= self._log_last_n_eps:
                self.print_statistics()

        self.step_time[0] += time.time() - start_time

        return (
            self.model_space_encoder.encode_observation(
                obs_dict,
                [OBS_DICT_KEYS.LASER, OBS_DICT_KEYS.GOAL, OBS_DICT_KEYS.LAST_ACTION],
            ),
            reward,
            done,
            False,
            info,
        )

    def call_service_takeSimStep(self, t=None):
        # request = StepWorld()
        # request.required_time = 0 if t == None else t

        self._step_world_srv()

        # self._step_world_publisher.publish(request)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # set task
        # regenerate start position end goal position of the robot and change the obstacles accordingly
        self._episode += 1
        self.agent_action_pub.publish(Twist())

        first_map = self._episode <= 1 if "sim_1" in self.ns else False
        self.task.reset(
            callback=lambda: False,
            first_map=first_map,
            reset_after_new_map=self._steps_curr_episode == 0,
        )
        self.reward_calculator.reset()
        self._steps_curr_episode = 0
        self._last_action = np.array([0, 0, 0])

        if self._is_train_mode:
            self.call_service_takeSimStep()

        obs_dict = self.observation_collector.get_observations()
        info_dict = {}
        return (
            self.model_space_encoder.encode_observation(
                obs_dict, ["laser_scan", "goal_in_robot_frame", "last_action"]
            ),
            info_dict,
        )

    def close(self):
        pass

    def update_statistics(self, **kwargs) -> None:
        self.step_time[1] += 1
        self.mean_reward[1] += 1
        self.mean_reward[0] += kwargs["reward"]
        self._steps_curr_episode += 1

    def print_statistics(self):
        mean_reward = self.mean_reward[0] / self._log_last_n_eps
        diff = round(mean_reward - self.last_mean_reward, 5)

        print(
            f"[{self.ns}] Last {self._log_last_n_eps} Episodes:\t"
            f"{self._done_reasons[str(0)]}: {self._done_hist[0]}\t"
            f"{self._done_reasons[str(1)]}: {self._done_hist[1]}\t"
            f"{self._done_reasons[str(2)]}: {self._done_hist[2]}\t"
            f"Mean step time: {round(self.step_time[0] / self.step_time[1] * 100, 2)}\t"
            f"Mean cum. reward: {round(mean_reward, 5)} ({'+' if diff >= 0 else ''}{diff})\t"
            f"Mean steps: {sum(self.step_count_hist) / self._log_last_n_eps}\t"
        )
        self._done_hist = [0] * 3
        self.step_time = [0, 0]
        self.last_mean_reward = mean_reward
        self.mean_reward = [0, 0]
        self.step_count_hist = [0] * self._log_last_n_eps

    @staticmethod
    def determine_termination(
        reward_info: dict,
        curr_steps: int,
        max_steps: int,
        info: dict = None,
    ) -> Tuple[dict, bool]:
        if info is None:
            info = {}

        terminated = reward_info["is_done"]

        if terminated:
            info["done_reason"] = reward_info["done_reason"]
            info["is_success"] = reward_info["is_success"]

        if curr_steps >= max_steps:
            terminated = True
            info["done_reason"] = 0
            info["is_success"] = 0

        return info, terminated


if __name__ == "__main__":
    rospy.init_node("flatland_gym_env", anonymous=True, disable_signals=False)
    print("start")

    flatland_env = FlatlandEnv()
    rospy.loginfo("======================================================")
    rospy.loginfo("CSVWriter initialized.")
    rospy.loginfo("======================================================")
    check_env(flatland_env, warn=True)

    # init env
    obs = flatland_env.reset()

    # run model
    n_steps = 200
    for _ in range(n_steps):
        # action, _states = model.predict(obs)
        action = flatland_env.action_space.sample()

        obs, rewards, done, info = flatland_env.step(action)

        time.sleep(0.1)

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from magbotsim.envs_without_rl_api.magbots.reconfiguration_env import ReconfigurationEnv
from magbotsim.rl_envs.magbots.reconfiguration_rl_env import ReconfigurationRLEnv


class ReconfigurationPlacementEnv(ReconfigurationRLEnv):
    """Goal-conditioned MagBot placement task: the movers have to pick up the MagBot from its station and place it at a goal position
    on the table. The environment follows the goal-conditioned API of Gymnasium-Robotics, i.e. it can be used with Hindsight Experience
    Replay (HER), e.g. the ``HerReplayBuffer`` of Stable-Baselines3.

    The actions, the switch of the rail mount welds, the collision checks and the info dictionary are inherited from the
    ``ReconfigurationRLEnv`` (see there). In particular, the MagBot can only be picked up if the movers enter the station along the
    entry axis, stop at the real rail mount positions and leave the station along the take-out axis.

    MagBot position: the mean of the real (x,y) positions of the rail mounts of the MagBot (see ``get_magbot_xy()``). The rail mounts
    are either parked in the station or carried by the movers, i.e. the MagBot position is always defined.

    Goal: an (x,y) position of the MagBot on the table, sampled uniformly at each reset (see ``sample_goal_xy()``). Each goal is
    reachable: if the MagBot is taken out of its station without rotation and moved to the goal, all movers are on the table and
    keep a clearance of ``goal_margin`` to the walls and to all stations. The goal is shown as a green sphere.

    Observation: a dictionary with the keys

    - ``observation``: the observation of the ``ReconfigurationRLEnv`` (see ``ReconfigurationRLEnv._get_obs()``)
    - ``achieved_goal``: the current MagBot position (numpy array of shape (2,)) [m]
    - ``desired_goal``: the goal position of the MagBot (numpy array of shape (2,)) [m]

    Reward (sparse): ``reward_collision`` in case of a mover, wall or station proxy collision, 0 if the MagBot is within
    ``threshold_pos`` of the goal, ``reward_per_step`` otherwise. The reward only depends on the goals and the collision information of
    the info dictionary, so that it can be recomputed for relabeled goals (HER).

    Termination and truncation: an episode terminates in case of a collision (see ``ReconfigurationRLEnv``). Reaching the goal does not
    terminate the episode, i.e. the agent has to keep the MagBot at the goal. The episode is never truncated by the environment itself,
    i.e. use a ``TimeLimit`` wrapper (the registered environments ``ReconfigurationPlacementGripperEnv-v0`` and
    ``ReconfigurationPlacementSixDPlatformEnv-v0`` use 100 steps).

    Info: the collision types of the ``ReconfigurationRLEnv`` and ``is_success`` (bool), i.e. whether the MagBot is within
    ``threshold_pos`` of the goal without a collision.

    :param magbot_types: the MagBot types that can be used, defaults to both types (see ``ReconfigurationEnv``). For vectorized
        environments, specify a single MagBot type, since the number of movers depends on the MagBot type.
    :param initial_mover_xy_pos: the initial (x, y) positions of the movers on the table [m] (see ``ReconfigurationRLEnv``), defaults
        to None, i.e. the start positions are sampled at each reset
    :param station_poses: the poses of the station anchors (see ``ReconfigurationEnv``), defaults to None
    :param station_switch_tol: the tolerance at which the welds are switched [m] (see ``ReconfigurationEnv``), defaults to 0.001
    :param station_proxy_tol: the sideways misalignment with which a mover can enter a station without colliding [m] (see
        ``ReconfigurationEnv``), defaults to 0.002
    :param randomize_magbot_station_pos: whether to shift each MagBot in its station by a random offset at each reset (see
        ``ReconfigurationEnv``), defaults to False
    :param magbot_station_randomization_x_range: the half-range of the random MagBot offset [m] (see ``ReconfigurationEnv``), defaults
        to 0.005
    :param num_cycles: the number of control cycles per step, defaults to 40
    :param render_every_cycle: whether to call ``render()`` after each integrator step, defaults to False
    :param v_max_xy: the maximum (x,y) set velocity of each mover [m/s], defaults to 2.0
    :param v_max_yaw: the maximum yaw set velocity of each mover [rad/s], defaults to pi/2
    :param reward_collision: the reward in case of a collision, defaults to -20.0
    :param reward_per_step: the reward per step if the goal is not reached, defaults to -1.0
    :param threshold_pos: the maximum distance between the MagBot position and the goal at which the goal is reached [m], defaults to
        0.01
    :param goal_margin: the minimum clearance of the movers to the walls and to the stations if the MagBot is at the goal [m] (see
        ``goal_xy_is_valid()``), defaults to 0.02
    :param seed: the seed of the RNG that chooses the MagBot type at initialization, defaults to None. The scene and the goal are
        randomized using ``np_random`` at each reset, i.e. use ``reset(seed=...)`` to make the episodes reproducible.
    :param render_mode: the render mode, defaults to 'human'. Use None to run without rendering (e.g. for training).
    """

    GOAL_SITE_NAME = 'magbot_goal_site'

    def __init__(
        self,
        magbot_types: Sequence[str] = ReconfigurationEnv.MAGBOT_TYPES,
        initial_mover_xy_pos: np.ndarray | dict[str, np.ndarray] | None = None,
        station_poses: np.ndarray | None = None,
        station_switch_tol: float = 0.001,
        station_proxy_tol: float = 0.002,
        randomize_magbot_station_pos: bool = False,
        magbot_station_randomization_x_range: float = 0.005,
        num_cycles: int = 40,
        render_every_cycle: bool = False,
        v_max_xy: float = 2.0,
        v_max_yaw: float = np.pi / 2,
        reward_collision: float = -20.0,
        reward_per_step: float = -1.0,
        threshold_pos: float = 0.01,
        goal_margin: float = 0.02,
        seed: int | None = None,
        render_mode: str | None = 'human',
    ) -> None:
        assert threshold_pos > 0.0 and goal_margin >= 0.0
        self.threshold_pos = threshold_pos
        self.goal_margin = goal_margin

        super().__init__(
            magbot_types=magbot_types,
            initial_mover_xy_pos=initial_mover_xy_pos,
            station_poses=station_poses,
            station_switch_tol=station_switch_tol,
            station_proxy_tol=station_proxy_tol,
            randomize_magbot_station_pos=randomize_magbot_station_pos,
            magbot_station_randomization_x_range=magbot_station_randomization_x_range,
            num_cycles=num_cycles,
            render_every_cycle=render_every_cycle,
            v_max_xy=v_max_xy,
            v_max_yaw=v_max_yaw,
            reward_collision=reward_collision,
            reward_per_step=reward_per_step,
            seed=seed,
            render_mode=render_mode,
        )
        assert self.num_magbots == 1, 'the placement task is defined for a single MagBot'
        self.site_id_goal = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.GOAL_SITE_NAME)

        # footprint of the movers if the MagBot is taken out of its station (MagBot 0 is parked in station 0) without rotation:
        # mover positions relative to the MagBot position and half sizes of the axis-aligned bounding boxes of the movers (incl. margin)
        station_mover_xy = self.station_mover_poses[0, :, :2]
        self.goal_footprint_offsets = station_mover_xy - station_mover_xy.mean(axis=0)
        self.goal_footprint_yaws = self.station_mover_poses[0, :, 5].copy()
        mover_half_sizes = self.get_c_size_arr(c_size=self.c_size, num_reps=self.num_movers) + self.c_size_offset
        self.goal_footprint_half_extents = self._aabb_half_extents(mover_half_sizes + self.goal_margin, self.goal_footprint_yaws)
        # axis-aligned bounding boxes of the stations: the movers at the nominal station positions, enlarged by the MagBot randomization
        # range (shape: (num_stations, 2) each)
        station_half_extents = np.array(
            [self._aabb_half_extents(mover_half_sizes, poses[:, 5]) for poses in self.station_mover_poses]
        ) + self.magbot_station_randomization_x_range * float(self.randomize_magbot_station_pos)
        self.station_xy_min = (self.station_mover_poses[:, :, :2] - station_half_extents).min(axis=1)
        self.station_xy_max = (self.station_mover_poses[:, :, :2] + station_half_extents).max(axis=1)
        # range of the MagBot positions for which all movers are on the table (incl. margin)
        table_xy_min, table_xy_max = self.table_xy_limits
        self.goal_xy_low = table_xy_min - (self.goal_footprint_offsets - self.goal_footprint_half_extents).min(axis=0)
        self.goal_xy_high = table_xy_max - (self.goal_footprint_offsets + self.goal_footprint_half_extents).max(axis=0)
        assert np.all(self.goal_xy_low < self.goal_xy_high), 'the MagBot does not fit on the table'

        # observation space: goal-conditioned dictionary (see _get_obs())
        self.observation_space = gym.spaces.Dict(
            {
                'observation': self.observation_space,
                'achieved_goal': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64),
                'desired_goal': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64),
            }
        )

        # initial goal (sampled again at each reset)
        self._set_goal(self.sample_goal_xy(rng=self.rng))

    ###################################################
    # Model and goal                                  #
    ###################################################

    def _custom_xml_string_callback(self, custom_model_xml_strings: dict[str, str] | None = None) -> dict[str, str]:
        """Add the MagBots, stations and actuators (see ``ReconfigurationEnv._custom_xml_string_callback()``) and the goal site to the
        MuJoCo model.

        :param custom_model_xml_strings: the current ``custom_model_xml_strings``-dict which is modified by this callback, defaults to None
        :return: the modified ``custom_model_xml_strings``-dict
        """
        custom_model_xml_strings = super()._custom_xml_string_callback(custom_model_xml_strings=custom_model_xml_strings)
        if hasattr(self, 'mover_joint_names'):
            # the goal site is shown at the height of the rail mounts; its position is set at each reset
            goal_z = self.initial_mover_z_pos + float(np.max(self.mover_size[:, 2])) + 0.006
            custom_model_xml_strings['custom_worldbody_xml_str'] = (
                custom_model_xml_strings.get('custom_worldbody_xml_str', '')
                + '\n\n\t\t<!-- goal of the MagBot -->'
                + f'\n\t\t<site name="{self.GOAL_SITE_NAME}" type="sphere" size="0.02" pos="0 0 {goal_z}" rgba="0.2852 0.5078 0.051 0.7"/>'
            )
        return custom_model_xml_strings

    @staticmethod
    def _aabb_half_extents(half_sizes: np.ndarray, yaws: np.ndarray) -> np.ndarray:
        """Return the half extents of the axis-aligned bounding boxes of rotated boxes.

        :param half_sizes: the (x,y) half sizes of the boxes (numpy array of shape (num_boxes,2)) [m]
        :param yaws: the yaws of the boxes (numpy array of shape (num_boxes,)) [rad]
        :return: the (x,y) half extents of the bounding boxes (numpy array of shape (num_boxes,2)) [m]
        """
        abs_cos, abs_sin = np.abs(np.cos(yaws)), np.abs(np.sin(yaws))
        return np.stack(
            [abs_cos * half_sizes[:, 0] + abs_sin * half_sizes[:, 1], abs_sin * half_sizes[:, 0] + abs_cos * half_sizes[:, 1]], axis=1
        )

    def get_magbot_xy(self) -> np.ndarray:
        """Return the MagBot position, i.e. the mean of the real (x,y) positions of the rail mounts of the MagBot.

        :return: the MagBot position (numpy array of shape (2,)) [m]
        """
        return self.get_rail_mount_xy().mean(axis=0)

    def goal_xy_is_valid(self, goal_xy: np.ndarray) -> bool:
        """Check whether the MagBot can be placed at the goal, i.e. whether the movers keep a clearance of ``goal_margin`` to the walls
        and to the stations if the MagBot is taken out of its station without rotation and moved to the goal. The clearance to the
        stations is checked with the axis-aligned bounding boxes of the movers (incl. the MagBot between them) and of the stations.

        :param goal_xy: the goal position of the MagBot (numpy array of shape (2,)) [m]
        :return: True if the goal is valid, False otherwise
        """
        if np.any(goal_xy < self.goal_xy_low) or np.any(goal_xy > self.goal_xy_high):
            return False
        mover_xy = goal_xy + self.goal_footprint_offsets
        footprint_xy_min = (mover_xy - self.goal_footprint_half_extents).min(axis=0)
        footprint_xy_max = (mover_xy + self.goal_footprint_half_extents).max(axis=0)
        overlaps = np.all((footprint_xy_min < self.station_xy_max) & (footprint_xy_max > self.station_xy_min), axis=1)
        return not overlaps.any()

    def sample_goal_xy(self, rng: np.random.Generator, max_trials: int = 1000) -> np.ndarray:
        """Sample a valid goal position of the MagBot uniformly on the table (rejection sampling, see ``goal_xy_is_valid()``).

        :param rng: the RNG used for sampling
        :param max_trials: the maximum number of trials, defaults to 1000
        :return: the goal position of the MagBot (numpy array of shape (2,)) [m]
        """
        for _ in range(0, max_trials):
            goal_xy = rng.uniform(low=self.goal_xy_low, high=self.goal_xy_high)
            if self.goal_xy_is_valid(goal_xy):
                return goal_xy
        raise RuntimeError(f'No valid goal position found within {max_trials} trials.')

    def _set_goal(self, goal_xy: np.ndarray) -> None:
        """Set the goal position of the MagBot and move the goal site.

        :param goal_xy: the goal position of the MagBot (numpy array of shape (2,)) [m]
        """
        self.goal_xy = np.asarray(goal_xy, dtype=np.float64).copy()
        assert self.goal_xy.shape == (2,)
        self.model.site_pos[self.site_id_goal, :2] = self.goal_xy

    def _reset_callback(self, options: dict[str, Any] | None = None) -> None:
        """Reset the scene (see ``ReconfigurationRLEnv._reset_callback()``) and sample a new goal using ``np_random``.

        :param options: a dictionary that can be used to specify the reset options of the ``ReconfigurationRLEnv`` and the following
            option, defaults to None:

            - ``goal_xy``: the goal position of the MagBot (numpy array of shape (2,)) [m]. If not specified, the goal is sampled (see
              ``sample_goal_xy()``). A specified goal is not checked for validity.
        """
        super()._reset_callback(options=options)
        options = options or {}
        goal_xy = options.get('goal_xy', None)
        self._set_goal(self.sample_goal_xy(rng=self.np_random) if goal_xy is None else goal_xy)

    ###################################################
    # Observation, info, reward                       #
    ###################################################

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Return an observation based on the current state of the environment.

        :return: a dictionary with the keys ``observation`` (see ``ReconfigurationRLEnv._get_obs()``), ``achieved_goal`` (the current
            MagBot position, see ``get_magbot_xy()``) and ``desired_goal`` (the goal position of the MagBot)
        """
        return {
            'observation': super()._get_obs(),
            'achieved_goal': self.get_magbot_xy(),
            'desired_goal': self.goal_xy.copy(),
        }

    def _get_info(
        self,
        mover_collision: bool,
        wall_collision: bool,
        other_collision: bool,
        achieved_goal: np.ndarray | None = None,
        desired_goal: np.ndarray | None = None,
        collision_info: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        """Return the info dictionary of the ``ReconfigurationEnv`` (collision types, see ``ReconfigurationEnv._get_info()``) with the
        additional key ``is_success``.

        :param mover_collision: whether there is a collision between two movers
        :param wall_collision: whether there is a collision between a mover and a wall
        :param other_collision: whether there is a station proxy collision
        :param achieved_goal: the current MagBot position (numpy array of shape (2,)), defaults to None, i.e. ``get_magbot_xy()``
        :param desired_goal: the goal position of the MagBot (numpy array of shape (2,)), defaults to None, i.e. ``goal_xy``
        :param collision_info: the station proxy collision types (see ``get_station_proxy_collisions()``), defaults to None
        :return: the info dictionary
        """
        info = super()._get_info(
            mover_collision=mover_collision,
            wall_collision=wall_collision,
            other_collision=other_collision,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
            collision_info=collision_info,
        )
        achieved_goal = self.get_magbot_xy() if achieved_goal is None else achieved_goal
        desired_goal = self.goal_xy if desired_goal is None else desired_goal
        collision = any(info[key] for key in self.COLLISION_KEYS)
        info['is_success'] = bool(self._goal_distance(achieved_goal, desired_goal)[0] <= self.threshold_pos) and not collision
        return info

    @staticmethod
    def _goal_distance(achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        """Return the Euclidean distances between the achieved and the desired goals.

        :param achieved_goal: the MagBot positions (numpy array of shape (2,) or (batch_size, 2))
        :param desired_goal: the goal positions of the MagBot (numpy array of shape (2,) or (batch_size, 2))
        :return: the distances (numpy array of shape (batch_size,)) [m]
        """
        return np.linalg.norm(np.reshape(achieved_goal, (-1, 2)) - np.reshape(desired_goal, (-1, 2)), axis=1)

    def compute_reward(
        self, achieved_goal: np.ndarray | None = None, desired_goal: np.ndarray | None = None, info: dict[str, Any] | None = None
    ) -> np.ndarray | float:
        """Compute the immediate reward (see class documentation). This method is vectorized, i.e. it can be used to recompute the
        rewards for relabeled goals (HER).

        :param achieved_goal: the MagBot positions (numpy array of shape (2,) or (batch_size, 2))
        :param desired_goal: the goal positions of the MagBot (numpy array of shape (2,) or (batch_size, 2))
        :param info: the info dictionary or an array of info dictionaries (shape (batch_size,))
        :return: a single float value or (if ``info`` is an array) a numpy array of shape (batch_size,) containing the immediate
            rewards
        """
        assert achieved_goal is not None and desired_goal is not None
        is_batch, collisions = self._preprocess_info_dict(info=info)
        goal_reached = self._goal_distance(achieved_goal, desired_goal) <= self.threshold_pos
        assert goal_reached.shape == collisions.shape, 'the batch sizes of the goals and of the info dictionaries do not match'
        reward = np.where(collisions, self.reward_collision, np.where(goal_reached, 0.0, self.reward_per_step)).astype(np.float64)
        return reward if is_batch else float(reward[0])


if __name__ == '__main__':
    # random actions (the goal is shown as a green sphere), 100 steps per episode
    env = ReconfigurationPlacementEnv(magbot_types=('sixD_platform',), render_mode='human')
    try:
        for episode in range(10):
            observation, info = env.reset(seed=episode)
            for _ in range(100):
                observation, reward, terminated, truncated, info = env.step(0.25 * env.action_space.sample())
                if terminated:
                    break
            print(f'episode {episode}: goal {observation["desired_goal"]}, MagBot {observation["achieved_goal"]}, info: {info}')
    finally:
        env.close()

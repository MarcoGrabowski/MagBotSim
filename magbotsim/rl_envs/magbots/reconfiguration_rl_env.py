from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np

from magbotsim import BasicMagBotSingleAgentEnv
from magbotsim.envs_without_rl_api.magbots.reconfiguration_env import ReconfigurationEnv
from magbotsim.utils import rotations_utils


class ReconfigurationRLEnv(ReconfigurationEnv, BasicMagBotSingleAgentEnv):
    """Single-agent RL variant (Gymnasium API) of the ``ReconfigurationEnv``. The scene (MagBot parked in a station, movers free on the
    table), the switch of the rail mount welds between the movers and the stations, the MagBot control and the collision checks are
    inherited from the ``ReconfigurationEnv``, i.e. both environments behave identically for the same mover set poses.

    This environment is a skeleton without a specific task: the agent is penalized for collisions and receives a constant reward per
    step otherwise. Concrete tasks can be implemented by subclasses that override ``_get_obs()``, ``compute_reward()``,
    ``compute_terminated()`` and ``compute_truncated()`` (and the ``observation_space``), see e.g. ``ReconfigurationPlacementEnv``.

    Action: normalized increments of the mover set poses (x, y, yaw) for each mover (numpy array of shape (num_movers*3,) with values in
    [-1, 1], connection order). The (x,y)-increments of each mover are clipped to a norm of at most 1 and scaled by
    ``v_max_xy * num_cycles * cycle_time``, the yaw increments are scaled by ``v_max_yaw * num_cycles * cycle_time``. The (x,y) set
    positions are clipped to the table. Within a step, the set poses are linearly interpolated from the previous to the new set poses
    over all cycles, i.e. the movers move with a constant set velocity. The z-position and the a,b-rotations of the set poses are fixed.

    Observation: see ``_get_obs()``.

    Info: the collision types of the ``ReconfigurationEnv`` (see ``ReconfigurationEnv._get_info()``), i.e. ``mover_collision``,
    ``wall_collision``, ``station_proxy_collision``, ``rail_rod_collision`` and ``station_pin_collision``.

    Reward and termination: ``reward_collision`` and termination if there is a mover, wall or station proxy collision, otherwise
    ``reward_per_step``. The episode is never truncated by the environment itself, i.e. use a ``TimeLimit`` wrapper to limit the
    episode length.

    The MagBot type (and thus the number of movers and the shapes of the action and observation spaces) is chosen once at
    initialization (see ``magbot_types``). For vectorized environments, specify a single MagBot type.

    :param magbot_types: the MagBot types that can be used, defaults to both types (see ``ReconfigurationEnv``)
    :param initial_mover_xy_pos: the initial (x, y) positions of the movers on the table [m] (see ``ReconfigurationEnv``), defaults to
        None. If specified, the movers start at these positions after each reset (unless other positions are specified in the reset
        options), otherwise the start positions are sampled at each reset.
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
    :param reward_per_step: the reward per step without collision, defaults to -1.0
    :param seed: the seed of the RNG that chooses the MagBot type at initialization, defaults to None. The scene is randomized using
        ``np_random`` at each reset, i.e. use ``reset(seed=...)`` to make the episodes reproducible.
    :param render_mode: the render mode, defaults to 'human'. Use None to run without rendering (e.g. for training).
    """

    # collision types that terminate an episode (station_proxy_collision includes rail_rod_collision and station_pin_collision)
    COLLISION_KEYS = ('mover_collision', 'wall_collision', 'station_proxy_collision')
    # number of observation values per mover (see _get_obs())
    NUM_OBS_PER_MOVER = 11

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
        seed: int | None = None,
        render_mode: str | None = 'human',
    ) -> None:
        assert num_cycles >= 1
        assert v_max_xy > 0.0 and v_max_yaw > 0.0

        # MRO: ReconfigurationEnv -> BasicMagBotSingleAgentEnv -> BasicMagBotEnv, i.e. the scene is created by the ReconfigurationEnv
        super().__init__(
            magbot_types=magbot_types,
            initial_mover_xy_pos=initial_mover_xy_pos,
            station_poses=station_poses,
            station_switch_tol=station_switch_tol,
            station_proxy_tol=station_proxy_tol,
            randomize_magbot_station_pos=randomize_magbot_station_pos,
            magbot_station_randomization_x_range=magbot_station_randomization_x_range,
            seed=seed,
            render_mode=render_mode,
        )
        # BasicMagBotSingleAgentEnv.__init__() sets its default values, since the ReconfigurationEnv does not pass these parameters
        self.num_cycles = num_cycles
        self.render_every_cycle = render_every_cycle

        # rewards
        self.reward_collision = reward_collision
        self.reward_per_step = reward_per_step

        # maximum set pose increments per step
        self.v_max_xy = v_max_xy
        self.v_max_yaw = v_max_yaw
        self.max_delta_xy = v_max_xy * self.num_cycles * self.cycle_time  # [m]
        self.max_delta_yaw = v_max_yaw * self.num_cycles * self.cycle_time  # [rad]

        # action space: normalized (x, y, yaw) increments of the mover set poses
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_movers * 3,), dtype=np.float64)
        # observation space (see _get_obs())
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_movers * self.NUM_OBS_PER_MOVER,), dtype=np.float64
        )

    ###################################################
    # Scene and set poses                             #
    ###################################################

    @property
    def table_xy_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """The minimum and maximum (x,y) positions on the table [m] (each a numpy array of shape (2,))."""
        xy_min = np.array([self.x_pos_tiles.min() - self.tile_size[0], self.y_pos_tiles.min() - self.tile_size[1]])
        xy_max = np.array([self.x_pos_tiles.max() + self.tile_size[0], self.y_pos_tiles.max() + self.tile_size[1]])
        return xy_min, xy_max

    def reset_scene(
        self,
        mover_start_xy_pos: np.ndarray | None = None,
        magbot_station_x_offsets: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Reset the scene (see ``ReconfigurationEnv.reset_scene()``) and the mover set poses, i.e. the movers are held at their start
        positions with a yaw of 0.

        :param mover_start_xy_pos: the (x,y) start positions of the movers (numpy array of shape (num_movers,2)) [m], defaults to None,
            i.e. the start positions are sampled
        :param magbot_station_x_offsets: the offsets of the MagBots along the x-axis of their stations (numpy array of shape
            (num_magbots,)) [m], defaults to None (see ``ReconfigurationEnv.reset_scene()``)
        :param rng: the RNG used for sampling, defaults to None, i.e. ``self.rng``
        """
        super().reset_scene(mover_start_xy_pos=mover_start_xy_pos, magbot_station_x_offsets=magbot_station_x_offsets, rng=rng)
        self.mover_setposes = np.zeros((self.num_movers, 6))
        self.mover_setposes[:, :2] = self.mover_start_xy_pos
        self.mover_setposes[:, 2] = self.initial_mover_z_pos
        self.prev_mover_setposes = self.mover_setposes.copy()
        self.idx_cycle = 0

    def _reset_callback(self, options: dict[str, Any] | None = None) -> None:
        """Reset the scene using ``np_random``, i.e. the scene is reproducible with ``reset(seed=...)``.

        :param options: a dictionary that can be used to specify the following reset options, defaults to None:

            - ``mover_start_xy_pos``: the (x,y) start positions of the movers (numpy array of shape (num_movers,2)) [m], defaults to the
              ``initial_mover_xy_pos`` specified at initialization. If both are None, the start positions are sampled.
            - ``magbot_station_x_offsets``: the offsets of the MagBots along the x-axis of their stations (numpy array of shape
              (num_magbots,)) [m]. If not specified, the offsets are sampled if ``randomize_magbot_station_pos`` is True, otherwise
              they are zero.
        """
        options = options or {}
        self.reset_scene(
            mover_start_xy_pos=options.get('mover_start_xy_pos', self.fixed_mover_start_xy_pos),
            magbot_station_x_offsets=options.get('magbot_station_x_offsets', None),
            rng=self.np_random,
        )

    ###################################################
    # Step                                            #
    ###################################################

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:  # type: ignore[override]
        """Execute one step of the environment's dynamics applying the given action (see ``BasicMagBotSingleAgentEnv.step()``).

        This method must be overridden explicitly, since ``ReconfigurationEnv.step()`` (control via absolute mover set poses) precedes
        ``BasicMagBotSingleAgentEnv.step()`` in the method resolution order.

        :param action: the normalized (x, y, yaw) increments of the mover set poses (numpy array of shape (num_movers*3,))
        :return: the next observation, the immediate reward, whether a terminal state is reached, whether the truncation condition is
            satisfied, and the info dictionary
        """
        return BasicMagBotSingleAgentEnv.step(self, action)

    def _step_callback(self, action: np.ndarray) -> np.ndarray:
        """Compute the new mover set poses from the action (see class documentation).

        :param action: the normalized (x, y, yaw) increments of the mover set poses (numpy array of shape (num_movers*3,))
        :return: the unchanged action
        """
        increments = np.asarray(action, dtype=np.float64).reshape((self.num_movers, 3))
        # clip the norm of the (x,y)-increments of each mover to 1
        delta_xy = increments[:, :2]
        norm_xy = np.linalg.norm(delta_xy, axis=1, keepdims=True)
        delta_xy = np.where(norm_xy > 1.0, delta_xy / np.maximum(norm_xy, 1e-12), delta_xy)

        self.prev_mover_setposes = self.mover_setposes.copy()
        xy_min, xy_max = self.table_xy_limits
        self.mover_setposes[:, :2] = np.clip(self.mover_setposes[:, :2] + self.max_delta_xy * delta_xy, xy_min, xy_max)
        self.mover_setposes[:, 5] += self.max_delta_yaw * increments[:, 2]
        self.idx_cycle = 0
        return action

    def _before_mujoco_step_callback(self, action: np.ndarray) -> None:
        """Apply the controls of the current cycle, i.e. the set poses linearly interpolated between the previous and the new set poses
        (see ``ReconfigurationEnv._apply_mover_controls()``).

        :param action: the action of the current step (not used, the set poses are computed in ``_step_callback()``)
        """
        self.idx_cycle += 1
        s = min(self.idx_cycle / self.num_cycles, 1.0)
        mover_setposes = self.prev_mover_setposes + s * (self.mover_setposes - self.prev_mover_setposes)
        self._apply_mover_controls(mover_setposes=mover_setposes)

    ###################################################
    # Observation, reward, termination                #
    ###################################################

    def _get_obs(self) -> np.ndarray:
        """Return an observation based on the current state of the environment.

        :return: a numpy array of shape (num_movers*11,) containing the following values for each mover (connection order):

            - the current (x, y, yaw) of the mover (with noise) [m, m, rad]
            - the current (x, y) velocities of the mover (with noise) [m/s]
            - the current (x, y, yaw) set pose of the mover [m, m, rad] (the yaw is not wrapped)
            - the current (x, y) position of the rail mount of the connection of the mover [m]
            - whether the rail mount is parked in a station (1.0) or welded to the mover (0.0)
        """
        mover_qpos = self.get_mover_qpos(mover_names=self.mover_names, add_noise=True)
        mover_yaws = rotations_utils.vec_quat2euler(mover_qpos[:, 3:])[:, 2]
        mover_xy_velos = self.get_mover_qvel(mover_names=self.mover_names, add_noise=True)[:, :2]
        observation = np.concatenate(
            [
                mover_qpos[:, :2],
                mover_yaws[:, np.newaxis],
                mover_xy_velos,
                self.mover_setposes[:, [0, 1, 5]],
                self.get_rail_mount_xy(),
                (self.mover_station_indices >= 0).astype(np.float64)[:, np.newaxis],
            ],
            axis=1,
        )
        assert observation.shape == (self.num_movers, self.NUM_OBS_PER_MOVER)
        return observation.flatten()

    def _preprocess_info_dict(self, info: np.ndarray | dict[str, Any] | None) -> tuple[bool, np.ndarray]:
        """Extract whether the info is batched and whether there is a collision (see ``COLLISION_KEYS``) from the info dictionary.

        :param info: the info dictionary or an array of info dictionaries (batch, e.g. from a HER replay buffer)
        :return: whether the info is batched, i.e. an array of info dictionaries (bool), and a numpy array of shape (batch_size,)
            containing whether there is a collision (bool)
        """
        assert info is not None
        is_batch = isinstance(info, np.ndarray)
        infos = list(info) if is_batch else [info]
        collisions = np.array([any(bool(info_i[key]) for key in self.COLLISION_KEYS) for info_i in infos], dtype=bool)
        return is_batch, collisions

    def compute_reward(
        self, achieved_goal: np.ndarray | None = None, desired_goal: np.ndarray | None = None, info: dict[str, Any] | None = None
    ) -> np.ndarray | float:
        """Compute the immediate reward: ``reward_collision`` in case of a collision, ``reward_per_step`` otherwise.

        :param achieved_goal: not used in this environment, defaults to None
        :param desired_goal: not used in this environment, defaults to None
        :param info: the info dictionary or an array of info dictionaries, defaults to None
        :return: a single float value or (if ``info`` is an array) a numpy array of shape (batch_size,) containing the immediate
            rewards
        """
        is_batch, collisions = self._preprocess_info_dict(info=info)
        reward = np.where(collisions, self.reward_collision, self.reward_per_step).astype(np.float64)
        return reward if is_batch else float(reward[0])

    def compute_terminated(
        self, achieved_goal: np.ndarray | None = None, desired_goal: np.ndarray | None = None, info: dict[str, Any] | None = None
    ) -> np.ndarray | bool:
        """Check whether a terminal state is reached, i.e. whether there is a mover, wall or station proxy collision.

        :param achieved_goal: not used in this environment, defaults to None
        :param desired_goal: not used in this environment, defaults to None
        :param info: the info dictionary or an array of info dictionaries, defaults to None
        :return: a single bool value or (if ``info`` is an array) a numpy array of shape (batch_size,) containing whether a terminal
            state is reached
        """
        is_batch, collisions = self._preprocess_info_dict(info=info)
        return collisions if is_batch else bool(collisions[0])

    def compute_truncated(
        self, achieved_goal: np.ndarray | None = None, desired_goal: np.ndarray | None = None, info: dict[str, Any] | None = None
    ) -> np.ndarray | bool:
        """Check whether the truncation condition is satisfied. The episode is never truncated by the environment itself (use a
        ``TimeLimit`` wrapper to limit the episode length).

        :param achieved_goal: not used in this environment, defaults to None
        :param desired_goal: not used in this environment, defaults to None
        :param info: the info dictionary or an array of info dictionaries, defaults to None
        :return: a single bool value or (if ``info`` is an array) a numpy array of shape (batch_size,) containing False
        """
        is_batch, collisions = self._preprocess_info_dict(info=info)
        return np.zeros(collisions.shape, dtype=bool) if is_batch else False


if __name__ == '__main__':
    # random actions until a collision occurs (Gymnasium API)
    env = ReconfigurationRLEnv(magbot_types=('sixD_platform',), randomize_magbot_station_pos=True, render_mode='human')
    try:
        for episode in range(50):
            observation, info = env.reset(seed=episode)
            terminated, num_steps = False, 0
            while not terminated and num_steps < 200:
                observation, reward, terminated, truncated, info = env.step(0.25 * env.action_space.sample())
                num_steps += 1
            print(f'episode {episode}: {num_steps} steps, info: {info}')
    finally:
        env.close()

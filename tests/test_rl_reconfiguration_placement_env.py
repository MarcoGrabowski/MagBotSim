import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from magbotsim.rl_envs.magbots.reconfiguration_placement_env import ReconfigurationPlacementEnv

# Tests of the goal-conditioned MagBot placement task (HER-compatible). All tests that use the env fixture are run for both MagBot types.

COLLISION_KEYS = {'mover_collision', 'wall_collision', 'station_proxy_collision', 'rail_rod_collision', 'station_pin_collision'}
# initial mover positions in connection order (6D-Platform MagBot: connection 0 (mover b) is parked at the +x side of the station)
INITIAL_MOVER_XY_POS = {
    'gripper': np.array([[0.250, 0.150], [0.710, 0.150], [0.480, 0.150]]),  # [m]
    'sixD_platform': np.array([[0.710, 0.150], [0.250, 0.150]]),  # [m]
}


@pytest.fixture(params=ReconfigurationPlacementEnv.MAGBOT_TYPES)
def env(request: pytest.FixtureRequest) -> ReconfigurationPlacementEnv:
    env = ReconfigurationPlacementEnv(magbot_types=(request.param,), render_mode=None)
    yield env
    env.close()


@pytest.mark.filterwarnings('ignore:.*Box observation space.*infinity')
def test_check_env(env: ReconfigurationPlacementEnv) -> None:
    check_env(env, skip_render_check=True)


@pytest.mark.parametrize('magbot_name, magbot_type', [('Gripper', 'gripper'), ('SixDPlatform', 'sixD_platform')])
def test_registered_envs(magbot_name: str, magbot_type: str) -> None:
    env = gym.make(f'ReconfigurationPlacement{magbot_name}Env-v0', render_mode=None)
    assert env.spec is not None and env.spec.max_episode_steps == 100
    assert env.unwrapped.magbot_type == magbot_type
    env.close()


def test_reset_obs_and_info(env: ReconfigurationPlacementEnv) -> None:
    obs, info = env.reset(seed=0)
    assert set(obs.keys()) == {'observation', 'achieved_goal', 'desired_goal'}
    assert env.observation_space.contains(obs)
    assert set(info.keys()) == COLLISION_KEYS | {'is_success'}
    assert all(value is False for value in info.values())
    # the MagBot is parked in its station, the goal is shown by the goal site
    assert np.allclose(obs['achieved_goal'], env.station_mover_poses[0, :, :2].mean(axis=0), atol=1e-4)
    assert np.allclose(obs['desired_goal'], env.goal_xy)
    assert np.allclose(env.data.site_xpos[env.site_id_goal, :2], env.goal_xy)


def test_goal_sampling(env: ReconfigurationPlacementEnv) -> None:
    goals = []
    for seed in range(30):
        obs, _ = env.reset(seed=seed)
        assert env.goal_xy_is_valid(obs['desired_goal'])
        # the goal is not reached at the beginning of an episode
        assert np.linalg.norm(obs['desired_goal'] - obs['achieved_goal']) > 5 * env.threshold_pos
        goals.append(obs['desired_goal'])
    assert len({tuple(goal) for goal in goals}) == len(goals)
    # same seed -> same goal; a specified goal is used
    assert np.allclose(env.reset(seed=3)[0]['desired_goal'], goals[3])
    obs, _ = env.reset(seed=0, options={'goal_xy': np.array([0.45, 0.25])})
    assert np.allclose(obs['desired_goal'], [0.45, 0.25])
    # the goal must not overlap the station
    assert not env.goal_xy_is_valid(env.station_mover_poses[0, :, :2].mean(axis=0))


def test_her_compatible_reward(env: ReconfigurationPlacementEnv) -> None:
    # collect a short episode
    obs, info = env.reset(seed=1)
    env.action_space.seed(1)
    achieved_goals, desired_goals, infos = [], [], []
    rewards = []
    for _ in range(8):
        obs, reward, terminated, _, info = env.step(0.25 * env.action_space.sample())
        achieved_goals.append(obs['achieved_goal'])
        desired_goals.append(obs['desired_goal'])
        infos.append(info)
        rewards.append(reward)
        if terminated:
            break
    achieved_goals, desired_goals, infos = np.array(achieved_goals), np.array(desired_goals), np.array(infos)

    # batched rewards (as computed by a HER replay buffer) equal the rewards returned by step()
    batch_rewards = env.compute_reward(achieved_goals, desired_goals, infos)
    assert isinstance(batch_rewards, np.ndarray) and batch_rewards.shape == (len(infos),)
    assert np.allclose(batch_rewards, rewards)
    # relabeling with the achieved goals -> goal reached (reward 0) unless there is a collision
    collisions = np.array([any(info_i[key] for key in env.COLLISION_KEYS) for info_i in infos])
    relabeled_rewards = env.compute_reward(achieved_goals, achieved_goals.copy(), infos)
    assert np.allclose(relabeled_rewards, np.where(collisions, env.reward_collision, 0.0))
    # a batch of size 1 returns an array, a single transition a float
    assert env.compute_reward(achieved_goals[:1], achieved_goals[:1], infos[:1]).shape == (1,)
    assert isinstance(env.compute_reward(achieved_goals[0], desired_goals[0], infos[0]), float)
    assert env.compute_terminated(achieved_goals, desired_goals, infos).shape == (len(infos),)
    assert env.compute_truncated(achieved_goals, desired_goals, infos).shape == (len(infos),)

    # a collision is penalized even if the goal is reached
    collision_info = {**dict.fromkeys(COLLISION_KEYS, False), 'is_success': False, 'station_proxy_collision': True}
    assert env.compute_reward(achieved_goals[0], achieved_goals[0], collision_info) == env.reward_collision
    # the batch sizes of the goals and of the infos must match
    with pytest.raises(AssertionError):
        env.compute_reward(np.concatenate([achieved_goals, achieved_goals]), np.concatenate([desired_goals, desired_goals]), infos)


def drive(env: ReconfigurationPlacementEnv, target_setposes: np.ndarray, speed: float, hold_steps: int = 10) -> tuple[dict, float, dict]:
    """Drive the mover set poses (x, y, yaw) linearly to the target set poses using the given fraction of the maximum velocities and
    hold them. All movers arrive simultaneously, i.e. a carried MagBot is moved rigidly.
    """
    obs, reward, info = {}, 0.0, {}
    for _ in range(0, 1000):
        delta = target_setposes[:, [0, 1, 5]] - env.mover_setposes[:, [0, 1, 5]]
        if np.abs(delta).max() < 1e-9:
            break
        max_delta = np.array([env.max_delta_xy, env.max_delta_xy, env.max_delta_yaw])
        scale = max(
            np.linalg.norm(delta[:, :2], axis=1).max() / (speed * env.max_delta_xy),
            np.abs(delta[:, 2]).max() / (speed * env.max_delta_yaw),
            1.0,
        )
        obs, reward, terminated, _, info = env.step((delta / max_delta / scale).flatten())
        assert not terminated, info
    for _ in range(0, hold_steps):
        obs, reward, terminated, _, info = env.step(np.zeros(env.action_space.shape))
        assert not terminated, info
    return obs, reward, info


def test_scripted_pick_and_place(env: ReconfigurationPlacementEnv) -> None:
    obs, info = env.reset(seed=2, options={'mover_start_xy_pos': INITIAL_MOVER_XY_POS[env.magbot_type]})
    assert not info['is_success']
    goal_xy = obs['desired_goal']

    # mover set poses of the pick-up (see the example in reconfiguration_env.py), connection idx_c is attached to mover idx_c
    station = env.stations[0]
    rot_station = station.rotation_matrix_xy()
    station_setposes = env.station_mover_poses[0].copy()
    station_setposes[:, 2] = env.initial_mover_z_pos
    station_setposes[:, :2] = env.get_rail_mount_xy()
    entrance_setposes = station_setposes.copy()
    entrance_setposes[:, :2] += np.stack([np.zeros(env.num_movers), 0.145 * station.MOVER_ENTRY_SIGNS], axis=1) @ rot_station.T
    pre_station_setposes = entrance_setposes.copy()
    pre_station_setposes[:, :2] += rot_station @ np.array([0.2, 0.0])
    # take-out along the x-axis of the station, then the MagBot is moved to the goal
    taken_out_setposes = station_setposes.copy()
    taken_out_setposes[:, :2] += rot_station @ np.array([0.25, 0.0])
    goal_setposes = taken_out_setposes.copy()
    goal_setposes[:, :2] += goal_xy - taken_out_setposes[:, :2].mean(axis=0)

    drive(env, pre_station_setposes, speed=0.15)
    drive(env, entrance_setposes, speed=0.15)
    drive(env, station_setposes, speed=0.05)
    assert (env.mover_station_indices == 0).all()
    obs, reward, info = drive(env, taken_out_setposes, speed=0.15)
    assert (env.mover_station_indices == -1).all()
    assert not info['is_success'] and reward == env.reward_per_step
    obs, reward, info = drive(env, goal_setposes, speed=0.15)
    assert np.linalg.norm(obs['achieved_goal'] - goal_xy) <= env.threshold_pos
    assert info['is_success'] and reward == 0.0

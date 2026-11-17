import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from magbotsim.rl_envs.magbots.reconfiguration_rl_env import ReconfigurationRLEnv

# Tests of the RL variant (Gymnasium API) of the reconfiguration env. All tests that use the env fixture are run for both MagBot types.

COLLISION_KEYS = {
    'mover_collision',
    'wall_collision',
    'station_proxy_collision',
    'rail_rod_collision',
    'station_pin_collision',
    'rotation_violation',
}
# initial mover positions in connection order (6D-Platform MagBot: connection 0 (mover b) is parked at the +x side of the station)
INITIAL_MOVER_XY_POS = {
    'gripper': np.array([[0.250, 0.150], [0.710, 0.150], [0.480, 0.150]]),  # [m]
    'sixD_platform': np.array([[0.710, 0.150], [0.250, 0.150]]),  # [m]
}
STATION_POSES = np.array([[0.480, 0.720, 0.01, np.deg2rad(-90)]])  # (x, y, z, yaw) of the station anchor


@pytest.fixture(params=ReconfigurationRLEnv.MAGBOT_TYPES)
def magbot_type(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=ReconfigurationRLEnv.MAGBOT_TYPES)
def env(request: pytest.FixtureRequest) -> ReconfigurationRLEnv:
    env = ReconfigurationRLEnv(magbot_types=(request.param,), station_poses=STATION_POSES, render_mode=None)
    yield env
    env.close()


def reset_fixed(env: ReconfigurationRLEnv, seed: int = 0) -> tuple[np.ndarray, dict[str, bool]]:
    """Reset the env with the fixed initial mover positions of its MagBot type."""
    return env.reset(seed=seed, options={'mover_start_xy_pos': INITIAL_MOVER_XY_POS[env.magbot_type]})


def split_obs(env: ReconfigurationRLEnv, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the observation into the values per mover and the values per connection."""
    num_obs_movers = env.num_movers * env.num_obs_per_mover
    obs_movers = obs[:num_obs_movers].reshape((env.num_movers, env.num_obs_per_mover))
    obs_connections = obs[num_obs_movers:].reshape((env.num_magbots * env.num_connections, ReconfigurationRLEnv.NUM_OBS_PER_CONNECTION))
    return obs_movers, obs_connections


@pytest.mark.filterwarnings('ignore:.*Box observation space.*infinity')
@pytest.mark.parametrize('num_extra_movers', [0, 2])
def test_check_env(magbot_type: str, num_extra_movers: int) -> None:
    num_movers = ReconfigurationRLEnv.NUM_CONNECTIONS[magbot_type] + num_extra_movers
    env = ReconfigurationRLEnv(magbot_types=(magbot_type,), num_movers=num_movers, station_poses=STATION_POSES, render_mode=None)
    check_env(env, skip_render_check=True)
    assert env.num_movers == num_movers
    assert env.action_space.shape == (env.num_movers * 3,)
    assert env.num_obs_per_mover == ReconfigurationRLEnv.NUM_OBS_PER_MOVER_BASE + env.num_connections
    num_obs = env.num_movers * env.num_obs_per_mover + env.num_connections * ReconfigurationRLEnv.NUM_OBS_PER_CONNECTION
    assert env.observation_space.shape == (num_obs,)
    env.close()


def test_reset_info_and_obs(env: ReconfigurationRLEnv) -> None:
    obs, info = env.reset(seed=0)
    assert set(info.keys()) == COLLISION_KEYS
    assert all(value is False for value in info.values())
    assert env.observation_space.contains(obs)
    obs_movers, obs_connections = split_obs(env, obs)
    # set poses at the start positions with a yaw of 0, no mover attached
    assert np.allclose(obs_movers[:, 5:7], env.mover_start_xy_pos)
    assert np.allclose(obs_movers[:, 7], 0.0)
    assert np.all(obs_movers[:, 8:] == 0.0)
    # rail mounts parked in the station, the docking yaws are the yaws of the movers in the station
    assert np.allclose(obs_connections[:, :2], env.get_rail_mount_xy())
    assert np.allclose(np.angle(np.exp(1j * (obs_connections[:, 2] - env.station_mover_poses[0, :, 5]))), 0.0, atol=1e-4)
    assert np.all(obs_connections[:, 3] == 1.0)


def test_actions_move_set_poses(env: ReconfigurationRLEnv) -> None:
    reset_fixed(env)
    setposes = env.mover_setposes.copy()
    # zero action -> set poses unchanged
    _, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
    assert np.allclose(env.mover_setposes, setposes)
    assert reward == env.reward_per_step and not terminated and not truncated
    assert set(info.keys()) == COLLISION_KEYS
    # full +x and -yaw action -> set poses move by the maximum increments
    action = np.zeros((env.num_movers, 3))
    action[:, 0], action[:, 2] = 1.0, -1.0
    env.step(action.flatten())
    expected = setposes.copy()
    expected[:, 0] += env.max_delta_xy
    expected[:, 5] -= env.max_delta_yaw
    assert np.allclose(env.mover_setposes, expected)
    # the norm of the (x,y)-increments is clipped to 1
    action = np.zeros((env.num_movers, 3))
    action[:, :2] = 1.0
    prev_setposes = env.mover_setposes.copy()
    env.step(action.flatten())
    assert np.allclose(np.linalg.norm(env.mover_setposes[:, :2] - prev_setposes[:, :2], axis=1), env.max_delta_xy)
    # the movers follow the set poses
    for _ in range(10):
        env.step(np.zeros(env.action_space.shape))
    mover_xy = env.get_mover_qpos(mover_names=env.mover_names)[:, :2]
    assert np.abs(mover_xy - env.mover_setposes[:, :2]).max() < 0.002


def test_wall_collision_terminates(env: ReconfigurationRLEnv) -> None:
    reset_fixed(env)
    action = np.zeros((env.num_movers, 3))
    action[0, 1] = -1.0  # mover 0 drives towards the wall at y=0
    terminated, info, reward = False, {}, 0.0
    for _ in range(50):
        _, reward, terminated, _, info = env.step(action.flatten())
        if terminated:
            break
    assert terminated
    assert info['wall_collision']
    assert reward == env.reward_collision


def test_rotation_violation_terminates(env: ReconfigurationRLEnv) -> None:
    reset_fixed(env)
    # the start position of mover 0 is not a rotation point, i.e. its yaw may deviate at most 10 deg from the nearest multiple of 90 deg
    assert not env.full_rotation_possible(qpos=env.get_mover_qpos(mover_names=env.mover_names)[:1]).any()
    action = np.zeros((env.num_movers, 3))
    action[0, 2] = 1.0  # mover 0 rotates at the maximum yaw velocity
    terminated, info, reward, num_steps = False, {}, 0.0, 0
    while not terminated and num_steps < 50:
        _, reward, terminated, _, info = env.step(action.flatten())
        num_steps += 1
    assert terminated
    assert {key for key in COLLISION_KEYS if info[key]} == {'rotation_violation'}
    assert reward == env.reward_collision
    # the yaw exceeds 10 deg after 3 steps (3.6 deg per step), the actual yaw lags behind the set yaw
    assert 3 <= num_steps <= 6


def test_initial_mover_xy_pos_reused_at_reset() -> None:
    kwargs = {'magbot_types': ('sixD_platform',), 'station_poses': STATION_POSES, 'render_mode': None}
    env = ReconfigurationRLEnv(initial_mover_xy_pos=INITIAL_MOVER_XY_POS, **kwargs)
    for seed in range(3):
        env.reset(seed=seed)
        assert np.allclose(env.mover_start_xy_pos, INITIAL_MOVER_XY_POS['sixD_platform'])
    env.close()
    # without fixed initial positions, the start positions are sampled at each reset
    env = ReconfigurationRLEnv(**kwargs)
    env.reset(seed=0)
    start_xy_pos_0 = env.mover_start_xy_pos.copy()
    env.reset(seed=1)
    assert not np.allclose(env.mover_start_xy_pos, start_xy_pos_0)
    env.close()


def test_batched_reward_and_termination(env: ReconfigurationRLEnv) -> None:
    no_collision = dict.fromkeys(COLLISION_KEYS, False)
    infos = np.array(
        [
            no_collision,
            {**no_collision, 'wall_collision': True},
            {**no_collision, 'station_proxy_collision': True, 'rail_rod_collision': True},
            {**no_collision, 'rotation_violation': True},
        ]
    )
    assert np.allclose(
        env.compute_reward(info=infos), [env.reward_per_step, env.reward_collision, env.reward_collision, env.reward_collision]
    )
    assert np.array_equal(env.compute_terminated(info=infos), [False, True, True, True])
    assert np.array_equal(env.compute_truncated(info=infos), [False, False, False, False])

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from magbotsim.rl_envs.magbots.reconfiguration_rl_env import ReconfigurationRLEnv

# Tests of the RL variant (Gymnasium API) of the reconfiguration env. All tests that use the env fixture are run for both MagBot types.

COLLISION_KEYS = {'mover_collision', 'wall_collision', 'station_proxy_collision', 'rail_rod_collision', 'station_pin_collision'}
# initial mover positions in connection order (6D-Platform MagBot: connection 0 (mover b) is parked at the +x side of the station)
INITIAL_MOVER_XY_POS = {
    'gripper': np.array([[0.250, 0.150], [0.710, 0.150], [0.480, 0.150]]),  # [m]
    'sixD_platform': np.array([[0.710, 0.150], [0.250, 0.150]]),  # [m]
}
STATION_POSES = np.array([[0.480, 0.720, 0.01, np.deg2rad(-90)]])  # (x, y, z, yaw) of the station anchor


@pytest.fixture(params=ReconfigurationRLEnv.MAGBOT_TYPES)
def env(request: pytest.FixtureRequest) -> ReconfigurationRLEnv:
    env = ReconfigurationRLEnv(magbot_types=(request.param,), station_poses=STATION_POSES, render_mode=None)
    yield env
    env.close()


def reset_fixed(env: ReconfigurationRLEnv, seed: int = 0) -> tuple[np.ndarray, dict[str, bool]]:
    """Reset the env with the fixed initial mover positions of its MagBot type."""
    return env.reset(seed=seed, options={'mover_start_xy_pos': INITIAL_MOVER_XY_POS[env.magbot_type]})


@pytest.mark.filterwarnings('ignore:.*Box observation space.*infinity')
def test_check_env(env: ReconfigurationRLEnv) -> None:
    check_env(env, skip_render_check=True)
    assert env.action_space.shape == (env.num_movers * 3,)
    assert env.observation_space.shape == (env.num_movers * ReconfigurationRLEnv.NUM_OBS_PER_MOVER,)


def test_reset_info_and_obs(env: ReconfigurationRLEnv) -> None:
    obs, info = env.reset(seed=0)
    assert set(info.keys()) == COLLISION_KEYS
    assert all(value is False for value in info.values())
    assert env.observation_space.contains(obs)
    obs_movers = obs.reshape((env.num_movers, ReconfigurationRLEnv.NUM_OBS_PER_MOVER))
    # set poses at the start positions with a yaw of 0
    assert np.allclose(obs_movers[:, 5:7], env.mover_start_xy_pos)
    assert np.allclose(obs_movers[:, 7], 0.0)
    # rail mounts parked in the station
    assert np.allclose(obs_movers[:, 8:10], env.get_rail_mount_xy())
    assert np.all(obs_movers[:, 10] == 1.0)


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
        ]
    )
    assert np.allclose(env.compute_reward(info=infos), [env.reward_per_step, env.reward_collision, env.reward_collision])
    assert np.array_equal(env.compute_terminated(info=infos), [False, True, True])
    assert np.array_equal(env.compute_truncated(info=infos), [False, False, False])

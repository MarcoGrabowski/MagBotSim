from pathlib import Path

import mujoco
import numpy as np
import pytest

from magbotsim import PlatformStation
from magbotsim.envs_without_rl_api.magbots import reconfiguration_transport_env as transport_env_module
from magbotsim.envs_without_rl_api.magbots.reconfiguration_env import ReconfigurationEnv
from magbotsim.envs_without_rl_api.magbots.reconfiguration_transport_env import (
    ReconfigurationTransportEnv,
    run_magbot_pickup,
    run_transport_cycle,
)

# Tests of the reconfiguration transport env: two movers pick up the 6D-Platform MagBot from the station and transport products from
# the supply rack (bottom long side) to the bin (right short side). The product is respawned in the supply rack after each delivery.

NUM_CYCLES = 20
INITIAL_MOVER_XY_POS = np.array([[0.130, 0.100], [0.860, 0.100]])  # [m]
COLLISION_KEYS = {'mover_collision', 'wall_collision', 'station_proxy_collision', 'rail_rod_collision', 'station_pin_collision'}
PRODUCT_KEYS = {'product_delivered', 'product_dropped'}


@pytest.fixture
def env() -> ReconfigurationTransportEnv:
    env = ReconfigurationTransportEnv(initial_mover_xy_pos=INITIAL_MOVER_XY_POS, render_mode=None)
    yield env
    env.close()


def initial_poses(env: ReconfigurationTransportEnv) -> np.ndarray:
    """Return the set poses of the movers at their start positions (numpy array of shape (num_movers, 6))."""
    poses = np.zeros((env.num_movers, 6))
    poses[:, :2] = env.mover_start_xy_pos
    poses[:, 2] = env.initial_mover_z_pos
    return poses


def hold(env: ReconfigurationTransportEnv, duration: float) -> list[dict[str, bool]]:
    """Hold the movers at their start positions for the given duration [s] and return the info dicts of all steps."""
    return [
        env.step(mover_setposes=initial_poses(env), num_cycles=NUM_CYCLES) for _ in range(int(duration / (env.cycle_time * NUM_CYCLES)))
    ]


def set_product_pos(env: ReconfigurationTransportEnv, pos: np.ndarray) -> None:
    """Teleport the product to a position (at rest, no rotation)."""
    env.data.qpos[env.product_qpos_adr : env.product_qpos_adr + 7] = np.concatenate([pos, [1.0, 0.0, 0.0, 0.0]])
    env.data.qvel[env.product_qvel_adr : env.product_qvel_adr + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def test_scene(env: ReconfigurationTransportEnv) -> None:
    assert env.num_movers == 2
    assert env.num_connections == 2
    # same station as in the reconfiguration env
    assert len(env.stations) == 1 and isinstance(env.stations[0], PlatformStation)
    assert np.allclose(env.STATION_POSES, ReconfigurationEnv.DEFAULT_STATION_POSES)
    # the MagBot is parked in the station, the movers are free at their start positions
    assert (env.connection_station_indices == 0).all() and (env.connection_mover_indices == -1).all()
    assert np.allclose(env.get_mover_qpos(mover_names=env.mover_names)[:, :2], INITIAL_MOVER_XY_POS)
    # the product is in the supply rack
    assert np.allclose(env.get_product_qpos()[:3], env.PRODUCT_SUPPLY_POS)
    assert not env.is_product_in_bin() and not env.is_product_dropped()
    assert env.num_delivered_products == 0 and env.num_dropped_products == 0


def test_no_collision_after_reset(env: ReconfigurationTransportEnv) -> None:
    info = env.step(mover_setposes=initial_poses(env), num_cycles=NUM_CYCLES)
    assert set(info.keys()) == COLLISION_KEYS | PRODUCT_KEYS
    assert all(value is False for value in info.values())


def test_product_rests_in_supply(env: ReconfigurationTransportEnv) -> None:
    infos = hold(env, duration=1.0)
    assert not any(info[key] for info in infos for key in COLLISION_KEYS | PRODUCT_KEYS)
    assert np.allclose(env.get_product_qpos()[:3], env.PRODUCT_SUPPLY_POS, atol=0.001)


def test_product_delivered_and_respawned(env: ReconfigurationTransportEnv) -> None:
    # the product falls into the center of the bin (center of the bin bottom: (0.1, -0.074) in the bin body frame)
    c, s = np.cos(env.BIN_POSE[3]), np.sin(env.BIN_POSE[3])
    bin_center = env.BIN_POSE[:2] + np.array([[c, -s], [s, c]]) @ np.array([0.1, -0.074])
    set_product_pos(env, np.array([*bin_center, env.BIN_POSE[2] + 0.05]))
    assert env.is_product_in_bin() and not env.is_product_dropped()
    infos = hold(env, duration=1.0)
    assert sum(info['product_delivered'] for info in infos) == 1
    assert not any(info['product_dropped'] for info in infos)
    assert env.num_delivered_products == 1 and env.num_dropped_products == 0
    # respawned in the supply rack
    assert np.allclose(env.get_product_qpos()[:3], env.PRODUCT_SUPPLY_POS, atol=0.001)


def test_product_dropped_and_respawned(env: ReconfigurationTransportEnv) -> None:
    # the product falls on the table (free space between the movers)
    set_product_pos(env, np.array([0.48, 0.30, 0.05]))
    assert env.is_product_dropped()
    infos = hold(env, duration=1.0)
    assert sum(info['product_dropped'] for info in infos) == 1
    assert not any(info['product_delivered'] for info in infos)
    assert env.num_delivered_products == 0 and env.num_dropped_products == 1
    assert np.allclose(env.get_product_qpos()[:3], env.PRODUCT_SUPPLY_POS, atol=0.001)
    # the counters are reset with the scene
    env.reset_scene(mover_start_xy_pos=INITIAL_MOVER_XY_POS)
    assert env.num_dropped_products == 0


def test_transport_cycle(env: ReconfigurationTransportEnv) -> None:
    # the movers pick up the MagBot from the station
    infos = run_magbot_pickup(env, num_cycles=NUM_CYCLES)
    assert (env.connection_station_indices == -1).all()
    assert sorted(env.connection_mover_indices[0].tolist()) == [0, 1]
    # one product is transported from the supply rack to the bin without collisions
    infos += run_transport_cycle(env, num_cycles=NUM_CYCLES)
    assert not any(info[key] for info in infos for key in COLLISION_KEYS)
    assert env.num_delivered_products == 1 and env.num_dropped_products == 0
    assert np.allclose(env.get_product_qpos()[:3], env.PRODUCT_SUPPLY_POS, atol=0.001)


def test_random_mover_start_positions() -> None:
    env = ReconfigurationTransportEnv(seed=1, randomize_magbot_station_pos=True, render_mode=None)
    for seed in range(20):
        env.reset_scene(rng=np.random.default_rng(seed))
        assert env.mover_start_xy_pos.shape == (env.num_movers, 2)
        assert not env.check_wall_collision(mover_names=env.mover_names, c_size=env.c_size).any()
        assert not env.check_mover_collision(mover_names=env.mover_names, c_size=env.c_size)
        assert not env.check_station_proxy_collision()
        assert np.abs(env.magbot_station_x_offsets).max() <= env.magbot_station_randomization_x_range
    env.close()


def test_invalid_initial_mover_xy_pos() -> None:
    with pytest.raises(AssertionError):
        ReconfigurationTransportEnv(initial_mover_xy_pos=np.zeros((3, 2)), render_mode=None)


def test_model_xml_saved_next_to_script(env: ReconfigurationTransportEnv) -> None:
    path = env.model_xml_path
    assert path == Path(transport_env_module.__file__).resolve().with_name('reconfiguration_transport_env.xml')
    assert path.read_text() == env.model_xml_str
    # the saved file is a standalone model (absolute asset paths) and equals the model of the environment
    model = mujoco.MjModel.from_xml_path(str(path))
    assert (model.nbody, model.njnt, model.neq, model.nu) == (env.model.nbody, env.model.njnt, env.model.neq, env.model.nu)

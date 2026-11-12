import mujoco
import numpy as np
import pytest

from magbotsim import GripperStation, PlatformStation
from magbotsim.envs_without_rl_api.magbots.reconfiguration_env import ReconfigurationEnv
from magbotsim.utils.rotations_utils import vec_quat2euler

# Tests of the station collision proxies (hidden proxies of the rail mounts, rail rods and station pins) of the reconfiguration env:
# entering a station with the correct entry axis must not cause a collision, whereas entering diagonally must cause a collision.
# Entering correctly is always checked first, i.e. a collision on the diagonal path is not caused by an earlier misalignment.
# All tests that use the env fixture are run for both MagBot types (Gripper MagBot in a GripperStation, 6D-Platform MagBot in a
# PlatformStation).

NUM_CYCLES = 20
# initial mover positions in connection order (6D-Platform MagBot: connection 0 (mover b) is parked at the +x side of the station)
INITIAL_MOVER_XY_POS = {
    'gripper': np.array([[0.250, 0.150], [0.710, 0.150], [0.480, 0.150]]),  # [m]
    'sixD_platform': np.array([[0.710, 0.150], [0.250, 0.150]]),  # [m]
}
STATION_POSES = np.array([[0.480, 0.720, 0.01, np.deg2rad(-90)]])  # (x, y, z, yaw) of the station anchor


@pytest.fixture(params=ReconfigurationEnv.MAGBOT_TYPES)
def env(request: pytest.FixtureRequest) -> ReconfigurationEnv:
    env = ReconfigurationEnv(
        magbot_types=(request.param,), initial_mover_xy_pos=INITIAL_MOVER_XY_POS, station_poses=STATION_POSES, render_mode=None
    )
    yield env
    env.close()


def initial_mover_xy_pos(env: ReconfigurationEnv) -> np.ndarray:
    """Return the initial mover positions of the MagBot type of the env."""
    return INITIAL_MOVER_XY_POS[env.magbot_type]


def entrance_offsets(env: ReconfigurationEnv) -> np.ndarray:
    """Return the offsets of the station entrances (145 mm along the entry axis, on the flat side of each rail mount) in the world frame."""
    station = env.stations[0]
    offsets_station = np.stack([np.zeros(env.num_movers), 0.145 * station.MOVER_ENTRY_SIGNS], axis=1)
    return offsets_station @ station.rotation_matrix_xy().T


@pytest.fixture
def poses(env: ReconfigurationEnv) -> dict[str, np.ndarray]:
    """Mover set poses (numpy arrays of shape (num_movers, 6)) and MagBot set poses (numpy arrays of shape (1, 7) or (1, 6)) of the
    station choreography (see the example in reconfiguration_env.py). Connection idx_c is attached to mover idx_c.
    """
    station = env.stations[0]
    rot_station = station.rotation_matrix_xy()
    p = {'station': station.get_mover_poses(z_m=env.initial_mover_z_pos)}
    # entrance: the movers left the station along the entry axis (y-axis of the station body frame)
    p['entrance'] = p['station'].copy()
    p['entrance'][:, :2] += entrance_offsets(env)
    # pre-station: 200 mm in front of the entrance (x-axis of the station body frame)
    p['pre_station'] = p['entrance'].copy()
    p['pre_station'][:, :2] += np.array([0.2, 0.0]) @ rot_station.T
    p['initial'] = p['station'].copy()
    p['initial'][:, :2] = initial_mover_xy_pos(env)
    p['initial'][:, 5] = 0.0
    # MagBot poses: MagBot in the station and taken out of the station by 0.25 m along the x-axis of the station body frame
    center = p['station'].copy()
    center[:, :2] += rot_station @ np.array([0.25, 0.0])
    p['magbot_station'] = env.get_magbot_setpose_from_mover_poses(mover_poses=p['station'])
    p['magbot_center'] = env.get_magbot_setpose_from_mover_poses(mover_poses=center)
    # MagBot shifted by 0.1 m along the y-axis of the station body frame, i.e. not aligned with the station pins
    p['magbot_shifted'] = p['magbot_center'].copy()
    p['magbot_shifted'][0, :2] += rot_station @ np.array([0.0, 0.1])
    return p


COLLISION_KEYS = {'mover_collision', 'wall_collision', 'station_proxy_collision', 'rail_rod_collision', 'station_pin_collision'}


def move_with_info(
    env: ReconfigurationEnv, start: np.ndarray, end: np.ndarray, duration: float, magbot: bool = False, hold: float = 0.5
) -> dict[str, bool]:
    """Move the movers (or the MagBot gripper / platform) linearly from start to end and hold the end pose. The movement is stopped
    at the first station proxy collision.

    :return: the info dict returned by ``step()`` at the first station proxy collision, or of the last step if no station proxy
        collision occurred
    """
    dt = env.cycle_time * NUM_CYCLES
    setposes = [start + s * (end - start) for s in np.linspace(0.0, 1.0, int(duration / dt))] + [end] * int(hold / dt)
    info = dict.fromkeys(COLLISION_KEYS, False)
    for setpose in setposes:
        mover_setposes = env.get_mover_poses_from_magbot_setpose(magbot_setposes=setpose) if magbot else setpose
        info = env.step(mover_setposes=mover_setposes, num_cycles=NUM_CYCLES)
        # step() stops at the first collision, i.e. the colliding contacts are still present after step()
        assert info['station_proxy_collision'] == env.check_station_proxy_collision()
        if info['station_proxy_collision']:
            break
    return info


def move(env: ReconfigurationEnv, start: np.ndarray, end: np.ndarray, duration: float, magbot: bool = False, hold: float = 0.5) -> bool:
    """Move the movers (or the MagBot gripper / platform) linearly from start to end and hold the end pose.

    :return: True if a station proxy collision occurred, False otherwise
    """
    return move_with_info(env, start, end, duration, magbot=magbot, hold=hold)['station_proxy_collision']


def proxy_contact_names(env: ReconfigurationEnv) -> list[str]:
    """Return the names of both geoms of each contact involving a collision proxy, joined by '|'."""
    contacts = env.data.contact.geom[: env.data.ncon]
    return [
        '|'.join(mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, idx_geom) for idx_geom in pair)
        for pair in contacts
        if np.isin(pair, env.geom_ids_proxy).any()
    ]


def test_no_collision_after_reset(env: ReconfigurationEnv) -> None:
    names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, idx_geom) for idx_geom in env.geom_ids_proxy]
    assert sum('_rail_rod' in name for name in names) == env.num_connections
    assert sum('_pins_' in name for name in names) == env.num_connections * 2
    for idx_c in range(env.num_connections):  # each rail mount proxy has jaws (lower layer) and an upper layer
        assert any(f'rail_mount_{idx_c}_0_proxy_lower_' in name for name in names)
        assert any(f'rail_mount_{idx_c}_0_proxy_upper_' in name for name in names)
    assert not env.check_station_proxy_collision()


def test_empty_movers_diagonal_entry_collides(env: ReconfigurationEnv, poses: dict[str, np.ndarray]) -> None:
    # the empty movers drive in front of the station (the MagBot is parked in the station)
    assert not move(env, poses['initial'], poses['pre_station'], duration=1.5)
    assert (env.mover_station_indices == 0).all()
    # the empty movers enter the station diagonally instead of along the entry axis -> the rail rods hit the jaws of the rail mounts
    info = move_with_info(env, poses['pre_station'], poses['station'], duration=1.5)
    assert info['station_proxy_collision'] and info['rail_rod_collision'] and not info['station_pin_collision']
    assert env.check_station_proxy_collision()
    assert any('rail_rod' in names and '_lower_' in names for names in proxy_contact_names(env))


def test_magbot_diagonal_entry_collides(env: ReconfigurationEnv, poses: dict[str, np.ndarray]) -> None:
    # the empty movers enter the station along the entry axis and take the MagBot out of the station
    assert not move(env, poses['initial'], poses['pre_station'], duration=1.5)
    assert not move(env, poses['pre_station'], poses['entrance'], duration=1.5)
    assert not move(env, poses['entrance'], poses['station'], duration=1.5)
    assert not move(env, poses['magbot_station'], poses['magbot_center'], duration=1.5, magbot=True)
    assert (env.mover_station_indices == -1).all()
    # the MagBot is shifted sideways (away from the station)
    assert not move(env, poses['magbot_center'], poses['magbot_shifted'], duration=1.0, magbot=True)
    # the MagBot is moved back into the station diagonally -> the rail mounts hit the station pins
    info = move_with_info(env, poses['magbot_shifted'], poses['magbot_station'], duration=1.5, magbot=True)
    assert info['station_proxy_collision'] and info['station_pin_collision']
    assert env.check_station_proxy_collision()
    assert any('_pins_' in names and '_upper_' in names for names in proxy_contact_names(env))


# Tests of the info dict returned by step(), which contains the collision types.


def test_step_returns_collision_info(env: ReconfigurationEnv) -> None:
    info = env.step(mover_setposes=initial_poses(env), num_cycles=NUM_CYCLES)
    assert set(info.keys()) == COLLISION_KEYS
    assert all(value is False for value in info.values())


@pytest.mark.parametrize('collision_type', ['wall_collision', 'mover_collision'])
def test_step_info_mover_and_wall_collision(env: ReconfigurationEnv, collision_type: str) -> None:
    start = initial_poses(env)
    end = start.copy()
    if collision_type == 'wall_collision':
        end[0, 1] = -0.1  # mover 0 drives through the wall at y=0
    else:
        end[0, :2] = start[1, :2]  # mover 0 drives onto mover 1
    dt = env.cycle_time * NUM_CYCLES
    info = dict.fromkeys(COLLISION_KEYS, False)
    for s in np.linspace(0.0, 1.0, int(1.5 / dt)):
        info = env.step(mover_setposes=start + s * (end - start), num_cycles=NUM_CYCLES)
        if any(info.values()):
            break
    assert info[collision_type]
    assert not info['station_proxy_collision']


# Tests of the randomization of the reconfiguration env: random collision-free mover start positions and random shift of the MagBot
# along the x-axis of the station (take-out axis, i.e. along the station pins).


def mover_contacts(env: ReconfigurationEnv) -> list[str]:
    """Return the names of both geoms of each contact between a mover (incl. its attached bodies) and any body that is not a mover."""
    mover_ids = {mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in env.mover_names}
    root_ids = env.model.body_rootid[env.model.geom_bodyid]
    return [
        '|'.join(mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, idx_geom) for idx_geom in pair)
        for pair in env.data.contact.geom[: env.data.ncon]
        if (root_ids[pair[0]] in mover_ids) != (root_ids[pair[1]] in mover_ids)
    ]


def station_frame_offsets(env: ReconfigurationEnv, rail_mount_xy_ref: np.ndarray) -> np.ndarray:
    """Return the offsets of the real rail mount positions w.r.t. the reference positions in the station frame [m]."""
    return (env.get_rail_mount_xy() - rail_mount_xy_ref) @ env.stations[0].rotation_matrix_xy()


def pickup_poses(env: ReconfigurationEnv, poses: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mover set poses and MagBot set pose at the real (shifted) positions of the parked rail mounts (connection idx_c = mover idx_c)."""
    rot_station = env.stations[0].rotation_matrix_xy()
    p = {'station': poses['station'].copy()}
    p['station'][:, :2] = env.get_rail_mount_xy()
    p['entrance'] = p['station'].copy()
    p['entrance'][:, :2] += entrance_offsets(env)
    p['pre_station'] = p['entrance'].copy()
    p['pre_station'][:, :2] += np.array([0.2, 0.0]) @ rot_station.T
    p['magbot_station'] = env.get_magbot_setpose_from_mover_poses(mover_poses=p['station'])
    return p


@pytest.mark.parametrize('magbot_type', ReconfigurationEnv.MAGBOT_TYPES)
def test_random_mover_start_positions(magbot_type: str) -> None:
    kwargs = {'magbot_types': (magbot_type,), 'station_poses': STATION_POSES, 'randomize_magbot_station_pos': True, 'render_mode': None}
    env = ReconfigurationEnv(seed=1, **kwargs)
    for seed in range(50):
        env.reset_scene(rng=np.random.default_rng(seed))
        assert env.mover_start_xy_pos.shape == (env.num_movers, 2)
        assert not env.check_wall_collision(mover_names=env.mover_names, c_size=env.c_size).any()
        assert not env.check_mover_collision(mover_names=env.mover_names, c_size=env.c_size)
        assert not env.check_station_proxy_collision()
        assert not mover_contacts(env)
    env.close()

    # same seed -> same start positions and MagBot offsets
    env_1 = ReconfigurationEnv(seed=3, **kwargs)
    env_2 = ReconfigurationEnv(seed=3, **kwargs)
    assert np.allclose(env_1.initial_mover_xy_pos, env_2.initial_mover_xy_pos)
    assert np.allclose(env_1.magbot_station_x_offsets, env_2.magbot_station_x_offsets)
    assert np.abs(env_1.magbot_station_x_offsets).max() <= env_1.magbot_station_randomization_x_range
    env_1.close()
    env_2.close()


def test_explicit_start_positions_are_kept(env: ReconfigurationEnv) -> None:
    assert np.allclose(env.initial_mover_xy_pos, initial_mover_xy_pos(env))
    assert np.allclose(env.get_mover_qpos(mover_names=env.mover_names)[:, :2], initial_mover_xy_pos(env))
    assert np.allclose(env.magbot_station_x_offsets, 0.0)


@pytest.mark.parametrize('x_offset', [-0.005, 0.0023, 0.005])
def test_magbot_station_shift(env: ReconfigurationEnv, poses: dict[str, np.ndarray], x_offset: float) -> None:
    rail_mount_xy_ref = env.get_rail_mount_xy()  # unshifted MagBot
    env.reset_scene(mover_start_xy_pos=initial_mover_xy_pos(env), magbot_station_x_offsets=np.array([x_offset]))
    # all rail mounts are shifted exactly by the offset along the x-axis of the station (no settling required)
    assert np.allclose(station_frame_offsets(env, rail_mount_xy_ref), [[x_offset, 0.0]] * env.num_connections, atol=1e-6)
    assert not env.check_station_proxy_collision()
    assert not mover_contacts(env)
    # the MagBot stays in the shifted pose while the movers are held
    rail_mount_xy_shifted = env.get_rail_mount_xy()
    assert not move(env, poses['initial'], poses['initial'], duration=0.0, hold=0.5)
    assert np.abs(env.get_rail_mount_xy() - rail_mount_xy_shifted).max() < 0.0002


def test_shifted_magbot_pickup_and_putback(env: ReconfigurationEnv, poses: dict[str, np.ndarray]) -> None:
    env.reset_scene(mover_start_xy_pos=initial_mover_xy_pos(env), magbot_station_x_offsets=np.array([0.004]))
    pickup = pickup_poses(env, poses)
    # the empty movers enter the station at the real rail mount positions and take the MagBot out of the station
    assert not move(env, poses['initial'], pickup['pre_station'], duration=1.5)
    assert not move(env, pickup['pre_station'], pickup['entrance'], duration=1.5)
    assert not move(env, pickup['entrance'], pickup['station'], duration=1.5)
    assert (env.mover_station_indices == 0).all()
    assert not move(env, pickup['magbot_station'], poses['magbot_center'], duration=1.5, magbot=True)
    assert (env.mover_station_indices == -1).all()
    # the rail mounts were picked up without a jump, i.e. they stay on the mover adapters
    mover_xy = env.get_mover_qpos(mover_names=env.mover_names)[:, :2]
    assert np.linalg.norm(env.get_rail_mount_xy() - mover_xy, axis=1).max() < 0.0005

    # the MagBot is put back at the nominal station pose and stays there (no re-randomization)
    assert not move(env, poses['magbot_center'], poses['magbot_station'], duration=1.5, magbot=True)
    assert not move(env, poses['station'], poses['entrance'], duration=1.5)
    assert (env.mover_station_indices == 0).all()
    assert np.linalg.norm(env.get_rail_mount_xy() - poses['station'][:, :2], axis=1).max() < 0.0005


def test_putback_off_nominal_does_not_switch(env: ReconfigurationEnv, poses: dict[str, np.ndarray]) -> None:
    # the movers take the MagBot out of the station
    assert not move(env, poses['initial'], poses['pre_station'], duration=1.5)
    assert not move(env, poses['pre_station'], poses['entrance'], duration=1.5)
    assert not move(env, poses['entrance'], poses['station'], duration=1.5)
    assert not move(env, poses['magbot_station'], poses['magbot_center'], duration=1.5, magbot=True)
    assert (env.mover_station_indices == -1).all()
    # the MagBot is put back 3 mm (> station_switch_tol) in front of the nominal station pose
    rot_station = env.stations[0].rotation_matrix_xy()
    magbot_off = poses['magbot_station'].copy()
    magbot_off[0, :2] += rot_station @ np.array([0.003, 0.0])
    station_off = poses['station'].copy()
    station_off[:, :2] += rot_station @ np.array([0.003, 0.0])
    assert not move(env, poses['magbot_center'], magbot_off, duration=1.5, magbot=True)
    # the movers try to leave along the entry axis -> the rail mounts are not welded to the station
    move(env, station_off, station_off + (poses['entrance'] - poses['station']), duration=0.5, hold=0.0)
    assert (env.mover_station_indices == -1).all()


# Tests of the parked MagBot, the MagBot kinematics helpers and the selection of the MagBot type.


def initial_poses(env: ReconfigurationEnv) -> np.ndarray:
    """Mover set poses at the initial mover positions (yaw 0)."""
    poses = env.station_mover_poses[0].copy()
    poses[:, :2] = initial_mover_xy_pos(env)
    poses[:, 5] = 0.0
    return poses


def test_magbot_parked_in_station(env: ReconfigurationEnv) -> None:
    # the rail mounts are parked at the nominal station poses (Gripper MagBot: rail mount 2 is generated at the rounded m2 offset of
    # 0.12 m) and stay there while the movers are held
    assert np.abs(env.get_rail_mount_xy() - env.station_mover_poses[0, :, :2]).max() < 1e-5
    assert not move(env, initial_poses(env), initial_poses(env), duration=0.0, hold=1.0)
    assert np.abs(env.get_rail_mount_xy() - env.station_mover_poses[0, :, :2]).max() < 0.0002
    if env.magbot_type == 'sixD_platform':
        # the platform of the parked MagBot is level and its c-rotation equals the mover rotation in the station
        platform_pose = env.magbots.get_platform_pose(env.model, env.data, indices_magbots=np.array([0]))[0]
        a, b, c = vec_quat2euler(platform_pose[np.newaxis, 3:])[0]
        assert abs(a) < np.deg2rad(0.5) and abs(b) < np.deg2rad(0.5)
        assert abs(np.angle(np.exp(1j * (c - env.station_mover_poses[0, 0, 5])))) < np.deg2rad(0.5)


def test_magbot_kinematics_round_trip(env: ReconfigurationEnv, poses: dict[str, np.ndarray]) -> None:
    # the mover set poses in the station are reproduced by the inverse kinematics of the MagBot set pose (connection order)
    mover_poses = env.get_mover_poses_from_magbot_setpose(magbot_setposes=poses['magbot_station'])
    assert mover_poses.shape == (env.num_movers, 6)
    assert np.allclose(mover_poses[:, :3], poses['station'][:, :3], atol=1e-6)
    assert np.allclose(np.angle(np.exp(1j * (mover_poses[:, 5] - poses['station'][:, 5]))), 0.0, atol=1e-6)


@pytest.mark.parametrize(
    ('magbot_type', 'station_class', 'num_movers'), [('gripper', GripperStation, 3), ('sixD_platform', PlatformStation, 2)]
)
def test_single_magbot_type(magbot_type: str, station_class: type, num_movers: int) -> None:
    env = ReconfigurationEnv(magbot_types=(magbot_type,), station_poses=STATION_POSES, seed=0, render_mode=None)
    assert env.magbot_type == magbot_type
    assert all(type(station) is station_class for station in env.stations)
    assert env.num_movers == num_movers
    assert env.station_mover_poses.shape == (1, num_movers, 6)
    env.close()


def test_random_magbot_type() -> None:
    magbot_types = []
    for seed in range(10):
        env = ReconfigurationEnv(initial_mover_xy_pos=INITIAL_MOVER_XY_POS, station_poses=STATION_POSES, seed=seed, render_mode=None)
        magbot_types.append(env.magbot_type)
        # the initial mover positions of the chosen MagBot type are used
        assert np.allclose(env.initial_mover_xy_pos, INITIAL_MOVER_XY_POS[env.magbot_type])
        env.close()
    # both MagBot types are chosen and the same seed leads to the same MagBot type
    assert set(magbot_types) == set(ReconfigurationEnv.MAGBOT_TYPES)
    for seed in (2, 7):
        env = ReconfigurationEnv(station_poses=STATION_POSES, seed=seed, render_mode=None)
        assert env.magbot_type == magbot_types[seed]
        env.close()


def test_invalid_initial_mover_xy_pos() -> None:
    with pytest.raises(AssertionError):
        ReconfigurationEnv(
            magbot_types=('sixD_platform',),
            initial_mover_xy_pos=INITIAL_MOVER_XY_POS['gripper'],
            station_poses=STATION_POSES,
            render_mode=None,
        )

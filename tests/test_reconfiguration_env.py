import mujoco
import numpy as np
import pytest

from magbotsim.envs_without_rl_api.magbots.reconfiguration_env import ReconfigurationEnv

# Tests of the station collision proxies (hidden proxies of the rail mounts, rail rods and station pins) of the reconfiguration env:
# entering a station with the correct entry axis must not cause a collision, whereas entering diagonally must cause a collision.
# Entering correctly is always checked first, i.e. a collision on the diagonal path is not caused by an earlier misalignment.

NUM_CYCLES = 20
INITIAL_MOVER_XY_POS = np.array([[0.250, 0.150], [0.710, 0.150], [0.480, 0.150]])  # [m]
STATION_POSES = np.array([[0.480, 0.720, 0.01, np.deg2rad(-90)]])  # (x, y, z, yaw) of the m2 mount


@pytest.fixture
def env() -> ReconfigurationEnv:
    env = ReconfigurationEnv(initial_mover_xy_pos=INITIAL_MOVER_XY_POS, station_poses=STATION_POSES, render_mode=None)
    yield env
    env.close()


@pytest.fixture
def poses(env: ReconfigurationEnv) -> dict[str, np.ndarray]:
    """Mover set poses (numpy arrays of shape (num_movers, 6)) and gripper set poses (numpy arrays of shape (1, 7)) of the station
    choreography (see the example in reconfiguration_env.py). Connection idx_c is attached to mover idx_c.
    """
    station = env.stations[0]
    rot_station = station.rotation_matrix_xy()
    p = {'station': station.get_mover_poses(z_m=env.initial_mover_z_pos)}
    # entrance: the movers left the station along the entry axis (y-axis of the station body frame)
    p['entrance'] = p['station'].copy()
    p['entrance'][:, :2] += np.array([[0.0, -0.145], [0.0, 0.145], [0.0, 0.145]]) @ rot_station.T
    # pre-station: 200 mm in front of the entrance (x-axis of the station body frame)
    p['pre_station'] = p['entrance'].copy()
    p['pre_station'][:, :2] += np.array([0.2, 0.0]) @ rot_station.T
    p['initial'] = p['station'].copy()
    p['initial'][:, :2] = INITIAL_MOVER_XY_POS
    p['initial'][:, 5] = 0.0
    # gripper poses: MagBot in the station and taken out of the station by 0.25 m along the x-axis of the station body frame
    center = p['station'].copy()
    center[:, :2] += rot_station @ np.array([0.25, 0.0])
    p['gripper_station'] = env.magbots.moverSetPose2GripperSetPose(mover_poses=p['station'][np.newaxis])
    p['gripper_center'] = env.magbots.moverSetPose2GripperSetPose(mover_poses=center[np.newaxis])
    # MagBot shifted by 0.1 m along the y-axis of the station body frame, i.e. not aligned with the station pins
    p['gripper_shifted'] = p['gripper_center'].copy()
    p['gripper_shifted'][0, :2] += rot_station @ np.array([0.0, 0.1])
    return p


def move(env: ReconfigurationEnv, start: np.ndarray, end: np.ndarray, duration: float, gripper: bool = False, hold: float = 0.5) -> bool:
    """Move the movers (or the MagBot gripper) linearly from start to end and hold the end pose.

    :return: True if a station proxy collision occurred, False otherwise
    """
    dt = env.magbots.cycle_time * NUM_CYCLES
    setposes = [start + s * (end - start) for s in np.linspace(0.0, 1.0, int(duration / dt))] + [end] * int(hold / dt)
    for setpose in setposes:
        mover_setposes = env.get_mover_poses_from_gripper_setPose(gripper_setposes=setpose) if gripper else setpose
        env.step(mover_setposes=mover_setposes, num_cycles=NUM_CYCLES)
        # step() stops at the first collision, i.e. the colliding contacts are still present after step()
        if env.check_station_proxy_collision():
            return True
    return False


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
    assert sum('_rail_rod' in name for name in names) == 3
    assert sum('_pins_' in name for name in names) == 3 * 2
    for idx_c in range(3):  # each rail mount proxy has jaws (lower layer) and an upper layer
        assert any(f'rail_mount_{idx_c}_0_proxy_lower_' in name for name in names)
        assert any(f'rail_mount_{idx_c}_0_proxy_upper_' in name for name in names)
    assert not env.check_station_proxy_collision()


def test_empty_movers_diagonal_entry_collides(env: ReconfigurationEnv, poses: dict[str, np.ndarray]) -> None:
    # the empty movers drive in front of the station (the MagBot is parked in the station)
    assert not move(env, poses['initial'], poses['pre_station'], duration=1.5)
    assert (env.mover_station_indices == 0).all()
    # the empty movers enter the station diagonally instead of along the entry axis -> the rail rods hit the jaws of the rail mounts
    assert move(env, poses['pre_station'], poses['station'], duration=1.5)
    assert env.check_station_proxy_collision()
    assert any('rail_rod' in names and '_lower_' in names for names in proxy_contact_names(env))


def test_magbot_diagonal_entry_collides(env: ReconfigurationEnv, poses: dict[str, np.ndarray]) -> None:
    # the empty movers enter the station along the entry axis and take the MagBot out of the station
    assert not move(env, poses['initial'], poses['pre_station'], duration=1.5)
    assert not move(env, poses['pre_station'], poses['entrance'], duration=1.5)
    assert not move(env, poses['entrance'], poses['station'], duration=1.5)
    assert not move(env, poses['gripper_station'], poses['gripper_center'], duration=1.5, gripper=True)
    assert (env.mover_station_indices == -1).all()
    # the MagBot is shifted sideways (away from the station)
    assert not move(env, poses['gripper_center'], poses['gripper_shifted'], duration=1.0, gripper=True)
    # the MagBot is moved back into the station diagonally -> the rail mounts hit the station pins
    assert move(env, poses['gripper_shifted'], poses['gripper_station'], duration=1.5, gripper=True)
    assert env.check_station_proxy_collision()
    assert any('_pins_' in names and '_upper_' in names for names in proxy_contact_names(env))


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
    """Mover set poses and gripper set pose at the real (shifted) positions of the parked rail mounts (connection idx_c = mover idx_c)."""
    rot_station = env.stations[0].rotation_matrix_xy()
    p = {'station': poses['station'].copy()}
    p['station'][:, :2] = env.get_rail_mount_xy()
    p['entrance'] = p['station'].copy()
    p['entrance'][:, :2] += np.array([[0.0, -0.145], [0.0, 0.145], [0.0, 0.145]]) @ rot_station.T
    p['pre_station'] = p['entrance'].copy()
    p['pre_station'][:, :2] += np.array([0.2, 0.0]) @ rot_station.T
    p['gripper_station'] = env.magbots.moverSetPose2GripperSetPose(mover_poses=p['station'][np.newaxis])
    return p


def test_random_mover_start_positions() -> None:
    env = ReconfigurationEnv(station_poses=STATION_POSES, randomize_magbot_station_pos=True, seed=1, render_mode=None)
    for seed in range(50):
        env.reset_scene(rng=np.random.default_rng(seed))
        assert env.mover_start_xy_pos.shape == (env.num_movers, 2)
        assert not env.check_wall_collision(mover_names=env.mover_names, c_size=env.c_size).any()
        assert not env.check_mover_collision(mover_names=env.mover_names, c_size=env.c_size)
        assert not env.check_station_proxy_collision()
        assert not mover_contacts(env)
    env.close()

    # same seed -> same start positions and MagBot offsets
    env_1 = ReconfigurationEnv(station_poses=STATION_POSES, randomize_magbot_station_pos=True, seed=3, render_mode=None)
    env_2 = ReconfigurationEnv(station_poses=STATION_POSES, randomize_magbot_station_pos=True, seed=3, render_mode=None)
    assert np.allclose(env_1.initial_mover_xy_pos, env_2.initial_mover_xy_pos)
    assert np.allclose(env_1.magbot_station_x_offsets, env_2.magbot_station_x_offsets)
    assert np.abs(env_1.magbot_station_x_offsets).max() <= env_1.magbot_station_randomization_x_range
    env_1.close()
    env_2.close()


def test_explicit_start_positions_are_kept(env: ReconfigurationEnv) -> None:
    assert np.allclose(env.initial_mover_xy_pos, INITIAL_MOVER_XY_POS)
    assert np.allclose(env.get_mover_qpos(mover_names=env.mover_names)[:, :2], INITIAL_MOVER_XY_POS)
    assert np.allclose(env.magbot_station_x_offsets, 0.0)


@pytest.mark.parametrize('x_offset', [-0.005, 0.0023, 0.005])
def test_magbot_station_shift(env: ReconfigurationEnv, poses: dict[str, np.ndarray], x_offset: float) -> None:
    rail_mount_xy_ref = env.get_rail_mount_xy()  # unshifted MagBot
    env.reset_scene(mover_start_xy_pos=INITIAL_MOVER_XY_POS, magbot_station_x_offsets=np.array([x_offset]))
    # all rail mounts are shifted exactly by the offset along the x-axis of the station (no settling required)
    assert np.allclose(station_frame_offsets(env, rail_mount_xy_ref), [[x_offset, 0.0]] * 3, atol=1e-6)
    assert not env.check_station_proxy_collision()
    assert not mover_contacts(env)
    # the MagBot stays in the shifted pose while the movers are held
    rail_mount_xy_shifted = env.get_rail_mount_xy()
    assert not move(env, poses['initial'], poses['initial'], duration=0.0, hold=0.5)
    assert np.abs(env.get_rail_mount_xy() - rail_mount_xy_shifted).max() < 0.0002


def test_shifted_magbot_pickup_and_putback(env: ReconfigurationEnv, poses: dict[str, np.ndarray]) -> None:
    env.reset_scene(mover_start_xy_pos=INITIAL_MOVER_XY_POS, magbot_station_x_offsets=np.array([0.004]))
    pickup = pickup_poses(env, poses)
    # the empty movers enter the station at the real rail mount positions and take the MagBot out of the station
    assert not move(env, poses['initial'], pickup['pre_station'], duration=1.5)
    assert not move(env, pickup['pre_station'], pickup['entrance'], duration=1.5)
    assert not move(env, pickup['entrance'], pickup['station'], duration=1.5)
    assert (env.mover_station_indices == 0).all()
    assert not move(env, pickup['gripper_station'], poses['gripper_center'], duration=1.5, gripper=True)
    assert (env.mover_station_indices == -1).all()
    # the rail mounts were picked up without a jump, i.e. they stay on the mover adapters
    mover_xy = env.get_mover_qpos(mover_names=env.mover_names)[:, :2]
    assert np.linalg.norm(env.get_rail_mount_xy() - mover_xy, axis=1).max() < 0.0005

    # the MagBot is put back at the nominal station pose and stays there (no re-randomization)
    assert not move(env, poses['gripper_center'], poses['gripper_station'], duration=1.5, gripper=True)
    assert not move(env, poses['station'], poses['entrance'], duration=1.5)
    assert (env.mover_station_indices == 0).all()
    assert np.linalg.norm(env.get_rail_mount_xy() - poses['station'][:, :2], axis=1).max() < 0.0005


def test_putback_off_nominal_does_not_switch(env: ReconfigurationEnv, poses: dict[str, np.ndarray]) -> None:
    # the movers take the MagBot out of the station
    assert not move(env, poses['initial'], poses['pre_station'], duration=1.5)
    assert not move(env, poses['pre_station'], poses['entrance'], duration=1.5)
    assert not move(env, poses['entrance'], poses['station'], duration=1.5)
    assert not move(env, poses['gripper_station'], poses['gripper_center'], duration=1.5, gripper=True)
    assert (env.mover_station_indices == -1).all()
    # the MagBot is put back 3 mm (> station_switch_tol) in front of the nominal station pose
    rot_station = env.stations[0].rotation_matrix_xy()
    gripper_off = poses['gripper_station'].copy()
    gripper_off[0, :2] += rot_station @ np.array([0.003, 0.0])
    station_off = poses['station'].copy()
    station_off[:, :2] += rot_station @ np.array([0.003, 0.0])
    assert not move(env, poses['gripper_center'], gripper_off, duration=1.5, gripper=True)
    # the movers try to leave along the entry axis -> the rail mounts are not welded to the station
    move(env, station_off, station_off + (poses['entrance'] - poses['station']), duration=0.5, hold=0.0)
    assert (env.mover_station_indices == -1).all()

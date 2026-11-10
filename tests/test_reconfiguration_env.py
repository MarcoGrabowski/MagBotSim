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

    :return: True if a collision occurred, False otherwise
    """
    dt = env.magbots.cycle_time * NUM_CYCLES
    setposes = [start + s * (end - start) for s in np.linspace(0.0, 1.0, int(duration / dt))] + [end] * int(hold / dt)
    for setpose in setposes:
        mover_setposes = env.get_mover_poses_from_gripper_setPose(gripper_setposes=setpose) if gripper else setpose
        if env.step(mover_setposes=mover_setposes, single_track=np.array([False]), num_cycles=NUM_CYCLES):
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

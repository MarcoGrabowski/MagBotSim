import mujoco
import numpy as np
import pytest

from magbotsim import GripperMagBotsAPM4330, SixDPlatformMagBotsAPM4330
from magbotsim.envs_without_rl_api.magbots.reconfiguration_env import ReconfigurationEnv
from magbotsim.magbots.mover_coupling import wrap_angle
from magbotsim.utils.rotations_utils import vec_quat2euler

# Tests of the coupling between the movers and the connections (rail mounts) of the MagBots: any mover can pick up any rail mount of a
# MagBot in the reconfiguration env (more movers than connections), the rotation control of the movers is switched when a mover is
# attached or detached, and the static mode (movers specified at initialization) of the MagBot classes.

NUM_CYCLES = 20
STATION_POSES = np.array([[0.480, 0.720, 0.01, np.deg2rad(-90)]])  # (x, y, z, yaw) of the station anchor
NUM_MOVERS = {'gripper': 5, 'sixD_platform': 4}
# start positions of the movers that do not carry the MagBot (bottom of the table, far enough from the walls to rotate in place)
PARKING_XY = np.array([[0.14, 0.13], [0.83, 0.13]])  # [m]
# distance between the pre-station points and the staging points (towards the station anchor), at which the movers rotate in place [m]
STAGING_DIST = 0.15


###################################################
# Helpers                                         #
###################################################


class Scene:
    """Reconfiguration env with extra movers: the movers that carry the MagBot (``connection_movers``, connection order) start at the
    staging points (yaw 0), all other movers at the parking positions. Provides the mover set poses of the station choreography (see the
    example in reconfiguration_env.py) in connection order.
    """

    def __init__(self, magbot_type: str, connection_movers: list[int]) -> None:
        self.env = ReconfigurationEnv(
            magbot_types=(magbot_type,), num_movers=NUM_MOVERS[magbot_type], station_poses=STATION_POSES, seed=0, render_mode=None
        )
        env = self.env
        self.connection_movers = np.array(connection_movers)
        station = env.stations[0]
        rot_station = station.rotation_matrix_xy()
        p = {'station': env.station_mover_poses[0].copy()}
        p['entrance'] = p['station'].copy()
        p['entrance'][:, :2] += np.stack([np.zeros(env.num_connections), 0.145 * station.MOVER_ENTRY_SIGNS], axis=1) @ rot_station.T
        p['pre_station'] = p['entrance'].copy()
        p['pre_station'][:, 1] -= 0.2
        p['staging'] = p['pre_station'].copy()
        p['staging'][:, 0] += STAGING_DIST * np.sign(station.pos[0] - p['staging'][:, 0])
        p['staging_yaw0'] = p['staging'].copy()
        p['staging_yaw0'][:, 5] = 0.0
        center = p['station'].copy()
        center[:, :2] += rot_station @ np.array([0.25, 0.0])
        p['magbot_station'] = env.get_magbot_setpose_from_mover_poses(mover_poses=p['station'])
        p['magbot_center'] = env.get_magbot_setpose_from_mover_poses(mover_poses=center)
        self.poses = p

        # start positions
        other_movers = [idx_mover for idx_mover in range(env.num_movers) if idx_mover not in connection_movers]
        start_xy = np.zeros((env.num_movers, 2))
        start_xy[other_movers] = PARKING_XY[: len(other_movers)]
        start_xy[self.connection_movers] = p['staging'][:, :2]
        env.reset_scene(mover_start_xy_pos=start_xy)
        self.setposes = np.zeros((env.num_movers, 6))
        self.setposes[:, :2] = start_xy
        self.setposes[:, 2] = env.initial_mover_z_pos

    def close(self) -> None:
        """Close the env."""
        self.env.close()

    def move(
        self, start: np.ndarray, end: np.ndarray, duration: float = 1.25, magbot: bool = False, hold: float = 0.25, check: bool = True
    ) -> list[dict[str, bool]]:
        """Move the movers that carry the MagBot (connection order) or the MagBot linearly from start to end and hold the end pose. All
        other movers hold their set poses. If ``check`` is True, no collision must occur.

        :return: the info dicts of all steps
        """
        dt = self.env.cycle_time * NUM_CYCLES
        infos = []
        for pose in [start + s * (end - start) for s in np.linspace(0.0, 1.0, int(duration / dt))] + [end] * int(hold / dt):
            poses_c = self.env.get_mover_poses_from_magbot_setpose(magbot_setposes=pose) if magbot else pose
            self.setposes[self.connection_movers] = poses_c
            infos.append(self.env.step(mover_setposes=self.setposes.copy(), num_cycles=NUM_CYCLES))
            if check:
                assert not any(infos[-1].values()), infos[-1]
        return infos

    def pick_up(self) -> None:
        """The movers rotate to the docking yaws at the staging points, enter the station and take the MagBot out of the station."""
        p = self.poses
        self.move(p['staging_yaw0'], p['staging'], duration=1.0)
        self.move(p['staging'], p['pre_station'], duration=0.75)
        self.move(p['pre_station'], p['entrance'])
        self.move(p['entrance'], p['station'])
        self.move(p['magbot_station'], p['magbot_center'], magbot=True)

    def put_back(self) -> None:
        """The movers put the MagBot back into the station and leave the station."""
        p = self.poses
        self.move(p['magbot_center'], p['magbot_station'], magbot=True)
        self.move(p['station'], p['entrance'])

    def return_to_staging(self) -> None:
        """The empty movers drive back from the station entrance to the staging points and rotate back to a yaw of 0."""
        p = self.poses
        self.move(p['entrance'], p['pre_station'])
        self.move(p['pre_station'], p['staging'], duration=0.75)
        self.move(p['staging'], p['staging_yaw0'], duration=1.0)

    def place_free_movers(self, mover_indices: np.ndarray, xy: np.ndarray) -> None:
        """Place free movers (not attached to any connection) at the given positions with a yaw of 0 (and hold them there)."""
        env = self.env
        assert (env.mover_magbot_indices[mover_indices] < 0).all()
        for idx_mover, (x, y) in zip(mover_indices, xy, strict=True):
            env.data.qpos[env.mover_qpos_indices[idx_mover]] = [x, y, env.initial_mover_z_pos + env.mover_size[idx_mover, 2], 1, 0, 0, 0]
            env.data.qvel[env.mover_qvel_qacc_indices[idx_mover]] = 0.0
            self.setposes[idx_mover] = [x, y, env.initial_mover_z_pos, 0.0, 0.0, 0.0]
        mujoco.mj_forward(env.model, env.data)
        env.last_mover_pos_d = None

    def mover_yaws(self) -> np.ndarray:
        """Return the current (wrapped) yaws of all movers."""
        return vec_quat2euler(self.env.get_mover_qpos(mover_names=self.env.mover_names)[:, 3:])[:, 2]


def assert_parked(env: ReconfigurationEnv) -> None:
    """All rail mounts are parked at the nominal station positions, no mover is attached and all mover welds are inactive."""
    assert (env.connection_station_indices == 0).all()
    assert (env.connection_mover_indices == -1).all()
    assert (env.mover_magbot_indices == -1).all() and (env.mover_connection_indices == -1).all()
    assert not env.data.eq_active[env.magbots.eq_ids_mover_weld].any()
    assert np.linalg.norm(env.get_rail_mount_xy() - env.station_mover_poses[0, :, :2], axis=1).max() < 0.0005


def assert_carried(scene: Scene) -> None:
    """The rail mounts are carried by the movers of the scene (only their welds are active) and follow the movers."""
    env, connection_movers = scene.env, scene.connection_movers
    assert (env.connection_station_indices == -1).all()
    assert np.array_equal(env.connection_mover_indices[0], connection_movers)
    assert np.array_equal(env.mover_connection_indices[connection_movers], np.arange(env.num_connections))
    assert (env.mover_magbot_indices[connection_movers] == 0).all()
    weld_active = env.data.eq_active[env.magbots.eq_ids_mover_weld].astype(bool)
    assert weld_active.sum() == env.num_connections
    assert weld_active[connection_movers, 0, np.arange(env.num_connections)].all()
    mover_xy = env.get_mover_qpos(mover_names=env.mover_names)[connection_movers, :2]
    assert np.linalg.norm(env.get_rail_mount_xy() - mover_xy, axis=1).max() < 0.0005


def gripper_joints(env: ReconfigurationEnv) -> np.ndarray:
    """Return the current gripper a-, b- and finger joint positions of the Gripper MagBot."""
    magbots = env.magbots
    assert isinstance(magbots, GripperMagBotsAPM4330)
    indices = [magbots.gripper_qpos_a_indices[0], magbots.gripper_qpos_b_indices[0], magbots.gripper_qpos_f_indices[0]]
    return env.data.qpos[indices].copy()


###################################################
# Dynamic coupling (reconfiguration env)          #
###################################################


@pytest.mark.parametrize(
    ('magbot_type', 'connection_movers'),
    [('gripper', [4, 1, 3]), ('gripper', [0, 2, 1]), ('sixD_platform', [3, 0]), ('sixD_platform', [1, 2])],
)
def test_any_mover_carries_any_connection(magbot_type: str, connection_movers: list[int]) -> None:
    scene = Scene(magbot_type=magbot_type, connection_movers=connection_movers)
    env = scene.env
    assert env.num_movers == NUM_MOVERS[magbot_type] > env.num_connections
    assert_parked(env)

    scene.pick_up()
    assert_carried(scene)
    # the docked movers have the docking yaws (Gripper: connection 0 is rotated by pi, 6D-Platform: connection 1 is rotated by pi)
    yaw_errors = wrap_angle(scene.mover_yaws()[scene.connection_movers] - env.station_mover_poses[0, :, 5])
    assert np.abs(yaw_errors).max() < np.deg2rad(1.0)

    scene.put_back()
    assert_parked(env)
    scene.close()


@pytest.mark.parametrize(('magbot_type', 'movers_1', 'movers_2'), [('gripper', [4, 1, 3], [1, 3, 4]), ('sixD_platform', [3, 0], [0, 3])])
def test_movers_swap_connections(magbot_type: str, movers_1: list[int], movers_2: list[int]) -> None:
    """The same movers carry different connections in two consecutive cycles (without resetting the scene)."""
    scene = Scene(magbot_type=magbot_type, connection_movers=movers_1)
    env = scene.env
    scene.pick_up()
    assert_carried(scene)
    scene.put_back()
    scene.return_to_staging()
    assert_parked(env)

    # the movers swap their staging points and pick up the MagBot again
    scene.place_free_movers(np.array(movers_2), scene.poses['staging'][:, :2])
    scene.connection_movers = np.array(movers_2)
    scene.pick_up()
    assert_carried(scene)
    if magbot_type == 'gripper':
        # the gripper is still in its default configuration (a=b=0, closed fingers)
        assert np.abs(gripper_joints(env)).max() < 0.01
    scene.put_back()
    assert_parked(env)
    scene.close()


def test_gripper_rotation_control_switching() -> None:
    scene = Scene(magbot_type='gripper', connection_movers=[4, 1, 3])
    env = scene.env
    magbots = env.magbots
    assert isinstance(magbots, GripperMagBotsAPM4330)
    free_movers = np.array([0, 2])

    def c_masks() -> np.ndarray:
        return np.array([controller.joint_mask[5] for controller in env.impedance_controllers])

    def magbot_c_ctrls() -> np.ndarray:
        return env.data.ctrl[magbots.mover_actuator_ids]

    # all movers are free: the c-rotations are controlled by the impedance controllers, not by the MagBot
    assert c_masks().all()
    scene.move(scene.poses['staging_yaw0'], scene.poses['staging_yaw0'], duration=0.0, hold=0.2)
    assert np.allclose(magbot_c_ctrls(), 0.0)

    # free movers rotate (impedance controllers) without moving the parked gripper
    joints_parked = gripper_joints(env)
    scene.setposes[free_movers, 5] = np.pi / 2
    scene.move(scene.poses['staging_yaw0'], scene.poses['staging_yaw0'], duration=0.0, hold=1.5)
    assert np.abs(wrap_angle(scene.mover_yaws()[free_movers] - np.pi / 2)).max() < np.deg2rad(1.0)
    assert np.abs(gripper_joints(env) - joints_parked).max() < 1e-4
    assert np.allclose(magbot_c_ctrls(), 0.0)

    # the attached movers are controlled by the MagBot, the free movers by the impedance controllers
    scene.pick_up()
    assert_carried(scene)
    assert np.array_equal(c_masks(), [True, False, True, False, False])

    # the gripper follows the gripper set pose (rotations and finger)
    magbot_pose = scene.poses['magbot_center'].copy()
    magbot_pose[0, 3:6] += np.deg2rad([-10.0, 15.0, 5.0])
    magbot_pose[0, 6] = 0.01
    scene.move(scene.poses['magbot_center'], magbot_pose, magbot=True, hold=1.0)
    a_d, b_d = magbot_pose[0, 3], magbot_pose[0, 4]
    f_d = float(magbots.finger_distance_to_angle(np.array([magbot_pose[0, 6]]))[0])
    abf_ctrl = env.data.ctrl[magbots.gripper_abf_actuator_ids[0]]
    assert np.allclose(abf_ctrl, [a_d, b_d, f_d], atol=np.deg2rad(1.0))
    scene.move(magbot_pose, scene.poses['magbot_center'], magbot=True, hold=1.0)

    # after the put-back, all movers are controlled by the impedance controllers again
    scene.put_back()
    assert_parked(env)
    assert c_masks().all()
    assert np.allclose(magbot_c_ctrls(), 0.0)
    assert np.abs(gripper_joints(env)).max() < 0.01
    scene.close()


@pytest.mark.parametrize('magbot_type', ReconfigurationEnv.MAGBOT_TYPES)
@pytest.mark.parametrize('yaw_error_deg', [5.0, -5.0, 180.0])
def test_wrongly_rotated_movers_collide(magbot_type: str, yaw_error_deg: float) -> None:
    """Movers whose adapters are not aligned with the rail mounts do not pick up the MagBot: a small yaw error causes a collision of the
    rail rods with the jaws of the rail mounts when entering the station. A mover rotated by pi fits through the jaws, but is not attached
    (orientation check), i.e. its rail rod hits the jaws when the mover leaves the station along the take-out axis.
    """
    scene = Scene(magbot_type=magbot_type, connection_movers=[0, 1] if magbot_type == 'sixD_platform' else [0, 1, 2])
    env = scene.env
    for key in ('station', 'entrance', 'pre_station', 'staging'):
        scene.poses[key][:, 5] += np.deg2rad(yaw_error_deg)
    scene.move(scene.poses['staging_yaw0'], scene.poses['staging'], duration=1.0)
    scene.move(scene.poses['staging'], scene.poses['pre_station'], duration=0.75)
    scene.move(scene.poses['pre_station'], scene.poses['entrance'])
    infos = scene.move(scene.poses['entrance'], scene.poses['station'], check=False)
    entering_collision = any(info['rail_rod_collision'] for info in infos)
    assert entering_collision == (abs(yaw_error_deg) < 90.0)
    if not entering_collision:
        # take-out along the x-axis of the station (the movers leave the parked rail mounts)
        taken_out = scene.poses['station'].copy()
        taken_out[:, :2] += env.stations[0].rotation_matrix_xy() @ np.array([0.05, 0.0])
        infos = scene.move(scene.poses['station'], taken_out, duration=0.5, check=False)
        # the first collision is caused by the rail rods (not by rail mounts welded to the movers that hit the station pins)
        first_collision = next(info for info in infos if info['station_proxy_collision'])
        assert first_collision['rail_rod_collision'] and not first_collision['station_pin_collision']
    assert (env.connection_station_indices == 0).all() and (env.connection_mover_indices == -1).all()
    scene.close()


def test_gripper_yaw_branches() -> None:
    """A mover that rotated by a full turn while it was free and whose set yaw is shifted by 2*pi is attached without spinning the mover
    and without moving the gripper (the continuous mover yaw and the set yaw are aligned with the kinematic connection yaw).
    """
    scene = Scene(magbot_type='gripper', connection_movers=[4, 1, 3])
    env = scene.env
    magbots = env.magbots
    assert isinstance(magbots, GripperMagBotsAPM4330)
    p = scene.poses
    # the mover of connection 1 rotates a full turn at its staging point (free mover, impedance controller)
    full_turn = p['staging_yaw0'].copy()
    full_turn[1, 5] = 2 * np.pi
    scene.move(p['staging_yaw0'], full_turn, duration=3.0, hold=0.5)
    for key in ('station', 'entrance', 'pre_station', 'staging'):
        p[key][1, 5] += 2 * np.pi
    scene.move(full_turn, p['staging'], duration=1.0)
    scene.move(p['staging'], p['pre_station'], duration=0.75)
    scene.move(p['pre_station'], p['entrance'])
    scene.move(p['entrance'], p['station'])
    yaw_start = magbots.mover_yaw_extractor.yaws[1]
    assert abs(yaw_start - 2 * np.pi) < np.deg2rad(2.0)

    # take-out with set yaws shifted by 2*pi for the mover of connection 1
    dt = env.cycle_time * NUM_CYCLES
    for s in np.linspace(0.0, 1.0, int(1.25 / dt)):
        magbot_pose = p['magbot_station'] + s * (p['magbot_center'] - p['magbot_station'])
        poses_c = env.get_mover_poses_from_magbot_setpose(magbot_setposes=magbot_pose)
        poses_c[1, 5] += 2 * np.pi
        scene.setposes[scene.connection_movers] = poses_c
        assert not any(env.step(mover_setposes=scene.setposes.copy(), num_cycles=NUM_CYCLES).values())
        assert np.abs(gripper_joints(env)).max() < 0.01
    assert_carried(scene)
    # the mover did not spin and the kinematic connection yaws are equal to the gripper c-rotation
    assert abs(magbots.mover_yaw_extractor.yaws[1] - yaw_start) < np.deg2rad(2.0)
    assert magbots.connection_yaws is not None
    assert np.abs(wrap_angle(magbots.connection_yaws[0] - p['magbot_center'][0, 5])).max() < np.deg2rad(1.0)
    assert np.abs(magbots.connection_yaws[0] - magbots.connection_yaws[0, 0]).max() < np.deg2rad(1.0)
    scene.close()


###################################################
# Static and dynamic mode (MagBot classes)        #
###################################################


def test_static_mode_gripper() -> None:
    magbots = GripperMagBotsAPM4330(
        num_magbots=2,
        num_movers=6,
        indices_mover_c0=np.array([0, 3]),
        indices_mover_c1=np.array([1, 4]),
        indices_mover_c2=np.array([2, 5]),
    )
    assert not magbots.dynamic_mover_coupling
    assert magbots.num_movers == 6
    assert np.array_equal(magbots.connection_mover_indices, [[0, 1, 2], [3, 4, 5]])
    # the movers have the adapter orientation of their connection, i.e. no yaw offsets (same kinematics as the MagBot)
    assert np.allclose(magbots.mover_adapter_yaws, [-np.pi / 2, np.pi / 2, np.pi / 2] * 2)
    for idx_mover in range(6):
        assert magbots.get_mover_yaw_offset(idx_mover=idx_mover, idx_c=idx_mover % 3) == pytest.approx(0.0)
    # welds between all movers and rail mounts, only the welds of the specified movers are active
    xml_strings = magbots.generate_magbot_xml_strings(
        initial_pos_xyz_mover_c0=np.zeros((2, 3)), custom_model_xml_strings={'custom_mover_body_xml_str_list': [None] * 6}
    )
    equality_xml = xml_strings['custom_outworldbody_xml_str']
    assert equality_xml.count('<weld name="mover_weld_') == 6 * 6
    for idx_mover in range(6):
        weld = f'<weld name="{magbots.mover_weld_name(idx_mover=idx_mover, idx_c=idx_mover % 3, idx_magbot=idx_mover // 3)}"'
        assert 'active="true"' in equality_xml.split(weld)[1].split('/>')[0]
    assert equality_xml.count('active="true"') == 6


def test_static_mode_unused_movers() -> None:
    """In static mode, only the specified movers get an adapter; the other movers of the environment are not attached."""
    magbots = GripperMagBotsAPM4330(
        num_magbots=1, num_movers=7, indices_mover_c0=np.array([2]), indices_mover_c1=np.array([3]), indices_mover_c2=np.array([4])
    )
    assert not magbots.dynamic_mover_coupling
    assert magbots.num_movers == 7
    assert np.array_equal(magbots.adapter_mover_indices, [2, 3, 4])
    assert magbots.get_mover_attachments() == [None, None, (0, 0), (0, 1), (0, 2), None, None]


def test_static_mode_platform() -> None:
    magbots = SixDPlatformMagBotsAPM4330(num_magbots=2, num_movers=4, indices_mover_a=np.array([0, 2]), indices_mover_b=np.array([1, 3]))
    assert not magbots.dynamic_mover_coupling
    # connection order [b, a]
    assert np.array_equal(magbots.connection_mover_indices, [[1, 0], [3, 2]])
    assert np.allclose(magbots.mover_adapter_yaws, [-np.pi / 2, np.pi / 2] * 2)
    for idx_magbot in range(2):
        for idx_c in range(2):
            idx_mover = magbots.connection_mover_indices[idx_magbot, idx_c]
            assert magbots.get_mover_yaw_offset(idx_mover=idx_mover, idx_c=idx_c) == pytest.approx(0.0)


@pytest.mark.parametrize('magbot_class', [GripperMagBotsAPM4330, SixDPlatformMagBotsAPM4330])
def test_dynamic_mode(magbot_class: type) -> None:
    magbots = magbot_class(num_magbots=1, num_movers=5)
    assert magbots.dynamic_mover_coupling
    assert (magbots.connection_mover_indices == -1).all()
    assert np.allclose(magbots.mover_adapter_yaws, np.pi / 2)
    # all movers carry the standard adapter, i.e. the yaw offsets only depend on the connection
    assert np.allclose(np.abs(magbots.pair_yaw_offsets), magbot_class.MOVER_YAW_OFFSETS[np.newaxis, :])
    assert np.allclose(magbot_class.mover_yaw_offsets(), magbot_class.MOVER_YAW_OFFSETS)
    assert magbots.get_mover_attachments() == [None] * 5
    assert not magbots.connections_carried.any()


def test_invalid_mode_arguments() -> None:
    indices = np.array([0])
    with pytest.raises(AssertionError):
        GripperMagBotsAPM4330(num_magbots=1, num_movers=3, indices_mover_c0=indices, indices_mover_c1=indices + 1)
    with pytest.raises(AssertionError):
        SixDPlatformMagBotsAPM4330(num_magbots=1, num_movers=2, indices_mover_a=indices)
    with pytest.raises(AssertionError):  # one mover for two connections
        SixDPlatformMagBotsAPM4330(num_magbots=1, num_movers=2, indices_mover_a=indices, indices_mover_b=indices)


def test_reconfiguration_env_num_movers() -> None:
    with pytest.raises(AssertionError):
        ReconfigurationEnv(magbot_types=('gripper',), num_movers=2, station_poses=STATION_POSES, render_mode=None)
    env = ReconfigurationEnv(magbot_types=('sixD_platform',), num_movers=6, station_poses=STATION_POSES, seed=1, render_mode=None)
    assert env.num_movers == 6 and env.num_connections == 2
    assert env.mover_start_xy_pos.shape == (6, 2)
    assert not env.check_mover_collision(mover_names=env.mover_names, c_size=env.c_size)
    env.close()


@pytest.fixture
def headless_example_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The example envs of the static mode do not expose the render mode, i.e. rendering is disabled for all BasicMagBotEnvs."""
    from magbotsim import BasicMagBotEnv

    init = BasicMagBotEnv.__init__

    def headless_init(self, *args, **kwargs) -> None:
        kwargs['render_mode'] = None
        init(self, *args, **kwargs)

    monkeypatch.setattr(BasicMagBotEnv, '__init__', headless_init)


@pytest.mark.usefixtures('headless_example_envs')
def test_static_mode_gripper_example_env_follows_setpose() -> None:
    from magbotsim.envs_without_rl_api.magbots.gripper_example_env import GripperMagBotExampleEnv

    env = GripperMagBotExampleEnv()
    p = env.magbot_params
    gripper_setposes = np.array(
        [
            [0.48, 0.36, p.z_p_min + p.n + 0.03, np.deg2rad(-10.0), np.deg2rad(15.0), np.deg2rad(5.0), 0.01],
            [1.44, 0.36, p.z_p_min + p.n + 0.03, np.deg2rad(-5.0), np.deg2rad(-10.0), np.deg2rad(-5.0), 0.005],
        ]
    )
    for _ in range(0, 100):
        env.step(gripper_setposes=gripper_setposes, single_track=np.array([False, True]), num_cycles=NUM_CYCLES)
    # the gripper rotations and finger angles (derived from the mover rotations) reach the set pose
    abf_d = np.stack([gripper_setposes[:, 3], gripper_setposes[:, 4], env.magbots.finger_distance_to_angle(gripper_setposes[:, 6])], axis=1)
    assert np.allclose(env.data.ctrl[env.magbots.gripper_abf_actuator_ids], abf_d, atol=np.deg2rad(1.0))
    gripper_pose = env.magbots.get_gripper_pose(env.model, env.data, indices_magbots=np.arange(2))
    assert np.linalg.norm(gripper_pose[:, :2] - gripper_setposes[:, :2], axis=1).max() < 0.01
    env.close()


@pytest.mark.usefixtures('headless_example_envs')
def test_static_mode_platform_example_env_follows_setpose() -> None:
    from magbotsim.envs_without_rl_api.magbots.sixD_platform_example_env import SixDPlatformMagBotExampleEnv

    env = SixDPlatformMagBotExampleEnv()
    platform_setposes = np.array(
        [[0.56, 0.40, 0.24, np.deg2rad(5.0), np.deg2rad(-4.0), np.deg2rad(10.0)], [1.30, 0.38, 0.24, np.deg2rad(-3.0), 0.0, 0.0]]
    )
    for _ in range(0, 100):
        env.step(platform_setposes=platform_setposes.copy(), num_cycles=NUM_CYCLES)
    platform_pose = env.magbots.get_platform_pose(env.model, env.data, indices_magbots=np.arange(2))
    assert np.linalg.norm(platform_pose[:, :3] - platform_setposes[:, :3], axis=1).max() < 0.01
    assert np.abs(wrap_angle(vec_quat2euler(platform_pose[:, 3:]) - platform_setposes[:, 3:])).max() < np.deg2rad(1.0)
    env.close()

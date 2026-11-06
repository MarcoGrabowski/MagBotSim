import numpy as np
import pytest
from magbotsim import GripperMagBotsAPM4330, GripperMagBotsAPM4330Params, SixDPlatformMagBotsAPM4330

NUM_SAMPLES = 50


########################################################################################################################
# Gripper MagBot
########################################################################################################################


@pytest.fixture
def gripper_magbots():
    magbot_params = GripperMagBotsAPM4330Params()
    magbots = GripperMagBotsAPM4330(
        num_magbots=1,
        indices_mover_c0=np.array([0]),
        indices_mover_c1=np.array([1]),
        indices_mover_c2=np.array([2]),
        magbot_params=magbot_params,
        cycle_time=0.001,
    )
    # synthetic, monotonic finger calibration (usually built from the MuJoCo model in update_cached_mujoco_data);
    # the angle decreases with increasing distance to also test the sorting in _set_finger_calibration
    magbots._set_finger_calibration(finger_distances=np.linspace(0.0, 0.02, 25), finger_angles=np.linspace(0.8, 0.0, 25))
    return magbots


def _sample_gripper_poses(magbot_params: GripperMagBotsAPM4330Params, z_m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample valid gripper poses, i.e. poses that are not clipped by the inverse kinematics. Shape: (NUM_SAMPLES,7)."""
    poses = np.zeros((NUM_SAMPLES, 7))
    poses[:, 0] = rng.uniform(0.3, 1.0, NUM_SAMPLES)  # x
    poses[:, 1] = rng.uniform(0.3, 1.0, NUM_SAMPLES)  # y
    poses[:, 3] = rng.uniform(-np.pi + 0.01, -0.01, NUM_SAMPLES)  # a
    poses[:, 4] = rng.uniform(-np.pi / 2, np.pi / 2, NUM_SAMPLES)  # b
    poses[:, 5] = rng.uniform(-np.pi + 0.01, np.pi - 0.01, NUM_SAMPLES)  # c
    poses[:, 6] = rng.uniform(0.0, magbot_params.f_d_max, NUM_SAMPLES)  # finger distance
    # z: choose the platform height within its limits, such that the gripper z-position is not less than z_d_min
    gripper_offset = z_m + magbot_params.n * np.cos(poses[:, 3])
    z_p_lo = np.maximum(magbot_params.z_p_min, magbot_params.z_d_min - gripper_offset) + 1e-4
    z_p = rng.uniform(z_p_lo, magbot_params.z_p_max - 1e-4)
    poses[:, 2] = z_p + gripper_offset
    return poses


@pytest.mark.parametrize('z_m_is_float', [True, False])
@pytest.mark.parametrize('single_track', [False, True])
def test_gripper_roundtrip_ik_fk(gripper_magbots, z_m_is_float, single_track):
    """Verify that gripper pose -> mover poses (IK) -> gripper pose (FK) returns the original gripper pose."""
    rng = np.random.default_rng(0)
    z_m = 0.001 if z_m_is_float else rng.uniform(0.0005, 0.003, NUM_SAMPLES)
    gripper_poses_d = _sample_gripper_poses(gripper_magbots.magbot_params, np.asarray(z_m), rng)

    mover_poses = gripper_magbots.gripperSetPose2MoverSetPose(gripper_poses_d.copy(), z_m=z_m, single_track=single_track)
    gripper_poses = gripper_magbots.moverSetPose2GripperSetPose(mover_poses)

    assert mover_poses.shape == (NUM_SAMPLES, 3, 6)
    assert gripper_poses.shape == (NUM_SAMPLES, 7)
    assert np.allclose(gripper_poses, gripper_poses_d, atol=1e-9)


@pytest.mark.parametrize('single_track', [False, True])
def test_gripper_roundtrip_fk_ik(gripper_magbots, single_track):
    """Verify that mover poses -> gripper pose (FK) -> mover poses (IK) returns the original mover poses (including the x,y-position
    of the mover attached to connection 2, which is not used by the FK).
    """
    rng = np.random.default_rng(1)
    z_m = 0.002
    mover_poses_d = gripper_magbots.gripperSetPose2MoverSetPose(
        _sample_gripper_poses(gripper_magbots.magbot_params, np.asarray(z_m), rng), z_m=z_m, single_track=single_track
    )

    gripper_poses = gripper_magbots.moverSetPose2GripperSetPose(mover_poses_d)
    mover_poses = gripper_magbots.gripperSetPose2MoverSetPose(gripper_poses, z_m=z_m, single_track=single_track)

    assert np.allclose(mover_poses, mover_poses_d, atol=1e-9)


def test_gripper_fk_single_sample(gripper_magbots):
    """Verify the FK for a single MagBot. Shape: (1,3,6)."""
    z_m, a = 0.001, -0.3
    z = 0.19 + z_m + gripper_magbots.magbot_params.n * np.cos(a)  # platform height 0.19 m
    gripper_pose_d = np.array([[0.48, 0.48, z, a, np.deg2rad(-45.0), 0.2, 0.01]])
    mover_poses = gripper_magbots.gripperSetPose2MoverSetPose(gripper_pose_d.copy(), z_m=z_m)

    gripper_pose = gripper_magbots.moverSetPose2GripperSetPose(mover_poses)

    assert gripper_pose.shape == (1, 7)
    assert np.allclose(gripper_pose, gripper_pose_d)


def test_gripper_finger_roundtrip(gripper_magbots):
    finger_distances = np.linspace(0.0, 0.02, NUM_SAMPLES)
    finger_angles = gripper_magbots.finger_distance_to_angle(finger_distances)
    assert np.allclose(gripper_magbots.finger_angle_to_distance(finger_angles), finger_distances)


def test_gripper_finger_angle_to_distance_without_calibration():
    magbots = GripperMagBotsAPM4330(
        num_magbots=1,
        indices_mover_c0=np.array([0]),
        indices_mover_c1=np.array([1]),
        indices_mover_c2=np.array([2]),
        magbot_params=GripperMagBotsAPM4330Params(),
        cycle_time=0.001,
    )
    with pytest.raises(RuntimeError):
        magbots.finger_angle_to_distance(np.zeros((1,)))


########################################################################################################################
# 6D-Platform MagBot
########################################################################################################################


@pytest.fixture
def platform_magbots():
    return SixDPlatformMagBotsAPM4330(num_magbots=1, indices_mover_a=np.array([0]), indices_mover_b=np.array([1]))


def _sample_platform_poses(magbots: SixDPlatformMagBotsAPM4330, z_m: float, rng: np.random.Generator) -> np.ndarray:
    """Sample valid platform poses, i.e. poses that are not clipped by the inverse kinematics. Shape: (NUM_SAMPLES,6)."""
    poses = np.zeros((NUM_SAMPLES, 6))
    poses[:, 0] = rng.uniform(0.3, 1.0, NUM_SAMPLES)  # x
    poses[:, 1] = rng.uniform(0.3, 1.0, NUM_SAMPLES)  # y
    poses[:, 2] = z_m + rng.uniform(0.21, 0.27, NUM_SAMPLES)  # z (reachable range: approx. z_m + [0.204, 0.278])
    poses[:, 3] = rng.uniform(-magbots.max_platform_rot_ab, magbots.max_platform_rot_ab, NUM_SAMPLES)  # a
    poses[:, 4] = rng.uniform(-magbots.max_platform_rot_ab, magbots.max_platform_rot_ab, NUM_SAMPLES)  # b
    poses[:, 5] = rng.uniform(-np.pi + 0.01, np.pi - 0.01, NUM_SAMPLES)  # c
    return poses


@pytest.mark.parametrize('use_euler', [True, False])
def test_platform_roundtrip_ik_fk(platform_magbots, use_euler):
    """Verify that platform pose -> mover poses (IK) -> platform pose (FK) returns the original platform pose."""
    rng = np.random.default_rng(2)
    z_m = 0.001
    platform_poses_d = _sample_platform_poses(platform_magbots, z_m, rng)

    # copy, since platformSetPose2MoverSetPose modifies its input
    mover_poses = platform_magbots.platformSetPose2MoverSetPose(platform_poses_d.copy(), mover_z_d=z_m, use_euler=use_euler)
    platform_poses = platform_magbots.moverSetPose2PlatformSetPose(mover_poses, use_euler=use_euler)

    assert mover_poses.shape == (NUM_SAMPLES, 2, 6 if use_euler else 7)
    assert platform_poses.shape == (NUM_SAMPLES, 6)
    assert np.allclose(platform_poses, platform_poses_d, atol=1e-9)


@pytest.mark.parametrize('use_euler', [True, False])
def test_platform_roundtrip_fk_ik(platform_magbots, use_euler):
    """Verify that mover poses -> platform pose (FK) -> mover poses (IK) returns the original mover poses."""
    rng = np.random.default_rng(3)
    z_m = 0.002
    platform_poses_d = _sample_platform_poses(platform_magbots, z_m, rng)
    mover_poses_d = platform_magbots.platformSetPose2MoverSetPose(platform_poses_d.copy(), mover_z_d=z_m, use_euler=use_euler)

    platform_poses = platform_magbots.moverSetPose2PlatformSetPose(mover_poses_d, use_euler=use_euler)
    mover_poses = platform_magbots.platformSetPose2MoverSetPose(platform_poses, mover_z_d=z_m, use_euler=use_euler)

    if use_euler:
        assert np.allclose(mover_poses, mover_poses_d, atol=1e-9)
    else:
        assert np.allclose(mover_poses[:, :, :3], mover_poses_d[:, :, :3], atol=1e-9)
        # q and -q represent the same rotation
        quat_dot = np.abs(np.sum(mover_poses[:, :, 3:] * mover_poses_d[:, :, 3:], axis=2))
        assert np.allclose(quat_dot, 1.0, atol=1e-9)


def test_platform_fk_clips_ab_rotation(platform_magbots):
    """Verify that the FK clips the platform's a- and b-rotations to the maximum platform rotation."""
    mover_poses = platform_magbots.platformSetPose2MoverSetPose(
        np.array([[0.5, 0.5, 0.25, 0.0, 0.0, 0.0]]), mover_z_d=0.001, use_euler=True
    )
    mover_poses[:, 0, 5] = -2.5  # exceeds the maximum a-rotation
    mover_poses[:, 1, 5] = 2.5  # exceeds the maximum b-rotation

    platform_pose = platform_magbots.moverSetPose2PlatformSetPose(mover_poses, use_euler=True)

    assert np.isclose(platform_pose[0, 3], platform_magbots.max_platform_rot_ab)
    assert np.isclose(platform_pose[0, 4], -platform_magbots.max_platform_rot_ab)


def test_platform_z_mover_dist_roundtrip(platform_magbots):
    mover_z = np.full((NUM_SAMPLES,), 0.001)
    mover_dist = np.linspace(platform_magbots.min_dist_movers_m, platform_magbots.max_dist_movers_m, NUM_SAMPLES)
    platform_z = platform_magbots.moverDist2PlatformZPos(mover_dist=mover_dist, mover_z=mover_z)
    assert np.allclose(platform_magbots.platformZPos2MoverDist(platform_z_d=platform_z, mover_z_d=mover_z), mover_dist)

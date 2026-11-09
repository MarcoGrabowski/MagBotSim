import numpy as np
import pytest

from magbotsim import GripperMagBotsAPM4330, GripperMagBotsAPM4330Params, GripperStation

# Round-trip tests of the kinematics of the Gripper MagBot: gripperSetPose2MoverSetPose (inverse kinematics) and
# moverSetPose2GripperSetPose (forward kinematics) must be the exact inverse of each other for reachable poses.

NUM_SAMPLES = 200
PARAMS = GripperMagBotsAPM4330Params()
# finger (jaw) angle range of the finger actuator (ctrlrange) and the reachable distance between the two finger pads
FINGER_ANGLE_RANGE = (0.0, 1.07)  # [rad]
FINGER_DISTANCE_RANGE = (0.0002, 0.0228)  # [m]


@pytest.fixture
def magbots() -> GripperMagBotsAPM4330:
    """Gripper MagBots with a synthetic (monotonic, nonlinear) finger calibration, i.e. the kinematics can be tested without building
    the MuJoCo model (the calibration is usually built in ``update_cached_mujoco_data()``).
    """
    magbots = GripperMagBotsAPM4330(
        num_magbots=1,
        indices_mover_c0=np.array([0]),
        indices_mover_c1=np.array([1]),
        indices_mover_c2=np.array([2]),
        magbot_params=PARAMS,
        cycle_time=0.001,
    )
    finger_angles = np.linspace(*FINGER_ANGLE_RANGE, 25)
    finger_distances = FINGER_DISTANCE_RANGE[0] + (FINGER_DISTANCE_RANGE[1] - FINGER_DISTANCE_RANGE[0]) * np.sin(
        0.5 * np.pi * finger_angles / FINGER_ANGLE_RANGE[1]
    )
    magbots._set_finger_calibration(finger_distances=finger_distances, finger_angles=finger_angles)
    return magbots


def sample_reachable_gripper_poses(rng: np.random.Generator, z_m: np.ndarray) -> np.ndarray:
    """Sample gripper poses that are reachable, i.e. which are not clipped by ``gripperSetPose2MoverSetPose()``.

    :param rng: the random number generator
    :param z_m: the mover z-pos of each sample (numpy array of shape (NUM_SAMPLES,)) [m]
    :return: the gripper poses (numpy array of shape (NUM_SAMPLES, 7))
    """
    # the platform height must be within its limits; the gripper height follows from the platform height and the a-rotation and
    # must not fall below the minimum gripper height (gripper pointing downwards, i.e. a-rotation close to -pi)
    a_d, z_p = np.zeros(NUM_SAMPLES), np.zeros(NUM_SAMPLES)
    num_valid = 0
    while num_valid < NUM_SAMPLES:
        a_d_sample = rng.uniform(-np.pi, 0.0)
        z_p_sample = rng.uniform(PARAMS.z_p_min, PARAMS.z_p_max)
        if z_p_sample + z_m[num_valid] + PARAMS.n * np.cos(a_d_sample) >= PARAMS.z_d_min:
            a_d[num_valid], z_p[num_valid] = a_d_sample, z_p_sample
            num_valid += 1

    gripper_poses = np.zeros((NUM_SAMPLES, 7))
    gripper_poses[:, :2] = rng.uniform(-1.0, 1.0, (NUM_SAMPLES, 2))
    gripper_poses[:, 2] = z_p + z_m + PARAMS.n * np.cos(a_d)
    gripper_poses[:, 3] = a_d
    gripper_poses[:, 4] = rng.uniform(-0.5, 0.5, NUM_SAMPLES)
    # the forward kinematics returns c-rotations in (-pi, pi], i.e. c-rotations beyond +/-pi are not recovered by the round trip
    gripper_poses[:, 5] = rng.uniform(-np.pi, np.pi, NUM_SAMPLES)
    gripper_poses[:, 6] = rng.uniform(FINGER_DISTANCE_RANGE[0], PARAMS.f_d_max, NUM_SAMPLES)
    return gripper_poses


@pytest.mark.parametrize('single_track', [False, True])
def test_gripper_pose_round_trip(magbots, single_track):
    """Round trip: gripper pose -> mover poses -> gripper pose."""
    rng = np.random.default_rng(seed=42)
    z_m = rng.uniform(0.001, 0.004, NUM_SAMPLES)
    gripper_poses = sample_reachable_gripper_poses(rng, z_m=z_m)

    mover_poses = magbots.gripperSetPose2MoverSetPose(gripper_pose_d=gripper_poses, z_m=z_m, single_track=single_track)
    gripper_poses_rt = magbots.moverSetPose2GripperSetPose(mover_poses=mover_poses)

    assert gripper_poses_rt.shape == gripper_poses.shape
    assert np.allclose(gripper_poses_rt, gripper_poses, rtol=0.0, atol=1e-9)


def test_mover_pose_round_trip(magbots):
    """Round trip: mover poses -> gripper pose -> mover poses (for reachable mover poses)."""
    rng = np.random.default_rng(seed=0)
    z_m = rng.uniform(0.001, 0.004, NUM_SAMPLES)
    mover_poses = magbots.gripperSetPose2MoverSetPose(gripper_pose_d=sample_reachable_gripper_poses(rng, z_m=z_m), z_m=z_m)

    gripper_poses = magbots.moverSetPose2GripperSetPose(mover_poses=mover_poses)
    mover_poses_rt = magbots.gripperSetPose2MoverSetPose(gripper_pose_d=gripper_poses, z_m=z_m)

    assert np.allclose(mover_poses_rt, mover_poses, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize('station_yaw', np.deg2rad([-180.0, -135.0, -90.0, 0.0, 45.0, 90.0, 180.0]))
def test_station_mover_pose_round_trip(magbots, station_yaw):
    """The gripper pose deduced from the station mover poses must map exactly to the station mover poses, since the rail mounts
    are only switched between the station and the movers at these positions (see ReconfigurationEnv).
    """
    z_m = 0.001
    station = GripperStation(pos=np.array([0.48, 0.72, 0.01]), yaw=station_yaw)
    station_mover_poses = station.get_mover_poses(z_m=z_m)[np.newaxis]

    gripper_pose = magbots.moverSetPose2GripperSetPose(mover_poses=station_mover_poses)
    mover_poses_rt = magbots.gripperSetPose2MoverSetPose(gripper_pose_d=gripper_pose, z_m=z_m)

    assert np.allclose(mover_poses_rt, station_mover_poses, rtol=0.0, atol=1e-9)
    # the gripper is closed and horizontal in the station
    assert np.allclose(gripper_pose[0, 3:5], 0.0, atol=1e-9)
    assert np.isclose(gripper_pose[0, 6], FINGER_DISTANCE_RANGE[0])

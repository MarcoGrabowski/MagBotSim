import numpy as np

from magbotsim.magbots.gripper import GripperMagBotsAPM4330Params
from magbotsim.stations.station_collision_proxies import StationCollisionProxies


def _magbot_m2_offset(m0m1_dist: float, magbot_params: GripperMagBotsAPM4330Params) -> float:
    """Calculate the distance between the mover attached to connection 2 and the line through the movers attached to connection 0 and 1
    for a given distance between the movers attached to connection 0 and 1, i.e. the m2 offset that results from the kinematics of the
    Gripper MagBot (platform height as in ``moverSetPose2GripperSetPose``, mover 2 distance as in ``gripperSetPose2MoverSetPose`` without
    single-track mode).

    :param m0m1_dist: the distance between the movers attached to connection 0 and 1 [m]
    :param magbot_params: the parameters of the Gripper MagBot
    :return: the distance between the mover attached to connection 2 and the line through the movers attached to connection 0 and 1 [m]
    """
    p = magbot_params
    z_p = p.z_b01 + p.z_t01 + np.sqrt(p.k_01**2 - (0.5 * m0m1_dist - p.x_b01 - p.x_t01) ** 2)
    return float(p.x_t2 + p.x_b2 + np.sqrt(p.k_2**2 - (z_p - p.z_t2 - p.z_b2) ** 2))


class GripperStation:
    """A station in which a Gripper MagBot can be parked, i.e. the movers can leave the MagBot in the station and pick it up again.
    The station consists of the m0, m1 and m2 station parts and the m2 mount. The parts have a fixed relative position to each other.
    The station m2 (including its mount) is used as the reference, i.e. it is located at the origin of the station body frame. The
    station fits the default configuration of the MagBot, i.e. the MagBot can be generated directly in the station.

    The movers enter and leave the station along the y-axis of the station body frame (rail mounts welded to the station). The
    MagBot is taken out of the station and moved back into it along the x-axis of the station body frame (rail mounts welded to the
    movers).

    The station meshes are visual-only. If a ``StationCollisionProxies`` instance is specified in ``generate_body_xml_str()``, the
    station gets hidden pins as collision proxies, which hold the parked rail mounts (see ``StationCollisionProxies``).

    :param pos: the (x, y, z) position of the m2 mount in the world frame [m]
    :param yaw: the rotation of the whole station about the world z-axis [rad], defaults to 0.0
    :param name: the name of the station body in the MuJoCo model, defaults to 'station'
    """

    MAGBOT_M0M1_DIST = 0.48         # [m] distance between mover 0 and mover 1
    # [m] distance between mover 2 and the line through mover 0 and mover 1 (approx. 0.12 m); derived from the kinematics of the default
    # Gripper MagBot, so that the mover poses in the station are exactly reachable (see get_mover_poses)
    MAGBOT_M2_OFFSET = _magbot_m2_offset(MAGBOT_M0M1_DIST, GripperMagBotsAPM4330Params())
    MOVER_C0C1_REL_X_POS = 0.0851   # [m] x-pos of the movers attached to connection 0 and 1 relative to the m2 mount (station body frame)

    # mover positions at which the MagBot is located in the station relative to the m2 mount (station body frame)
    MOVER_REL_XY_POS = np.array(
        [
            [MOVER_C0C1_REL_X_POS, -0.5 * MAGBOT_M0M1_DIST],
            [MOVER_C0C1_REL_X_POS, 0.5 * MAGBOT_M0M1_DIST],
            [MOVER_C0C1_REL_X_POS + MAGBOT_M2_OFFSET, 0.0],
        ]
    )  # [m]
    MOVER_REL_YAW = np.pi / 2  # [rad]

    STATION_M0M1_Y_POS = 0.5 * MAGBOT_M0M1_DIST  # [m] y-pos of the m0/m1 station meshes relative to the m2 mount (station body frame, before the mesh origin correction)
    STATION_M2_MOVER_C2_X_DIST = 0.22401599     # [m] x-distance between the m2 station mesh and the mover c2 at which rail mount 2 is fully seated in the m2 station

    STATION_M0M1_X_OFFSET = 0.05665             # [m] offset of the m0/m1 station mesh in its x-direction (mesh frame)
    STATION_M2_CORRECTION = 0.025               # [m] correction of the m2 station and m2 mount meshes in their y-direction (mesh frame)
    # z-pos of the station meshes relative to the m2 mount, such that the station pins are centered in the holes of the rail mounts
    STATION_M0M1_Z_POS = 0.00515                # [m]
    STATION_M2_Z_POS = 0.00185                  # [m]

    def __init__(self, pos: np.ndarray, yaw: float = 0.0, name: str = 'station') -> None:
        assert pos.shape == (3,)
        self.pos = pos.copy()
        self.yaw = yaw
        self.name = name

    def rotation_matrix_xy(self) -> np.ndarray:
        """Return the 2D rotation matrix of the station about the world z-axis (station body frame -> world frame).

        :return: the rotation matrix (numpy array of shape (2, 2))
        """
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s], [s, c]])

    def get_mover_poses(self, z_m: float) -> np.ndarray:
        """Return the mover poses in the world frame at which the MagBot is located in the station. At these positions, the rail mounts
        can be switched between the movers and the station. The rotation of the station is considered.

        :param z_m: the mover z-pos (distance between the tile surface and the bottom of a mover) [m]
        :return: the mover poses (numpy array of shape (3, 6) using Euler angles (xyz) in rad).
            [0,:]: pose of the mover attached to connection 0; [1,:]: connection 1; [2,:]: connection 2
        """
        mover_poses = np.zeros((3, 6))
        mover_poses[:, :2] = self.pos[:2] + self.MOVER_REL_XY_POS @ self.rotation_matrix_xy().T
        mover_poses[:, 2] = z_m
        mover_poses[:, 5] = np.arctan2(np.sin(self.yaw + self.MOVER_REL_YAW), np.cos(self.yaw + self.MOVER_REL_YAW))
        return mover_poses

    def is_along_entry_axis(self, delta_xy: np.ndarray) -> bool:
        """Check whether a movement in the world frame is mainly along the entry axis of the station (y-axis of the station body frame),
        i.e. whether a mover enters or leaves the station without the MagBot.

        :param delta_xy: the (x, y) movement in the world frame [m]
        :return: True if the movement is mainly along the entry axis, False otherwise
        """
        delta_station = self.rotation_matrix_xy().T @ delta_xy
        return bool(np.abs(delta_station[1]) > np.abs(delta_station[0]))

    @staticmethod
    def generate_assets_xml_str() -> str:
        """Generate the XML string for the station meshes. The assets are only required once, regardless of the number of stations.

        :return: the XML string of the station assets
        """
        assets_list = [
            '\n\t\t<!-- assets station -->',
            '\t\t<mesh name="station_m0m1_mesh" file="./misc/reconfiguration_benchmark/station_m0m1.stl" scale="1 1 1"/>',
            '\t\t<mesh name="station_mount_m2_mesh" file="./misc/reconfiguration_benchmark/station_mount_m2.stl" scale="1 1 1"/>',
            '\t\t<mesh name="station_m2_mesh" file="./misc/reconfiguration_benchmark/station_m2.stl" scale="1 1 1"/>',
        ]
        return '\n'.join(assets_list)

    def generate_body_xml_str(self, proxies: StationCollisionProxies | None = None, rail_mount_z: float | None = None) -> str:
        """Generate the XML string for the station body.

        :param proxies: the hidden collision proxies for the station switch, defaults to None. If specified, the station pins are
            added to the station body. If None, the station does not collide with anything.
        :param rail_mount_z: the z-pos of the rail mount origin of a parked MagBot in the world frame [m], i.e. the z-pos of the mover
            adapter (mover z-pos + mover half height + 0.006 m for the Gripper MagBot). Required if ``proxies`` is specified.
        :return: the XML string of the body containing the station
        """
        # positions of the m0 and m1 station relative to the m2 station (station body frame)
        # the m0/m1 station meshes are rotated by -90 deg, i.e. the mesh x-direction corresponds to the -y-direction of the station body
        rel_pos_m0 = np.array([0.0, self.STATION_M0M1_Y_POS + self.STATION_M0M1_X_OFFSET, self.STATION_M0M1_Z_POS])  # [m]
        rel_pos_m1 = np.array([0.0, -self.STATION_M0M1_Y_POS + self.STATION_M0M1_X_OFFSET, self.STATION_M0M1_Z_POS])  # [m]
        rel_pos_m2 = np.array(
            [self.MOVER_REL_XY_POS[2, 0] - self.STATION_M2_MOVER_C2_X_DIST, self.STATION_M2_CORRECTION, self.STATION_M2_Z_POS]
        )  # [m]
        rel_pos_mount_m2 = np.array([0.0, self.STATION_M2_CORRECTION, 0.0])  # [m]

        n = self.name
        lines = [
            f'\n\t\t<!-- {n} -->',
            f'\t\t<body name="{n}" pos="{self.pos[0]:.5f} {self.pos[1]:.5f} {self.pos[2]:.5f}" euler="0 0 {self.yaw:.7f}">',
            # m2 station (reference)
            f'\t\t\t<geom name="{n}_mount_m2_geom" type="mesh" mesh="station_mount_m2_mesh" mass="0.001" '
            f'pos="{rel_pos_mount_m2[0]:.5f} {rel_pos_mount_m2[1]:.5f} {rel_pos_mount_m2[2]:.5f}" material="gray" contype="0" '
            'conaffinity="0"/>',
            f'\t\t\t<geom name="{n}_m2_geom" type="mesh" mesh="station_m2_mesh" mass="0.001" '
            f'pos="{rel_pos_m2[0]:.5f} {rel_pos_m2[1]:.5f} {rel_pos_m2[2]:.5f}" material="black" contype="0" conaffinity="0"/>',
            # m0 station
            f'\t\t\t<geom name="{n}_m0_geom" type="mesh" mesh="station_m0m1_mesh" mass="0.001" '
            f'pos="{rel_pos_m0[0]:.5f} {rel_pos_m0[1]:.5f} {rel_pos_m0[2]:.5f}" euler="0 0 -1.5707963" material="black" '
            'contype="0" conaffinity="0"/>',
            # m1 station
            f'\t\t\t<geom name="{n}_m1_geom" type="mesh" mesh="station_m0m1_mesh" mass="0.001" '
            f'pos="{rel_pos_m1[0]:.5f} {rel_pos_m1[1]:.5f} {rel_pos_m1[2]:.5f}" euler="0 0 -1.5707963" material="black" '
            'contype="0" conaffinity="0"/>',
        ]
        if proxies is not None:
            assert rail_mount_z is not None, 'rail_mount_z is required if station collision proxies are specified'
            lines.append(
                proxies.station_pins_xml_str(
                    body_name=f'{n}_pins', rail_mount_xy_pos=self.MOVER_REL_XY_POS, rail_mount_z=rail_mount_z - self.pos[2]
                )
            )
        lines.append('\t\t</body>')
        return '\n'.join(lines)

    def generate_equality_xml_str(self, idx_magbot: int) -> str:
        """Generate the XML string for the welds between the rail mounts of a MagBot and the station. The welds are inactive at the
        beginning, i.e. they must be activated by the environment.

        :param idx_magbot: the index of the MagBot whose rail mounts are welded to this station
        :return: the XML string of the welds (without the surrounding equality tag)
        """
        lines = [
            f'\t\t<weld name="{self.name}_weld_c{idx_c}_{idx_magbot}" body1="rail_mount_{idx_c}_{idx_magbot}" body2="{self.name}" '
            'active="false"/>'
            for idx_c in range(0, 3)
        ]
        return '\n'.join(lines)

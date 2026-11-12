import numpy as np

from magbotsim.magbots.gripper import GripperMagBotsAPM4330Params
from magbotsim.stations.station import Station


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


class GripperStation(Station):
    """A station in which a Gripper MagBot can be parked, i.e. the movers can leave the MagBot in the station and pick it up again.
    The station consists of the m0, m1 and m2 station parts and the m2 mount. The parts have a fixed relative position to each other.
    The station m2 mount is used as the anchor, i.e. it is located at the origin of the station body frame. The station fits the default
    configuration of the MagBot, i.e. the MagBot can be generated directly in the station (see ``Station`` for the entry and take-out
    axes and the collision proxies).

    :param pos: the (x, y, z) position of the m2 mount in the world frame [m]
    :param yaw: the rotation of the whole station about the world z-axis [rad], defaults to 0.0
    :param name: the name of the station body in the MuJoCo model, defaults to 'station'
    """

    MAGBOT_M0M1_DIST = 0.48         # [m] distance between mover 0 and mover 1
    # [m] distance between mover 2 and the line through mover 0 and mover 1 (approx. 0.12 m); derived from the kinematics of the default
    # Gripper MagBot, so that the mover poses in the station are exactly reachable (see get_mover_poses)
    MAGBOT_M2_OFFSET = _magbot_m2_offset(MAGBOT_M0M1_DIST, GripperMagBotsAPM4330Params())
    MOVER_C0C1_REL_X_POS = Station.MOVER_C0C1_REL_X_POS  # [m] x-pos of the movers attached to connection 0 and 1 relative to the m2 mount

    # mover positions at which the MagBot is located in the station relative to the m2 mount (station body frame)
    MOVER_REL_XY_POS = np.array(
        [
            [MOVER_C0C1_REL_X_POS, -0.5 * MAGBOT_M0M1_DIST],
            [MOVER_C0C1_REL_X_POS, 0.5 * MAGBOT_M0M1_DIST],
            [MOVER_C0C1_REL_X_POS + MAGBOT_M2_OFFSET, 0.0],
        ]
    )  # [m]
    # the movers attached to connection 0 enter from -y, the movers attached to connection 1 and 2 from +y
    MOVER_ENTRY_SIGNS = np.array([-1.0, 1.0, 1.0])
    MESH_ASSETS = {
        'station_m0m1_mesh': './misc/reconfiguration_benchmark/station_m0m1.stl',
        'station_mount_m2_mesh': './misc/reconfiguration_benchmark/station_mount_m2.stl',
        'station_m2_mesh': './misc/reconfiguration_benchmark/station_m2.stl',
    }

    # [m] y-pos of the m0/m1 station meshes relative to the m2 mount (station body frame, before the mesh origin correction)
    STATION_M0M1_Y_POS = 0.5 * MAGBOT_M0M1_DIST
    # [m] x-distance between the m2 station mesh and the mover c2 at which rail mount 2 is fully seated in the m2 station
    STATION_M2_MOVER_C2_X_DIST = 0.22401599

    STATION_M2_CORRECTION = 0.025               # [m] correction of the m2 station and m2 mount meshes in their y-direction (mesh frame)
    # z-pos of the m2 station mesh relative to the m2 mount, such that the station pins are centered in the holes of the rail mounts
    STATION_M2_Z_POS = 0.00185                  # [m]

    def _mesh_geoms_xml_str(self) -> list[str]:
        """Generate the XML strings of the (visual-only) m2 mount, m2, m0 and m1 station part geoms.

        :return: a list of XML strings (one per geom)
        """
        rel_pos_m2 = np.array(
            [self.MOVER_REL_XY_POS[2, 0] - self.STATION_M2_MOVER_C2_X_DIST, self.STATION_M2_CORRECTION, self.STATION_M2_Z_POS]
        )  # [m]
        rel_pos_mount_m2 = np.array([0.0, self.STATION_M2_CORRECTION, 0.0])  # [m]

        n = self.name
        return [
            # m2 station (reference)
            f'\t\t\t<geom name="{n}_mount_m2_geom" type="mesh" mesh="station_mount_m2_mesh" mass="0.001" '
            f'pos="{rel_pos_mount_m2[0]:.5f} {rel_pos_mount_m2[1]:.5f} {rel_pos_mount_m2[2]:.5f}" material="gray" contype="0" '
            'conaffinity="0"/>',
            f'\t\t\t<geom name="{n}_m2_geom" type="mesh" mesh="station_m2_mesh" mass="0.001" '
            f'pos="{rel_pos_m2[0]:.5f} {rel_pos_m2[1]:.5f} {rel_pos_m2[2]:.5f}" material="black" contype="0" conaffinity="0"/>',
            # m0 and m1 station (the geom m0 holds the rail mount at +y, the geom m1 the rail mount at -y)
            self._station_m0m1_geom_xml_str(geom_name=f'{n}_m0_geom', mover_y_pos=self.STATION_M0M1_Y_POS),
            self._station_m0m1_geom_xml_str(geom_name=f'{n}_m1_geom', mover_y_pos=-self.STATION_M0M1_Y_POS),
        ]

import numpy as np

from magbotsim.stations.station_collision_proxies import StationCollisionProxies


class Station:
    """Base class of a station in which a MagBot can be parked, i.e. the movers can leave the MagBot in the station and pick it up again.
    A station consists of several station parts with a fixed relative position to each other. Each connection (rail mount) of the
    MagBot is held by the pins of one station part. The station is anchored at a reference point (origin of the station body frame),
    e.g. a mount point on an aluminum extrusion. The station fits the default configuration of the MagBot, i.e. the MagBot can be
    generated directly in the station.

    The movers enter and leave the station along the y-axis of the station body frame (rail mounts welded to the station). The
    MagBot is taken out of the station and moved back into it along the x-axis of the station body frame (rail mounts welded to the
    movers).

    The station meshes are visual-only. If a ``StationCollisionProxies`` instance is specified in ``generate_body_xml_str()``, the
    station gets hidden pins as collision proxies, which hold the parked rail mounts (see ``StationCollisionProxies``).

    Subclasses define the mover positions in the station (``MOVER_REL_XY_POS``), the entry sides (``MOVER_ENTRY_SIGNS``), the mesh
    assets (``MESH_ASSETS``) and the station part geoms (``_mesh_geoms_xml_str()``).

    :param pos: the (x, y, z) position of the station anchor in the world frame [m]
    :param yaw: the rotation of the whole station about the world z-axis [rad], defaults to 0.0
    :param name: the name of the station body in the MuJoCo model, defaults to 'station'
    """

    # mover positions at which the MagBot is located in the station relative to the station anchor (station body frame), in
    # connection order, i.e. [idx_c, :]: position of the mover attached to connection idx_c (numpy array of shape (num_connections, 2))
    MOVER_REL_XY_POS: np.ndarray = np.zeros((0, 2))  # [m]
    MOVER_REL_YAW = np.pi / 2  # [rad]
    # side along the y-axis of the station body frame from which the mover attached to connection idx_c enters and leaves the station
    # (+1: from +y, -1: from -y), i.e. the side to which the flat side of the rail mount points (numpy array of shape (num_connections,))
    MOVER_ENTRY_SIGNS: np.ndarray = np.zeros(0)
    # mesh assets of the station: {mesh name: file relative to the mesh directory}
    MESH_ASSETS: dict[str, str] = {}

    # station part that holds a single rail mount and is mounted directly to the extrusion (station_m0m1.stl)
    MOVER_C0C1_REL_X_POS = 0.0851  # [m] x-pos of the movers held by this station part relative to the anchor (station body frame)
    STATION_M0M1_X_OFFSET = 0.05665  # [m] offset of the station part mesh in its x-direction (mesh frame)
    # z-pos of the station part mesh relative to the anchor, such that the station pins are centered in the holes of the rail mounts
    STATION_M0M1_Z_POS = 0.00515  # [m]

    def __init__(self, pos: np.ndarray, yaw: float = 0.0, name: str = 'station') -> None:
        assert pos.shape == (3,)
        assert self.MOVER_REL_XY_POS.shape == (len(self.MOVER_ENTRY_SIGNS), 2)
        self.pos = pos.copy()
        self.yaw = yaw
        self.name = name

    @property
    def num_connections(self) -> int:
        """The number of MagBot connections (rail mounts) held by the station."""
        return self.MOVER_REL_XY_POS.shape[0]

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
        :return: the mover poses (numpy array of shape (num_connections, 6) using Euler angles (xyz) in rad).
            [idx_c,:]: pose of the mover attached to connection idx_c
        """
        mover_poses = np.zeros((self.num_connections, 6))
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

    @classmethod
    def generate_assets_xml_str(cls) -> str:
        """Generate the XML string for the station meshes. The assets are only required once, regardless of the number of stations.

        :return: the XML string of the station assets
        """
        assets_list = ['\n\t\t<!-- assets station -->']
        assets_list += [f'\t\t<mesh name="{name}" file="{file}" scale="1 1 1"/>' for name, file in cls.MESH_ASSETS.items()]
        return '\n'.join(assets_list)

    def _station_m0m1_geom_xml_str(self, geom_name: str, mover_y_pos: float) -> str:
        """Generate the XML string of a station part (station_m0m1.stl) that holds the rail mount of a mover at the y-pos
        ``mover_y_pos`` (station body frame). The mesh is rotated by -90 deg, i.e. the mesh x-direction corresponds to the -y-direction
        of the station body.

        :param geom_name: the name of the geom
        :param mover_y_pos: the y-pos of the mover held by the station part relative to the anchor (station body frame) [m]
        :return: the XML string of the geom
        """
        rel_pos = np.array([0.0, mover_y_pos + self.STATION_M0M1_X_OFFSET, self.STATION_M0M1_Z_POS])  # [m]
        return (
            f'\t\t\t<geom name="{geom_name}" type="mesh" mesh="station_m0m1_mesh" mass="0.001" '
            f'pos="{rel_pos[0]:.5f} {rel_pos[1]:.5f} {rel_pos[2]:.5f}" euler="0 0 -1.5707963" material="black" '
            'contype="0" conaffinity="0"/>'
        )

    def _mesh_geoms_xml_str(self) -> list[str]:
        """Generate the XML strings of the (visual-only) station part geoms, which are added to the station body.

        :return: a list of XML strings (one per geom)
        """
        raise NotImplementedError

    def generate_body_xml_str(self, proxies: StationCollisionProxies | None = None, rail_mount_z: float | None = None) -> str:
        """Generate the XML string for the station body.

        :param proxies: the hidden collision proxies for the station switch, defaults to None. If specified, the station pins are
            added to the station body. If None, the station does not collide with anything.
        :param rail_mount_z: the z-pos of the rail mount origin of a parked MagBot in the world frame [m], i.e. the z-pos of the mover
            adapter (mover z-pos + mover half height + 0.006 m). Required if ``proxies`` is specified.
        :return: the XML string of the body containing the station
        """
        n = self.name
        lines = [
            f'\n\t\t<!-- {n} -->',
            f'\t\t<body name="{n}" pos="{self.pos[0]:.5f} {self.pos[1]:.5f} {self.pos[2]:.5f}" euler="0 0 {self.yaw:.7f}">',
            *self._mesh_geoms_xml_str(),
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
            for idx_c in range(0, self.num_connections)
        ]
        return '\n'.join(lines)

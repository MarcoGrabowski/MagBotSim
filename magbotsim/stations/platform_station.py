import numpy as np

from magbotsim.magbots.sixD_platform import SixDPlatformMagBotsAPM4330
from magbotsim.stations.station import Station


class PlatformStation(Station):
    """A station in which a 6D-Platform MagBot can be parked, i.e. the movers can leave the MagBot in the station and pick it up again.
    The station consists of two station parts (the same parts as the m0 and m1 parts of the ``GripperStation``), which are mounted
    directly to the extrusion. Their distance corresponds to the default distance between the rail mounts of the MagBot
    (``SixDPlatformMagBotsAPM4330.DEFAULT_MOVER_DIST``), i.e. the MagBot can be generated directly in the station.

    The anchor of the station (origin of the station body frame) is the middle point on the extrusion between both station parts, i.e.
    it corresponds to the m2 mount of the ``GripperStation``. Thus, a ``PlatformStation`` and a ``GripperStation`` with the same pose
    occupy the same place on the extrusion. See ``Station`` for the entry and take-out axes and the collision proxies.

    The mover attached to connection 0 (rail mount 0, mover b) is located at +y of the station body frame, the mover attached to
    connection 1 (rail mount 1, mover a) at -y. Thus, the platform c-rotation (direction from mover a to mover b) is equal to the mover
    rotation in the station, i.e. the platform of a parked MagBot is not tilted.

    :param pos: the (x, y, z) position of the station anchor in the world frame [m]
    :param yaw: the rotation of the whole station about the world z-axis [rad], defaults to 0.0
    :param name: the name of the station body in the MuJoCo model, defaults to 'station'
    """

    MAGBOT_MOVER_DIST = SixDPlatformMagBotsAPM4330.DEFAULT_MOVER_DIST  # [m] distance between mover a and mover b

    # mover positions at which the MagBot is located in the station relative to the anchor (station body frame)
    MOVER_REL_XY_POS = np.array(
        [
            [Station.MOVER_C0C1_REL_X_POS, 0.5 * MAGBOT_MOVER_DIST],  # connection 0 (mover b)
            [Station.MOVER_C0C1_REL_X_POS, -0.5 * MAGBOT_MOVER_DIST],  # connection 1 (mover a)
        ]
    )  # [m]
    # the flat sides of both rail mounts point outwards, i.e. the mover attached to connection 0 enters from +y, connection 1 from -y
    MOVER_ENTRY_SIGNS = np.array([1.0, -1.0])
    MESH_ASSETS = {'station_m0m1_mesh': './misc/reconfiguration_benchmark/station_m0m1.stl'}

    def _mesh_geoms_xml_str(self) -> list[str]:
        """Generate the XML strings of the (visual-only) station part geoms (one station part per connection).

        :return: a list of XML strings (one per geom)
        """
        return [
            self._station_m0m1_geom_xml_str(geom_name=f'{self.name}_p{idx_c}_geom', mover_y_pos=self.MOVER_REL_XY_POS[idx_c, 1])
            for idx_c in range(0, self.num_connections)
        ]

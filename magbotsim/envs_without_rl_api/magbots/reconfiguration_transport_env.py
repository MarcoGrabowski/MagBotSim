from pathlib import Path
from typing import Any
import numpy as np
import mujoco
from magbotsim import (
    BasicMagBotEnv,
    MoverImpedanceController,
    PlatformStation,
    SixDPlatformMagBotsAPM4330,
    StationCollisionProxies,
)
from magbotsim.magbots.mover_coupling import NO_INDEX
from magbotsim.utils import geometry_2D_utils, rotations_utils


class ReconfigurationTransportEnv(BasicMagBotEnv):
    """Transport benchmark environment with two movers and one 6D-Platform MagBot. The tile layout, the profile frame and the station
    are the same as in ``ReconfigurationEnv``. At the beginning, the MagBot is parked in a ``PlatformStation`` at the top extrusion and
    the movers are free on the table at the positions ``initial_mover_xy_pos``.

    The task is an infinite transport task: the movers pick up the MagBot from the station, take a product from the supply rack at the
    bottom long side of the table (the product can only be lifted off the poles of the rack by the platform) and drop it into the bin
    next to the supply rack (also at the bottom long side) by tilting the platform. Since the supply rack and the bin are located at the
    same side of the table, the MagBot does not have to turn (platform c-rotation 0). A product that rests in the bin is counted as
    delivered (``num_delivered_products``) and respawned in the supply rack (see ``respawn_product()``), i.e. the supply never runs empty. A
    product that rests anywhere else below the platform heights (table, frame, floor) is counted as dropped (``num_dropped_products``)
    and respawned as well. The supply rack, the product and the bin are those of ``SixDPlatformMagBotApplicationExampleEnv``.

    The station mechanics are the same as in ``ReconfigurationEnv``: both movers carry the same mover adapter, i.e. any mover can pick
    up any rail mount of the MagBot (see ``MagBotMoverCoupling``). The assignment of the movers to the connections is determined at
    runtime: a mover is attached to a rail mount when it picks up the rail mount from the station and detached when it puts the rail
    mount back (see ``_update_equality_constraints()``). The current assignment is given by ``connection_mover_indices`` (and
    ``mover_magbot_indices``, ``mover_connection_indices``), the station in which a rail mount is parked by
    ``connection_station_indices``. Connection 0 (rail mount 0) is attached to the mover that controls the platform's beta-rotation
    (mover b) and connection 1 (rail mount 1) to the mover that controls the alpha-rotation (mover a), see
    ``SixDPlatformMagBotsAPM4330``. The mover docked at connection 1 is rotated by pi (see ``MOVER_YAW_OFFSETS``).

    The movers are controlled by impedance controllers (all DoFs). The platform a,b-rotations result from the rail mount rotations,
    which are held by the movers or by the station.

    At initialization, the full model XML (``model_xml_str``) is saved next to this script (``reconfiguration_transport_env.xml``, see
    ``model_xml_path``), e.g. to inspect the generated model or to load it in the MuJoCo viewer. The file is overwritten by every new
    environment instance.

    The rotation limits of the MagLev system are checked in every cycle (see ``check_rotation_violation()``): a full rotation of a mover
    around the z-axis is only possible within ``rotation_tol`` of one of the default rotation points (``get_tile_rotation_points()``, i.e.
    the centers, corners and edge midpoints of the tiles that are surrounded by tiles). At all other positions, the yaw of a mover may
    deviate at most ``max_yaw_deviation`` from the nearest multiple of 90 deg. Thus, the movers rotate to the docking yaws at rotation
    points, and the platform is tilted at the bin with mover a (alpha-rotation) at a rotation point, since the platform a,b-rotations
    rotate movers a and b. A rotation violation is treated like a collision.

    ``step()`` returns an info dictionary that contains the collision types and the product events (all bool):

    - ``mover_collision``: collision between two movers
    - ``wall_collision``: collision between a mover and a wall
    - ``station_proxy_collision``: contact of any station collision proxy (rail mount proxy, rail rod or station pin) with any other
      geom (see ``get_station_proxy_collisions()``)
    - ``rail_rod_collision``: contact of a rail rod, e.g. an empty mover enters or leaves a station with a sideways misalignment
      larger than the station proxy tolerance (the rail rod hits the jaws of a rail mount)
    - ``station_pin_collision``: contact of a station pin, e.g. a MagBot carried by movers is moved into or out of a station with a
      sideways misalignment larger than the station proxy tolerance (the rail mount hits the station pins)
    - ``rotation_violation``: a mover violates the rotation limits (see ``check_rotation_violation()``)
    - ``product_delivered``: the product came to rest in the bin during this step (it has been respawned in the supply rack)
    - ``product_dropped``: the product came to rest outside the bin below the platform heights during this step (it has been respawned
      in the supply rack)

    :param initial_mover_xy_pos: the initial (x, y) positions of the movers on the table [m] (numpy array of shape (2, 2)), defaults to
        None, i.e. the positions are sampled uniformly on the free space of the table (no wall, mover or parked rail mount collisions)
    :param station_switch_tol: the maximum distance between a rail mount and the nominal station position (put-back), or between a
        mover set position and a parked rail mount (pick-up), at which the welds are switched [m], defaults to 0.001
    :param station_proxy_tol: the sideways misalignment with which a mover (or the MagBot carried by the movers) can enter the station
        without colliding [m], defaults to 0.002. The station switch is modelled by hidden collision proxies (rail mount proxies and
        station pins, see ``StationCollisionProxies``), which are shown in the viewer by pressing '3' (geom group 3). With 0, the proxies
        match the real geometries exactly, i.e. in practice they always collide.
    :param randomize_magbot_station_pos: whether to shift the MagBot in the station by a random offset along the station x-axis
        (take-out axis, i.e. along the station pins) at each reset, defaults to False. All rail mounts get the same offset.
    :param magbot_station_randomization_x_range: the half-range of the uniform random offset of the MagBot along the station x-axis
        [m], defaults to 0.005. Only used if ``randomize_magbot_station_pos`` is True.
    :param rotation_tol: the maximum (x,y) distance between the center of a mover and a rotation point at which a full rotation is
        still possible [m], defaults to 0.001 (see ``check_rotation_violation()``)
    :param max_yaw_deviation: the maximum deviation of the yaw of a mover from the nearest multiple of 90 deg if the mover is not at a
        rotation point [rad], defaults to 10 deg (see ``check_rotation_violation()``)
    :param seed: the seed of the RNG used for the randomization (``self.rng``), defaults to None
    :param render_mode: the render mode, defaults to 'human'. Use None to run without rendering (e.g. in tests).
    """

    NUM_MOVERS = 2
    NUM_CONNECTIONS = SixDPlatformMagBotsAPM4330.NUM_CONNECTIONS
    # single station at the top extrusion (pose of the station anchor: x, y, z, yaw), same as in ReconfigurationEnv
    STATION_POSES = np.array([[0.480, 0.720, 0.01, np.deg2rad(-90)]])
    # MuJoCo options of the Gripper MagBot (set by GripperMagBotsAPM4330), which are also used by ReconfigurationEnv for the 6D-Platform
    # MagBot
    OPTION_XML_STR = (
        '\t<option timestep="0.001" cone="elliptic" jacobian="dense" gravity="0 0 -9.81" '
        'integrator="implicitfast"><flag diagexact="enable"/></option>'
    )
    # joint mask of the impedance controllers: all DoFs (the movers attached to a 6D-Platform MagBot control their c-rotation themselves)
    JOINT_MASK = np.array([1, 1, 1, 1, 1, 1])

    # supply rack, product and bin of SixDPlatformMagBotApplicationExampleEnv. All z-positions are shifted by the difference between the
    # top of the profile frame (0.065 m) and the top of the extrusions of the application example (0.06 m), so that the rack and the bin
    # are mounted to the profile frame in the same way.
    FRAME_Z_OFFSET = 0.005  # [m]
    # pose of the supply rack (x, y, z, yaw) at the bottom long side of the table, same x,y-position as in the application example
    SUPPLY_RACK_POSE = np.array([0.66518, 0.0055, 0.03 + FRAME_Z_OFFSET, np.pi])
    # position of the product in the supply rack (x, y, z) [m]
    PRODUCT_SUPPLY_POS = np.array([0.61019, 0.0673, 0.22752 + FRAME_Z_OFFSET])
    # pose of the bin (x, y, z, yaw) at the bottom long side of the table, left of the supply rack: the container of the application
    # example (top long side) mirrored to the bottom long side, i.e. it hangs outside the bottom profile of the frame (x from 0.22041 to
    # 0.42041 m, i.e. centered below the platform at the bin, see BIN_PLATFORM_POSE)
    BIN_POSE = np.array([0.22041, -0.0555, -0.037 + FRAME_Z_OFFSET, 0.0])
    # inner box of the bin in the bin body frame: (x_min, x_max), (y_min, y_max), z_max (rim) [m]
    BIN_INNER_X_RANGE = (0.003, 0.197)
    BIN_INNER_Y_RANGE = (-0.150, 0.0)
    BIN_RIM_Z = 0.15
    # a product that rests outside the bin below this z-position is dropped [m]
    PRODUCT_DROP_Z = 0.1
    # the product is at rest if its linear speed is below this value [m/s]
    PRODUCT_REST_SPEED = 0.05

    # platform set poses (x, y, z, a, b, c) of the transport task [m, rad]. The platform c-rotation is always 0, i.e. the movers stay
    # parallel to the bottom long side, at which the supply rack and the bin are located (as at the supply of the application example).
    SUPPLY_APPROACH_PLATFORM_POSE = np.array([0.56133, 0.36, 0.205 + FRAME_Z_OFFSET, 0.0, 0.0, 0.0])
    SUPPLY_PLATFORM_POSE = np.array([0.61019, 0.133, 0.205 + FRAME_Z_OFFSET, 0.0, 0.0, 0.0])  # under the product
    # the product is lifted off the poles of the supply rack, but the platform stays below the poles
    PICK_PLATFORM_Z = 0.235 + FRAME_Z_OFFSET
    # the platform is pulled out of the supply rack with the product (the platform is clear of the poles) and moves to the bin at this
    # y-position
    RETRACT_PLATFORM_Y = 0.222
    # above the bin, the platform is close to the bottom long side, i.e. the product rests at the bin-side edge of the platform. When the
    # platform is tilted towards the bin, the product tips over this edge and falls over the frame into the bin. Tilting the platform
    # rotates mover a (gear ratio ~8.4, i.e. the max. yaw deviation of 10 deg only allows a tilt of ~1.2 deg), i.e. mover a must be at a
    # rotation point: with the mover distance at PICK_PLATFORM_Z (0.40082 m), mover a is at the center of the bottom left tile
    # (0.12, 0.12) and mover b at (0.52082, 0.12).
    BIN_PLATFORM_POSE = np.array([0.32041, 0.12, PICK_PLATFORM_Z, 0.0, 0.0, 0.0])
    DUMP_PLATFORM_A_ROT = 0.244  # [rad] tilt towards -y (bin), i.e. the max. platform a-rotation

    def __init__(
        self,
        initial_mover_xy_pos: np.ndarray | None = None,
        station_switch_tol: float = 0.001,
        station_proxy_tol: float = 0.002,
        randomize_magbot_station_pos: bool = False,
        magbot_station_randomization_x_range: float = 0.005,
        rotation_tol: float = 0.001,
        max_yaw_deviation: float = np.deg2rad(10.0),
        seed: int | None = None,
        render_mode: str | None = 'human',
    ) -> None:
        # randomization
        self.rng = np.random.default_rng(seed)
        self.randomize_magbot_station_pos = randomize_magbot_station_pos
        assert magbot_station_randomization_x_range >= 0.0
        self.magbot_station_randomization_x_range = magbot_station_randomization_x_range

        # rotation limits (the default rotation points of the tiles are used, see check_rotation_violation())
        assert rotation_tol >= 0.0 and max_yaw_deviation >= 0.0
        self.rotation_tol = rotation_tol
        self.max_yaw_deviation = max_yaw_deviation

        self.num_connections = self.NUM_CONNECTIONS
        mover_count = self.NUM_MOVERS

        # mover parameters, collision parameters and initial start positions
        mover_mass = np.array([1.264] * mover_count)
        bumper_mass = np.array([0.05] * mover_count)
        mover_params = {
            'shape': 'mesh',
            'mesh': {
                'mover_stl_path': ['beckhoff_apm4330_mover'] * mover_count,
                'bumper_stl_path': ['beckhoff_apm4330_bumper'] * mover_count,
            },
            'mass': mover_mass - bumper_mass,
            'bumper_mass': bumper_mass,
        }
        collision_params = {'shape': 'box', 'size': np.array([[0.155 / 2, 0.155 / 2]] * mover_count) + 0.001}
        self.initial_mover_z_pos = 0.001

        # MagBot (both movers carry a mover adapter, the movers are attached to the connections at runtime)
        self.num_magbots = 1
        # hidden collision proxies of the rail mounts and the station pins
        self.station_proxies = StationCollisionProxies(tol=station_proxy_tol)
        self.magbots = SixDPlatformMagBotsAPM4330(
            num_magbots=self.num_magbots, num_movers=mover_count, station_proxies=self.station_proxies
        )

        # initial mover positions (movers free on the table); None: sampled in reset_scene()
        assert initial_mover_xy_pos is None or initial_mover_xy_pos.shape == (mover_count, 2), (
            f'initial_mover_xy_pos must have the shape ({mover_count}, 2)'
        )
        # fixed start positions (None: the start positions are sampled), e.g. reused at each reset
        self.fixed_mover_start_xy_pos = None if initial_mover_xy_pos is None else initial_mover_xy_pos.copy()

        # station (MagBot 0 is generated in station 0)
        self.num_stations = self.STATION_POSES.shape[0]
        self.stations = [
            PlatformStation(pos=self.STATION_POSES[idx_station, :3], yaw=self.STATION_POSES[idx_station, 3], name=f'station_{idx_station}')
            for idx_station in range(0, self.num_stations)
        ]
        self.station_switch_tol = station_switch_tol

        # mover set poses at the stations (the equality constraints of the rail mounts are only switched at these positions)
        # shape: (num_stations, num_connections, 6), [:, idx_c, :]: pose of the mover docked at connection idx_c
        self.station_mover_poses = np.array([station.get_mover_poses(z_m=self.initial_mover_z_pos) for station in self.stations])

        # product (counters are reset in reset_scene())
        self.product_joint_name = 'product_joint'
        self.num_delivered_products = 0
        self.num_dropped_products = 0

        # init BasicMagBotEnv
        super().__init__(
            layout_tiles=np.ones((4, 3)),
            num_movers=mover_count,
            mover_params=mover_params,
            initial_mover_zpos=self.initial_mover_z_pos,
            table_height=0.2,
            collision_params=collision_params,
            initial_mover_start_xy_pos=initial_mover_xy_pos,
            custom_model_xml_strings=None,
            render_mode=render_mode,
            use_mj_passive_viewer=True,
            use_profile_frame=True,
        )

        # impedance controllers (all DoFs, i.e. actuators for all DoFs are generated)
        # the rotational damping is computed from the rotational inertia of the movers, which keeps the controller stable regardless of
        # whether a mover carries the MagBot or not, i.e. the same rotational stiffness can be used in both cases
        self.impedance_controllers = [
            MoverImpedanceController(
                model=self.model,
                mover_joint_name=self.mover_joint_names[mover_idx],
                mover_half_height=self.mover_size[mover_idx, 2],
                joint_mask=self.JOINT_MASK.copy(),
                translational_stiffness=np.array([5000.0, 5000.0, 10.0]),
                rotational_stiffness=np.array([5.0, 5.0, 5.0]),
            )
            for mover_idx in range(self.num_movers)
        ]

        # reload model to add MagBot, scene and actuators (includes reset_scene(), i.e. the start positions are sampled if not specified)
        self.reload_model(mover_start_xy_pos=initial_mover_xy_pos)
        self.initial_mover_xy_pos = self.mover_start_xy_pos.copy()
        # save the full model XML next to this script
        self.save_model_xml()

    ###################################################
    # Attachments                                     #
    ###################################################

    @property
    def connection_mover_indices(self) -> np.ndarray:
        """The index of the mover attached to each connection (numpy array of shape (num_magbots, num_connections), ``NO_INDEX``: no mover
        attached, i.e. the rail mount is parked in a station, see ``connections_parked``).
        """
        return self.magbots.connection_mover_indices

    @property
    def connections_parked(self) -> np.ndarray:
        """Whether the rail mount of a connection is parked in a station, i.e. not carried by a mover (numpy array of shape (num_magbots,
        num_connections), bool).
        """
        return self.connection_station_indices != NO_INDEX

    @property
    def mover_magbot_indices(self) -> np.ndarray:
        """The index of the MagBot to which each mover is attached (numpy array of shape (num_movers,), ``NO_INDEX``: mover not attached,
        see ``MagBotMoverCoupling.get_mover_attachments()``).
        """
        return np.array([NO_INDEX if attachment is None else attachment[0] for attachment in self.magbots.get_mover_attachments()])

    @property
    def mover_connection_indices(self) -> np.ndarray:
        """The index of the connection to which each mover is attached (numpy array of shape (num_movers,), ``NO_INDEX``: mover not
        attached, see ``MagBotMoverCoupling.get_mover_attachments()``).
        """
        return np.array([NO_INDEX if attachment is None else attachment[1] for attachment in self.magbots.get_mover_attachments()])

    def _attach_mover(self, idx_mover: int, idx_magbot: int, idx_c: int) -> None:
        """Pick up a parked rail mount, i.e. weld the rail mount to the mover instead of the station.

        :param idx_mover: the index of the mover
        :param idx_magbot: the index of the MagBot
        :param idx_c: the index of the connection (the rail mount must be parked)
        """
        assert self.connections_parked[idx_magbot, idx_c]
        idx_station = self.connection_station_indices[idx_magbot, idx_c]
        self.data.eq_active[self.eq_ids_station_weld[idx_magbot, idx_c, idx_station]] = False
        self.magbots.attach_mover(data=self.data, idx_mover=idx_mover, idx_magbot=idx_magbot, idx_c=idx_c)
        self.connection_station_indices[idx_magbot, idx_c] = NO_INDEX

    def _detach_mover(self, idx_magbot: int, idx_c: int, idx_station: int) -> None:
        """Put back a carried rail mount, i.e. weld the rail mount to the station (in its current pose) instead of the mover.

        :param idx_magbot: the index of the MagBot
        :param idx_c: the index of the connection (the rail mount must be carried by a mover)
        :param idx_station: the index of the station
        """
        self._set_station_weld_relpose(idx_magbot=idx_magbot, idx_c=idx_c, idx_station=idx_station)
        self.data.eq_active[self.eq_ids_station_weld[idx_magbot, idx_c, idx_station]] = True
        self.magbots.detach_mover(data=self.data, idx_magbot=idx_magbot, idx_c=idx_c)
        self.connection_station_indices[idx_magbot, idx_c] = idx_station

    ###################################################
    # Model                                           #
    ###################################################

    def _generate_magbot_xml_strings(self, custom_model_xml_strings: dict[str, str]) -> dict[str, str]:
        """Generate the custom model XML strings for the MagBots in the stations (MagBot i in station i) and add them to the
        ``custom_model_xml_strings``-dict. The pose of a MagBot is deduced from the station mover poses.

        :param custom_model_xml_strings: the current ``custom_model_xml_strings``-dict which is modified by this method
        :return: the modified ``custom_model_xml_strings``-dict
        """
        mover_qpos = self.get_mover_qpos(mover_names=self.mover_names, add_noise=False)
        # (x, y, z) of the MagBot bases; the z-pos is the z-pos of the movers, the (x, y) positions are deduced from the stations
        station_pos_xyz = np.tile(mover_qpos[0, :3], (self.num_magbots, 1))
        station_mover_xy_pos = self.station_mover_poses[: self.num_magbots, :, :2]  # (num_magbots, num_connections, 2)
        # the MagBot base is located at rail mount 0 (connection 0, mover b) and the platform c-rotation is given by the direction from
        # mover a (connection 1) to mover b (connection 0)
        station_pos_xyz[:, :2] = station_mover_xy_pos[:, 0]
        station_ab_xy_dirs = station_mover_xy_pos[:, 0] - station_mover_xy_pos[:, 1]
        station_platform_yaws = np.arctan2(station_ab_xy_dirs[:, 1], station_ab_xy_dirs[:, 0])
        custom_model_xml_strings = self.magbots.generate_magbot_xml_strings(
            initial_pos_xyz_mover_b=station_pos_xyz,
            custom_model_xml_strings=custom_model_xml_strings,
            initial_yaw=station_platform_yaws,
        )
        custom_model_xml_strings['custom_option_xml_str'] = self.OPTION_XML_STR
        return custom_model_xml_strings

    @staticmethod
    def _pose_attr_str(pose: np.ndarray) -> str:
        """Return the XML attributes of a body pose.

        :param pose: the pose (x, y, z, yaw) [m, rad]
        :return: the ``pos`` and ``euler`` attributes
        """
        return f'pos="{pose[0]:.5f} {pose[1]:.5f} {pose[2]:.5f}" euler="0 0 {pose[3]:.7f}"'

    def _generate_transport_xml_strings(self) -> tuple[str, str]:
        """Generate the XML strings of the supply rack, the product and the bin (see ``SixDPlatformMagBotApplicationExampleEnv``).

        :return: the XML string of the assets and the XML string of the bodies
        """
        mesh_dir = './misc/6D_platform_application_example'
        assets = '\n'.join(
            [
                '\n\t\t<!-- assets transport task -->',
                f'\t\t<mesh name="supply_rack_left_mesh" file="{mesh_dir}/supply_rack_left.stl" scale="1 1 1"/>',
                f'\t\t<mesh name="supply_rack_right_mesh" file="{mesh_dir}/supply_rack_right.stl" scale="1 1 1"/>',
                f'\t\t<mesh name="spacer_mesh" file="{mesh_dir}/spacer.stl" scale="1 1 1"/>',
                f'\t\t<mesh name="container_front_mesh" file="{mesh_dir}/container_front.stl" scale="1 1 1"/>',
                f'\t\t<mesh name="container_side_mesh" file="{mesh_dir}/container_side.stl" scale="1 1 1"/>',
                f'\t\t<mesh name="container_back_mesh" file="{mesh_dir}/container_back.stl" scale="1 1 1"/>',
            ]
        )
        rack_pose = self._pose_attr_str(self.SUPPLY_RACK_POSE)
        bodies = '\n'.join(
            [
                '\n\t\t<!-- supply rack -->',
                f'\t\t<body name="supply_rack" {rack_pose} gravcomp="1">',
                '\t\t\t<geom name="spacer_geom" type="mesh" mesh="spacer_mesh" mass="0.5" pos="0 0 0" material="black" contype="0" '
                'conaffinity="0"/>',
                '\t\t\t<geom name="supply_rack_left_geom" type="mesh" mesh="supply_rack_left_mesh" mass="1.0" pos="0 0 0.0835" '
                'material="black" contype="0" conaffinity="0"/>',
                '\t\t\t<geom name="supply_rack_right_geom" type="mesh" mesh="supply_rack_right_mesh" mass="1.0" pos="0.11 0.0 0.0835" '
                'euler="0 0 1.5708" material="black" contype="0" conaffinity="0"/>',
                '\t\t</body>',
                f'\t\t<body name="holding_rack" {rack_pose} gravcomp="1">',
                '\t\t\t<geom name="left_pole" type="box" size="0.0025 0.03775 0.002" mass="0.2" pos="0.0025 -0.06175 0.2054" '
                'material="black"/>',
                '\t\t\t<geom name="right_pole" type="box" size="0.0025 0.03775 0.002" mass="0.2" pos="0.1075 -0.06175 0.2054" '
                'material="black"/>',
                '\t\t\t<geom name="left_front_pole" type="box" size="0.0025 0.002 0.007" mass="0.2" pos="0.0025 -0.1014 0.2105" '
                'material="black"/>',
                '\t\t\t<geom name="right_front_pole" type="box" size="0.0025 0.002 0.007" mass="0.2" pos="0.1075 -0.1014 0.2105" '
                'material="black"/>',
                '\t\t</body>',
                '\n\t\t<!-- bin -->',
                f'\t\t<body name="product_container" {self._pose_attr_str(self.BIN_POSE)} gravcomp="1">',
                '\t\t\t<geom name="container_front" type="mesh" mesh="container_front_mesh" mass="0.5" pos="0.0 0.0 0.0" '
                'material="black"/>',
                '\t\t\t<geom name="container_back" type="mesh" mesh="container_back_mesh" mass="0.5" pos="0.0 -0.153 0.0" '
                'material="black"/>',
                '\t\t\t<geom name="container_side_left" type="mesh" mesh="container_side_mesh" mass="0.5" pos="0.0 -0.15 0.0" '
                'material="black"/>',
                '\t\t\t<geom name="container_side_right" type="mesh" mesh="container_side_mesh" mass="0.5" pos="0.197 -0.15 0.0" '
                'material="black"/>',
                '\t\t\t<geom name="container_bottom" type="box" size="0.1 0.079 0.0025" mass="0.5" pos="0.1 -0.074 -0.002" '
                'material="black"/>',
                '\t\t</body>',
                '\n\t\t<!-- product -->',
                f'\t\t<body name="product_body" pos="{" ".join(f"{v:.5f}" for v in self.PRODUCT_SUPPLY_POS)}" gravcomp="0">',
                '\t\t\t<geom name="product_geom_0" type="box" size="0.0495 0.03205 0.01" mass="0.06" pos="0 0 0" euler="0 0 0" '
                'material="white" solref="0.005 1" priority="1" friction="0.24 0.0001 0.00001" condim="6"/>',
                '\t\t\t<geom name="product_geom_1" type="box" size="0.055 0.0375 0.0015" mass="0.004" pos="0 0 0.0115" euler="0 0 0" '
                'material="white"/>',
                f'\t\t\t<joint name="{self.product_joint_name}" type="free"/>',
                '\t\t</body>',
            ]
        )
        return assets, bodies

    def _custom_xml_string_callback(self, custom_model_xml_strings: dict[str, str] | None = None) -> dict[str, str]:
        """Add the MagBot, the station, the supply rack, the product, the bin and the actuators for the movers to the MuJoCo model by
        modifying the ``custom_model_xml_strings``-dict.

        :param custom_model_xml_strings: the current ``custom_model_xml_strings``-dict which is modified by this callback, defaults to None
        :return: the modified ``custom_model_xml_strings``-dict
        """
        if custom_model_xml_strings is None:
            custom_model_xml_strings = {}
            custom_model_xml_strings.update({'custom_mover_body_xml_str_list': [None] * self.num_movers})

        if hasattr(self, 'mover_joint_names'):
            # MagBots in the stations (MagBot i in station i)
            custom_model_xml_strings = self._generate_magbot_xml_strings(custom_model_xml_strings=custom_model_xml_strings)

            # add actuators
            actuator_lines = ['\n\n\t<actuator>']
            for idx_mover in range(0, self.num_movers):
                actuator_lines.append(self.impedance_controllers[idx_mover].generate_actuator_xml_string(idx_mover=idx_mover))
            # MagBot platform a,b rot actuators
            actuator_lines.append(self.magbots.generate_platform_abRot_actuator_xml_strings())
            # join
            actuator_lines.append('\n\t</actuator>')
            # station welds of the rail mounts (inactive at the beginning, switched by _update_equality_constraints())
            station_equality_lines = ['\n\t<equality>']
            for station in self.stations:
                for idx_magbot in range(0, self.num_magbots):
                    station_equality_lines.append(station.generate_equality_xml_str(idx_magbot=idx_magbot))
            station_equality_lines.append('\t</equality>')
            custom_outworldbody_xml_str = custom_model_xml_strings.get('custom_outworldbody_xml_str', None)
            custom_model_xml_strings['custom_outworldbody_xml_str'] = (
                custom_outworldbody_xml_str + ''.join(actuator_lines) + '\n'.join(station_equality_lines)
            )

            # supply rack, product and bin
            transport_assets_xml_str, transport_bodies_xml_str = self._generate_transport_xml_strings()

            # assets application
            custom_assets_xml_str = custom_model_xml_strings['custom_assets_xml_str']
            application_assets_list = [
                '\n\t\t<!-- assets reconfiguration transport benchmark -->',
                # stations (all stations have the same type)
                type(self.stations[0]).generate_assets_xml_str(),
                transport_assets_xml_str,
            ]
            application_assets_xml_str = '\n'.join(application_assets_list)
            custom_assets_xml_str += application_assets_xml_str
            custom_model_xml_strings['custom_assets_xml_str'] = custom_assets_xml_str

            # z-pos of the rail mount origin of a parked MagBot (= z-pos of the mover adapter: mover z-pos + mover half height + 0.006)
            rail_mount_z = self.initial_mover_z_pos + float(np.max(self.mover_size[:, 2])) + 0.006

            custom_worldbody_xml_str = custom_model_xml_strings['custom_worldbody_xml_str']
            application_bodies_list = [
                # stations
                *[station.generate_body_xml_str(proxies=self.station_proxies, rail_mount_z=rail_mount_z) for station in self.stations],
                transport_bodies_xml_str,
            ]
            application_bodies_xml_str = '\n'.join(application_bodies_list)
            custom_worldbody_xml_str += application_bodies_xml_str
            custom_model_xml_strings['custom_worldbody_xml_str'] = custom_worldbody_xml_str

        return custom_model_xml_strings

    def reload_model(self, mover_start_xy_pos: np.ndarray | None = None) -> None:
        """Generate and compile a new model XML string and reset the scene (see ``reset_scene()``).

        :param mover_start_xy_pos: None or a numpy array of shape (num_movers,2) containing the (x,y) starting positions of each mover.
            If None, the start positions are sampled in ``reset_scene()``.
        """
        # generate a new model XML string
        custom_model_xml_strings = self._custom_xml_string_callback(custom_model_xml_strings=self.custom_model_xml_strings_before_cb)
        model_xml_str = self.generate_model_xml_string(
            mover_start_xy_pos=mover_start_xy_pos, mover_goal_xy_pos=None, custom_xml_strings=custom_model_xml_strings
        )
        # the full model XML of the current model (see save_model_xml())
        self.model_xml_str = model_xml_str
        # compile the MuJoCo model
        self.model = mujoco.MjModel.from_xml_string(model_xml_str)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # update cached mujoco data
        self.update_cached_mover_mujoco_data()
        for idx_mover in range(0, self.num_movers):
            self.impedance_controllers[idx_mover].update_cached_mujoco_data(self.model)
        self.magbots.update_cached_mujoco_data(self.model)
        self._update_cached_equality_ids()
        # IDs of all station collision proxies (rail mount proxies, rail rods and station pins) and of the rail rods and station pins
        proxy_geoms = {
            idx_geom: name
            for idx_geom in range(0, self.model.ngeom)
            if (name := mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, idx_geom) or '').startswith(
                StationCollisionProxies.PROXY_NAME_PREFIX
            )
        }
        self.geom_ids_proxy = np.array(list(proxy_geoms.keys()), dtype=int)
        self.geom_ids_rail_rod = np.array([idx_geom for idx_geom, name in proxy_geoms.items() if '_rail_rod' in name], dtype=int)
        self.geom_ids_station_pin = np.array([idx_geom for idx_geom, name in proxy_geoms.items() if '_pins_' in name], dtype=int)
        # qpos addresses and joint IDs of the slide joints of the MagBot bases (used to shift the MagBots in their stations)
        self.jnt_ids_magbot_xy = np.array(
            [
                [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) for joint_name in joint_names_xy]
                for joint_names_xy in self.magbots.base_xy_joint_names
            ],
            dtype=int,
        )
        self.qpos_adrs_magbot_xy = self.model.jnt_qposadr[self.jnt_ids_magbot_xy]
        # product joint (the initial qpos of the model is the qpos of the product in the supply rack)
        product_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.product_joint_name)
        self.product_qpos_adr = int(self.model.jnt_qposadr[product_joint_id])
        self.product_qvel_adr = int(self.model.jnt_dofadr[product_joint_id])
        self.product_supply_qpos = self.model.qpos0[self.product_qpos_adr : self.product_qpos_adr + 7].copy()

        if self.render_mode is not None:
            self.viewer_collection.reload_model(self.model, self.data)

        self.reset_scene(mover_start_xy_pos=mover_start_xy_pos)

    ###################################################
    # Scene                                           #
    ###################################################

    @property
    def model_xml_path(self) -> Path:
        """The path of the file to which the full model XML is saved (see ``save_model_xml()``): next to this script, named
        ``reconfiguration_transport_env.xml``.
        """
        return Path(__file__).resolve().with_name('reconfiguration_transport_env.xml')

    def save_model_xml(self) -> None:
        """Save the full model XML of the current model (``model_xml_str``) to ``model_xml_path``, e.g. to inspect the generated model
        or to load it in the MuJoCo viewer (the paths of the assets are absolute). An existing file is overwritten.
        """
        self.model_xml_path.write_text(self.model_xml_str)

    def reset_scene(
        self,
        mover_start_xy_pos: np.ndarray | None = None,
        magbot_station_x_offsets: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Reset the scene without reloading the model: MagBot i is parked in station i (optionally shifted along the station x-axis),
        all movers are free on the table (not attached to any connection) and the product is in the supply rack. The MuJoCo data, the
        welds of the rail mounts, the MagBot controller states and the product counters are reset.

        :param mover_start_xy_pos: the (x,y) start positions of the movers (numpy array of shape (num_movers,2)) [m], defaults to None.
            If None, the start positions are sampled on the free space of the table (see ``sample_mover_start_xy_pos()``).
        :param magbot_station_x_offsets: the offsets of the MagBots along the x-axis of their stations (numpy array of shape
            (num_magbots,)) [m], defaults to None. If None, the offsets are sampled if ``randomize_magbot_station_pos`` is True (see
            ``sample_magbot_station_x_offsets()``), otherwise they are zero.
        :param rng: the RNG used for sampling, defaults to None, i.e. ``self.rng``
        """
        rng = self.rng if rng is None else rng

        mujoco.mj_resetData(self.model, self.data)
        # no mover is attached (resets the MagBot controller states)
        self.magbots.reset_mover_coupling(data=self.data)
        self.last_mover_pos_d = None

        # shift the MagBots in their stations
        if magbot_station_x_offsets is None:
            if self.randomize_magbot_station_pos:
                magbot_station_x_offsets = self.sample_magbot_station_x_offsets(rng=rng)
            else:
                magbot_station_x_offsets = np.zeros(self.num_magbots)
        assert magbot_station_x_offsets.shape == (self.num_magbots,)
        self.magbot_station_x_offsets = magbot_station_x_offsets.copy()
        self._shift_magbots(x_offsets=self.magbot_station_x_offsets)
        mujoco.mj_forward(self.model, self.data)

        # the MagBots are parked in their stations (MagBot i in station i), i.e. all rail mounts are welded to the stations
        # [idx_magbot, idx_c]: index of the station to which the rail mount is welded (NO_INDEX: rail mount carried by a mover, see
        # connections_parked)
        self.connection_station_indices = np.repeat(np.arange(0, self.num_magbots)[:, np.newaxis], self.num_connections, axis=1)
        for idx_magbot in range(0, self.num_magbots):
            for idx_c in range(0, self.num_connections):
                idx_station = self.connection_station_indices[idx_magbot, idx_c]
                self._set_station_weld_relpose(idx_magbot=idx_magbot, idx_c=idx_c, idx_station=idx_station)
                self.data.eq_active[self.eq_ids_station_weld[idx_magbot, idx_c, idx_station]] = True

        # mover start positions (movers free on the table)
        if mover_start_xy_pos is None:
            mover_start_xy_pos = self.sample_mover_start_xy_pos(rng=rng)
        else:
            assert mover_start_xy_pos.shape == (self.num_movers, 2)
            assert self._mover_start_xy_pos_is_valid(mover_start_xy_pos), 'invalid mover start positions (collision)'
        self.mover_start_xy_pos = mover_start_xy_pos.copy()
        self.data.qpos[self.mover_qpos_indices] = self._mover_start_qpos(mover_start_xy_pos)
        self.data.qvel[self.mover_qvel_qacc_indices] = 0.0

        # product in the supply rack
        self.num_delivered_products = 0
        self.num_dropped_products = 0
        self.respawn_product()

        self.render()

    def _shift_magbots(self, x_offsets: np.ndarray) -> None:
        """Shift each MagBot rigidly along the x-axis of its station (MagBot i is parked in station i) by setting the slide joints of
        the MagBot base. The rail mounts are not welded yet, i.e. this method must be called before the station welds are set.

        :param x_offsets: the offsets along the x-axis of the stations (numpy array of shape (num_magbots,)) [m]
        """
        if not np.any(x_offsets):
            return
        # world axes of the slide joints of the MagBot bases
        mujoco.mj_kinematics(self.model, self.data)
        for idx_magbot in range(0, self.num_magbots):
            shift_xy = x_offsets[idx_magbot] * self.stations[idx_magbot].rotation_matrix_xy()[:, 0]
            joint_axes_xy = self.data.xaxis[self.jnt_ids_magbot_xy[idx_magbot], :2].T  # columns: axes of the x and y slide joints
            self.data.qpos[self.qpos_adrs_magbot_xy[idx_magbot]] += np.linalg.solve(joint_axes_xy, shift_xy)

    def sample_magbot_station_x_offsets(self, rng: np.random.Generator) -> np.ndarray:
        """Sample the offsets of the MagBots along the x-axis of their stations uniformly in
        [-magbot_station_randomization_x_range, magbot_station_randomization_x_range].

        :param rng: the RNG used for sampling
        :return: the offsets (numpy array of shape (num_magbots,)) [m]
        """
        r = self.magbot_station_randomization_x_range
        return rng.uniform(low=-r, high=r, size=(self.num_magbots,))

    def _mover_start_qpos(self, mover_xy_pos: np.ndarray) -> np.ndarray:
        """Return the qpos of the movers at the given (x,y) positions (z-pos: ``initial_mover_z_pos``, no rotation).

        :param mover_xy_pos: the (x,y) positions of the movers (numpy array of shape (num_qpos,2)) [m]
        :return: the qpos (numpy array of shape (num_qpos,7))
        """
        qpos = np.zeros((mover_xy_pos.shape[0], 7))
        qpos[:, :2] = mover_xy_pos
        qpos[:, 2] = self.initial_mover_z_pos + self.mover_size[: mover_xy_pos.shape[0], 2]  # z-pos of the mover center
        qpos[:, 3] = 1.0
        return qpos

    def _get_rail_mount_keepouts(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the keep-out boxes of the movers around the parked rail mounts (rail mounts welded to a station). The boxes are
        centered at the real rail mount positions, aligned with the station mover poses and have the size of the mover collision shape.

        :return: the qpos (numpy array of shape (num_parked,7)) and the half sizes (numpy array of shape (num_parked,2)) of the boxes
        """
        indices_magbot, indices_c = np.nonzero(self.connections_parked)
        yaws = self.station_mover_poses[self.connection_station_indices[indices_magbot, indices_c], indices_c, 5]
        qpos = np.zeros((len(indices_c), 7))
        qpos[:, :2] = self.get_rail_mount_xy().reshape((self.num_magbots, self.num_connections, 2))[indices_magbot, indices_c]
        qpos[:, 3] = np.cos(0.5 * yaws)
        qpos[:, 6] = np.sin(0.5 * yaws)
        sizes = np.repeat(self.get_c_size_arr(c_size=self.c_size, num_reps=self.num_movers)[:1], len(indices_c), axis=0)
        return qpos, sizes

    def _mover_start_xy_pos_is_valid(self, mover_xy_pos: np.ndarray) -> bool:
        """Check whether the start positions of the first ``mover_xy_pos.shape[0]`` movers are collision-free, i.e. without wall
        collisions, mover collisions and collisions with the parked rail mounts (see ``_get_rail_mount_keepouts()``).

        :param mover_xy_pos: the (x,y) positions of the first movers (numpy array of shape (num_qpos,2)) [m]
        :return: True if the positions are valid, False otherwise
        """
        num_qpos = mover_xy_pos.shape[0]
        qpos = self._mover_start_qpos(mover_xy_pos)
        c_size_arr = self.get_c_size_arr(c_size=self.c_size, num_reps=self.num_movers)[:num_qpos] + self.c_size_offset
        # walls
        if not self.qpos_is_valid(qpos=qpos, c_size=c_size_arr, add_safety_offset=False).all():
            return False
        # movers
        if num_qpos > 1 and self.check_mover_collision(
            mover_names=self.mover_names[:num_qpos], c_size=c_size_arr, add_safety_offset=False, mover_qpos=qpos
        ):
            return False
        # parked rail mounts
        keepout_qpos, keepout_sizes = self._get_rail_mount_keepouts()
        if len(keepout_qpos) > 0:
            idx_q, idx_k = np.meshgrid(np.arange(num_qpos), np.arange(len(keepout_qpos)), indexing='ij')
            idx_q, idx_k = idx_q.flatten(), idx_k.flatten()
            if geometry_2D_utils.check_rectangles_intersect(
                qpos_r1=qpos[idx_q], qpos_r2=keepout_qpos[idx_k], size_r1=c_size_arr[idx_q], size_r2=keepout_sizes[idx_k]
            ).any():
                return False
        return True

    def sample_mover_start_xy_pos(self, rng: np.random.Generator, max_trials: int = 1000) -> np.ndarray:
        """Sample collision-free start positions of the movers uniformly on the table (rejection sampling, one mover after the other).
        The positions are collision-free w.r.t. the walls, the other movers and the parked rail mounts (see
        ``_get_rail_mount_keepouts()``), i.e. this method must be called after the rail mounts are welded to the stations.

        :param rng: the RNG used for sampling
        :param max_trials: the maximum number of trials per mover, defaults to 1000
        :return: the (x,y) start positions of the movers (numpy array of shape (num_movers,2)) [m]
        """
        xy_min = np.array([self.x_pos_tiles.min() - self.tile_size[0], self.y_pos_tiles.min() - self.tile_size[1]])
        xy_max = np.array([self.x_pos_tiles.max() + self.tile_size[0], self.y_pos_tiles.max() + self.tile_size[1]])
        mover_xy_pos = np.zeros((self.num_movers, 2))
        for idx_mover in range(0, self.num_movers):
            for _ in range(0, max_trials):
                mover_xy_pos[idx_mover] = rng.uniform(low=xy_min, high=xy_max)
                if self._mover_start_xy_pos_is_valid(mover_xy_pos[: idx_mover + 1]):
                    break
            else:
                raise RuntimeError(f'No collision-free start position found for mover {idx_mover} within {max_trials} trials.')
        return mover_xy_pos

    def get_rail_mount_xy(self) -> np.ndarray:
        """Return the real (x,y) positions of the rail mounts.

        :return: the positions in connection order (numpy array of shape (num_magbots*num_connections,2)),
            [idx_magbot*num_connections+idx_c,:]: rail mount of connection idx_c of MagBot idx_magbot [m]
        """
        return self.data.xpos[self.body_ids_rail_mount.flatten(), :2].copy()

    ###################################################
    # Product                                         #
    ###################################################

    def get_product_qpos(self) -> np.ndarray:
        """Return the current qpos of the product.

        :return: the position (x, y, z) [m] and the rotation (quaternion (w, x, y, z)) of the product (numpy array of shape (7,))
        """
        return self.data.qpos[self.product_qpos_adr : self.product_qpos_adr + 7].copy()

    def get_product_speed(self) -> float:
        """Return the current linear speed of the product.

        :return: the linear speed [m/s]
        """
        return float(np.linalg.norm(self.data.qvel[self.product_qvel_adr : self.product_qvel_adr + 3]))

    def respawn_product(self) -> None:
        """Put the product back into the supply rack (at rest)."""
        self.data.qpos[self.product_qpos_adr : self.product_qpos_adr + 7] = self.product_supply_qpos
        self.data.qvel[self.product_qvel_adr : self.product_qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def is_product_in_bin(self) -> bool:
        """Check whether the product is inside the bin, i.e. its position is inside the inner box of the bin (below the rim).

        :return: True if the product is inside the bin, False otherwise
        """
        product_pos = self.data.qpos[self.product_qpos_adr : self.product_qpos_adr + 3]
        c, s = np.cos(self.BIN_POSE[3]), np.sin(self.BIN_POSE[3])
        delta_xy = product_pos[:2] - self.BIN_POSE[:2]
        rel_x, rel_y = c * delta_xy[0] + s * delta_xy[1], -s * delta_xy[0] + c * delta_xy[1]  # position in the bin body frame
        return bool(
            self.BIN_INNER_X_RANGE[0] <= rel_x <= self.BIN_INNER_X_RANGE[1]
            and self.BIN_INNER_Y_RANGE[0] <= rel_y <= self.BIN_INNER_Y_RANGE[1]
            and product_pos[2] <= self.BIN_POSE[2] + self.BIN_RIM_Z
        )

    def is_product_dropped(self) -> bool:
        """Check whether the product is dropped, i.e. it is outside the bin below ``PRODUCT_DROP_Z`` (e.g. on the table, on the frame or
        on the floor).

        :return: True if the product is dropped, False otherwise
        """
        return bool(self.data.qpos[self.product_qpos_adr + 2] < self.PRODUCT_DROP_Z and not self.is_product_in_bin())

    def _update_product(self) -> tuple[bool, bool]:
        """Count and respawn the product if it came to rest in the bin (delivered) or outside the bin below the platform heights
        (dropped).

        :return: whether the product was delivered and whether it was dropped (each bool)
        """
        if self.get_product_speed() >= self.PRODUCT_REST_SPEED:
            return False, False
        delivered, dropped = self.is_product_in_bin(), self.is_product_dropped()
        if delivered:
            self.num_delivered_products += 1
        if dropped:
            self.num_dropped_products += 1
        if delivered or dropped:
            self.respawn_product()
        return delivered, dropped

    ###################################################
    # Collisions and info                             #
    ###################################################

    def get_station_proxy_collisions(self) -> dict[str, bool]:
        """Check the current contacts of the station collision proxies (see ``StationCollisionProxies``). If a station is entered
        correctly, the proxies are not in contact with anything.

        :return: a dictionary with the following keys (bool values):

            - ``station_proxy_collision``: whether any collision proxy (rail mount proxy, rail rod or station pin) is in contact with
              any other geom (includes the two collision types below)
            - ``rail_rod_collision``: whether a rail rod is in contact with any other geom, e.g. because an empty mover entered or
              left a station with a sideways misalignment larger than the station proxy tolerance (the rail rod hits the jaws of a
              rail mount)
            - ``station_pin_collision``: whether a station pin is in contact with any other geom, e.g. because a MagBot carried by
              movers was moved into or out of a station with a sideways misalignment larger than the station proxy tolerance (the
              rail mount hits the station pins)
        """
        contact_geoms = self.data.contact.geom[: self.data.ncon]
        return {
            'station_proxy_collision': bool(np.isin(contact_geoms, self.geom_ids_proxy).any()),
            'rail_rod_collision': bool(np.isin(contact_geoms, self.geom_ids_rail_rod).any()),
            'station_pin_collision': bool(np.isin(contact_geoms, self.geom_ids_station_pin).any()),
        }

    def check_station_proxy_collision(self) -> bool:
        """Check whether a station collision proxy (rail mount proxy, rail rod or station pin) is in contact with any other
        geom, e.g. because a mover or a MagBot entered a station with a sideways misalignment larger than the station proxy
        tolerance. If the station is entered correctly, the proxies are not in contact with anything.

        :return: True if a collision proxy is in contact with any other geom, False otherwise
        """
        return self.get_station_proxy_collisions()['station_proxy_collision']

    def check_rotation_violation(self, add_qpos_noise: bool = True) -> bool:
        """Check whether any mover violates the rotation limits of the MagLev system (see ``BasicMagBotEnv.check_mover_rotation()``):
        a full rotation around the z-axis is only possible within ``rotation_tol`` of one of the default rotation points
        (``rotation_points``, see ``get_tile_rotation_points()``). At all other positions, the yaw of a mover may deviate at most
        ``max_yaw_deviation`` from the nearest multiple of 90 deg.

        :param add_qpos_noise: whether to add Gaussian noise to the qpos of the movers, defaults to True, since it would also occur in a
            real system
        :return: True if at least one mover violates the rotation limits, False otherwise
        """
        return bool(
            self.check_mover_rotation(
                mover_names=self.mover_names,
                mover_qpos=None,
                add_qpos_noise=add_qpos_noise,
                tol=self.rotation_tol,
                max_yaw_deviation=self.max_yaw_deviation,
            ).any()
        )

    def _check_for_other_collisions_callback(self) -> tuple[bool, dict[str, bool]]:
        """Check for collisions besides mover and wall collisions, i.e. contacts of the station collision proxies (see
        ``get_station_proxy_collisions()``) and rotation violations (see ``check_rotation_violation()``), which are treated like
        collisions. This method has the signature of the corresponding callback of ``BasicMagBotSingleAgentEnv``.

        :return:
            - whether there is a station proxy collision or a rotation violation (bool)
            - a dictionary with the station proxy collision types (see ``get_station_proxy_collisions()``) and ``rotation_violation``
        """
        collision_info = self.get_station_proxy_collisions()
        collision_info['rotation_violation'] = self.check_rotation_violation()
        return collision_info['station_proxy_collision'] or collision_info['rotation_violation'], collision_info

    def _check_collisions(self) -> tuple[bool, bool, bool, dict[str, bool]]:
        """Check for mover, wall and other collisions (station collision proxies and rotation violations) in the current state. Noise
        is added to the mover positions, since it would also occur in a real system.

        :return: whether there is a mover collision, a wall collision or another collision (each bool), and a dictionary with the
            station proxy collision types and the rotation violation (see ``_check_for_other_collisions_callback()``)
        """
        wall_collision = self.check_wall_collision(
            mover_names=self.mover_names, c_size=self.c_size, add_safety_offset=False, mover_qpos=None, add_qpos_noise=True
        ).any()
        mover_collision = self.check_mover_collision(
            mover_names=self.mover_names, c_size=self.c_size, add_safety_offset=False, mover_qpos=None, add_qpos_noise=True
        )
        other_collision, collision_info = self._check_for_other_collisions_callback()
        return bool(mover_collision), bool(wall_collision), bool(other_collision), collision_info

    def _get_info(
        self,
        mover_collision: bool,
        wall_collision: bool,
        other_collision: bool,
        achieved_goal: np.ndarray | None = None,
        desired_goal: np.ndarray | None = None,
        collision_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a dictionary that contains the collision types. This method has the signature of the corresponding method of
        ``BasicMagBotSingleAgentEnv``.

        :param mover_collision: whether there is a collision between two movers
        :param wall_collision: whether there is a collision between a mover and a wall
        :param other_collision: whether there is a station proxy collision or a rotation violation (already contained in
            ``collision_info``)
        :param achieved_goal: not used in this environment, defaults to None
        :param desired_goal: not used in this environment, defaults to None
        :param collision_info: a dictionary with the station proxy collision types and the rotation violation (see
            ``_check_for_other_collisions_callback()``), defaults to None. If None, both are checked in the current state.
        :return: the info dictionary with keys ``mover_collision``, ``wall_collision``, ``station_proxy_collision``,
            ``rail_rod_collision``, ``station_pin_collision`` and ``rotation_violation`` (bool values)
        """
        if collision_info is None:
            collision_info = self._check_for_other_collisions_callback()[1]
        info = {'mover_collision': bool(mover_collision), 'wall_collision': bool(wall_collision)}
        info.update({key: bool(value) for key, value in collision_info.items()})
        return info

    ###################################################
    # Welds                                           #
    ###################################################

    def _update_cached_equality_ids(self) -> None:
        """Update the cached IDs of the station welds of the rail mounts and of the bodies connected by the station welds (the mover welds
        are cached by the MagBots).
        """
        # [idx_magbot, idx_c]: rail mount of connection idx_c
        self.body_ids_rail_mount = self.magbots.rail_mount_body_ids.copy()
        # [idx_magbot, idx_c, idx_station]: weld between the rail mount and the station
        self.eq_ids_station_weld = np.zeros((self.num_magbots, self.num_connections, self.num_stations), dtype=int)
        for idx_magbot in range(0, self.num_magbots):
            for idx_c in range(0, self.num_connections):
                for idx_station, station in enumerate(self.stations):
                    self.eq_ids_station_weld[idx_magbot, idx_c, idx_station] = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f'{station.name}_weld_c{idx_c}_{idx_magbot}'
                    )
        self.body_ids_station = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, station.name) for station in self.stations], dtype=int
        )

    def _set_station_weld_relpose(self, idx_magbot: int, idx_c: int, idx_station: int) -> None:
        """Set the relative pose of the station weld of a rail mount to the current relative pose between the rail mount and the
        station, i.e. the rail mount is held in its current pose when the weld is activated.

        :param idx_magbot: the index of the MagBot
        :param idx_c: the index of the connection whose rail mount is welded to the station
        :param idx_station: the index of the station to which the rail mount is welded
        """
        id_body1 = self.body_ids_rail_mount[idx_magbot, idx_c]
        id_body2 = self.body_ids_station[idx_station]
        # weld (anchor = 0): xpos1 + xmat1 @ relpos = xpos2 and xquat1 * relquat = xquat2
        quat1_inv = np.zeros(4)
        mujoco.mju_negQuat(quat1_inv, self.data.xquat[id_body1])
        relpos = np.zeros(3)
        mujoco.mju_rotVecQuat(relpos, self.data.xpos[id_body2] - self.data.xpos[id_body1], quat1_inv)
        relquat = np.zeros(4)
        mujoco.mju_mulQuat(relquat, quat1_inv, self.data.xquat[id_body2])

        eq_id = self.eq_ids_station_weld[idx_magbot, idx_c, idx_station]
        self.model.eq_data[eq_id, 0:3] = 0.0
        self.model.eq_data[eq_id, 3:6] = relpos
        self.model.eq_data[eq_id, 6:10] = relquat

    def _update_equality_constraints(self, mover_pos_d: np.ndarray) -> None:
        """Switch the welds of the rail mounts between the movers and the stations based on the real rail mount positions. The switch
        happens on the first step in which a mover leaves (i.e. the distance increases), so that the rail mount does not jump.

        - pick-up: a free mover (not attached to any connection) leaves a parked rail mount, i.e. the last mover set position is within
          the station switch tolerance of the real position of the nearest parked rail mount and the mover leaves along the take-out axis
          of the station. Additionally, the adapter of the mover must point in the direction of the rail mount, i.e. the mover yaw must be
          within +/-90 deg of the docking yaw (see ``MagBotMoverCoupling.get_docking_mover_yaw()``). Then, the rail mount is welded to the
          mover, i.e. any mover can pick up any rail mount. Smaller yaw errors are not checked, since such a mover collides with the
          station collision proxies (see ``StationCollisionProxies``) when entering the station. A mover rotated by pi, however, fits
          through the jaws of the rail mount, i.e. it is not attached and its rail rod hits the jaws when it leaves the station.
        - put-back: the rail mount carried by a mover is within the station switch tolerance of the nominal station position of its
          connection and the mover set position moves away from it. If the mover leaves along the entry axis of the station, the rail
          mount is welded to this station and the mover is free again.

        Entering a station never switches the welds, since the decision only depends on the direction in which the mover leaves.

        :param mover_pos_d: the current mover set positions (numpy array of shape (num_movers, >=2))
        """
        if self.last_mover_pos_d is not None:
            rail_mount_xy = self.get_rail_mount_xy().reshape((self.num_magbots, self.num_connections, 2))
            mover_attachments = self.magbots.get_mover_attachments()
            for idx_mover, attachment in enumerate(mover_attachments):
                last_xy, curr_xy = self.last_mover_pos_d[idx_mover, :2], mover_pos_d[idx_mover, :2]
                delta = curr_xy - last_xy
                if np.linalg.norm(delta) < 1e-6:  # ignore deltas close to numerical noise
                    continue

                if attachment is None:
                    # pick-up: the free mover leaves the nearest parked rail mount along the take-out axis -> weld to mover
                    indices_magbot_parked, indices_c_parked = np.nonzero(self.connections_parked)
                    if len(indices_c_parked) == 0:
                        continue
                    last_dists = np.linalg.norm(last_xy - rail_mount_xy[indices_magbot_parked, indices_c_parked], axis=1)
                    idx_nearest = int(np.argmin(last_dists))
                    idx_magbot, idx_c = indices_magbot_parked[idx_nearest], indices_c_parked[idx_nearest]
                    last_dist = last_dists[idx_nearest]
                    curr_dist = np.linalg.norm(curr_xy - rail_mount_xy[idx_magbot, idx_c])
                    is_leaving = last_dist < self.station_switch_tol and curr_dist > last_dist
                    idx_station = self.connection_station_indices[idx_magbot, idx_c]
                    if not is_leaving or self.stations[idx_station].is_along_entry_axis(delta_xy=delta):
                        continue
                    # the mover adapter must point in the direction of the rail mount (a mover rotated by pi fits through the jaws)
                    mover_quat = self.get_mover_qpos(mover_names=[self.mover_names[idx_mover]], add_noise=False)[0, 3:]
                    mover_yaw = rotations_utils.vec_quat2euler(mover_quat)[0, 2]
                    docking_yaw = self.magbots.get_docking_mover_yaw(
                        data=self.data, idx_mover=idx_mover, idx_magbot=idx_magbot, idx_c=idx_c
                    )
                    if abs(np.angle(np.exp(1j * (mover_yaw - docking_yaw)))) >= np.pi / 2:
                        continue
                    self._attach_mover(idx_mover=idx_mover, idx_magbot=idx_magbot, idx_c=idx_c)
                else:
                    # put-back: the carried rail mount is at the nominal station position of its connection and the mover leaves along
                    # the entry axis -> weld to station
                    idx_magbot, idx_c = attachment
                    station_xy = self.station_mover_poses[:, idx_c, :2]
                    rail_dists = np.linalg.norm(rail_mount_xy[idx_magbot, idx_c] - station_xy, axis=1)
                    idx_station = int(np.argmin(rail_dists))
                    last_dist = np.linalg.norm(last_xy - station_xy[idx_station])
                    curr_dist = np.linalg.norm(curr_xy - station_xy[idx_station])
                    is_leaving = rail_dists[idx_station] < self.station_switch_tol and curr_dist > last_dist
                    if not is_leaving or not self.stations[idx_station].is_along_entry_axis(delta_xy=delta):
                        continue
                    self._detach_mover(idx_magbot=idx_magbot, idx_c=idx_c, idx_station=idx_station)

        self.last_mover_pos_d = mover_pos_d.copy()

    ###################################################
    # Kinematics and control                          #
    ###################################################

    def get_mover_poses_from_magbot_setpose(self, magbot_setposes: np.ndarray) -> np.ndarray:
        """Return the mover set poses resulting from MagBot set poses (inverse kinematics of the MagBot), incl. the yaw offsets of the
        docked movers (see ``MOVER_YAW_OFFSETS`` of the MagBot). The poses are returned in connection order, i.e. the poses of the movers
        attached to the connections (see ``connection_mover_indices``) have to be set to these poses, e.g.
        ``mover_setposes[env.connection_mover_indices.flatten()] = poses``.

        :param magbot_setposes: the desired platform poses (numpy array of shape (num_magbots, 6) using Euler angles (xyz) in rad)
        :return: the mover set poses in connection order (numpy array of shape (num_magbots*num_connections, 6) using Euler angles (xyz)
            in rad)
        """
        # [a, b] -> connection order [b, a] (the platform set poses are copied, since they are clipped in-place)
        mover_poses = self.magbots.platformSetPose2MoverSetPose(
            platform_pose_d=magbot_setposes.copy(), mover_z_d=float(self.initial_mover_zpos), use_euler=True
        )[:, ::-1].copy()
        mover_poses[:, :, 5] += self.magbots.MOVER_YAW_OFFSETS[np.newaxis, :]
        return mover_poses.reshape((-1, 6))

    def get_magbot_setpose_from_mover_poses(self, mover_poses: np.ndarray) -> np.ndarray:
        """Return the MagBot set poses resulting from mover set poses (forward kinematics of the MagBot). The yaw offsets of the docked
        movers (see ``MOVER_YAW_OFFSETS`` of the MagBot) are removed.

        :param mover_poses: the mover set poses in connection order (numpy array of shape (num_magbots*num_connections, 6) using Euler
            angles (xyz) in rad)
        :return: the platform set poses (numpy array of shape (num_magbots, 6) using Euler angles (xyz) in rad)
        """
        mover_poses = mover_poses.reshape((self.num_magbots, self.num_connections, 6)).copy()
        mover_poses[:, :, 5] -= self.magbots.MOVER_YAW_OFFSETS[np.newaxis, :]
        # connection order [b, a] -> [a, b]
        return self.magbots.moverSetPose2PlatformSetPose(mover_poses=mover_poses[:, ::-1], use_euler=True)

    def _control_magbot(self) -> None:
        """Couple the MagBot mechanics with the rail mounts (platform a,b-rotations)."""
        # the platform a,b-rotations result from the rail mount rotations, which are held by the movers or by the station
        mover_quats = self.magbots.get_mover_quats_from_rail_mounts(data=self.data)
        self.magbots.control_platform_ab_rot(
            model=self.model, data=self.data, mover_a_quats=mover_quats[:, 0], mover_b_quats=mover_quats[:, 1]
        )

    def _apply_mover_controls(self, mover_setposes: np.ndarray, mover_quats_d: np.ndarray | None = None) -> None:
        """Calculate the controls of one control cycle, i.e. switch the welds of the rail mounts, update the mover impedance controllers
        and couple the MagBot mechanics with the movers. This method must be called before each MuJoCo integrator step.

        :param mover_setposes: the desired mover poses (numpy array of shape (num_movers, 6) using Euler angles (xyz) in rad)
        :param mover_quats_d: the desired mover rotations (numpy array of shape (num_movers, 4)), defaults to None, i.e. they are
            computed from ``mover_setposes``
        """
        quats_d = rotations_utils.vec_euler2quat(mover_setposes[:, 3:]) if mover_quats_d is None else mover_quats_d
        # weld the rail mounts to the movers or the station
        self._update_equality_constraints(mover_pos_d=mover_setposes)
        # update controls for mover impedance controllers
        mover_attachments = self.magbots.get_mover_attachments()
        for idx_mover, attachment in enumerate(mover_attachments):
            # only movers that are attached to a connection carry the MagBot
            additional_mass = 0.0 if attachment is None else self.magbots.magbot_masses[attachment[0]] / 2
            self.impedance_controllers[idx_mover].update(
                model=self.model,
                data=self.data,
                pos_d=mover_setposes[idx_mover, :3],
                quat_d=quats_d[idx_mover],
                additional_mass=additional_mass,
            )
        # control MagBot
        self._control_magbot()

    def step(
        self,
        mover_setposes: np.ndarray,
        num_cycles: int = 40,
        render_every_cycle: bool = False,
    ) -> dict[str, Any]:
        """Perform one or multiple simulation steps, including caluclation of mover controls, MuJoCo integrator steps, rendering, and
        collision checking. The remaining cycles are skipped if a collision or a rotation violation is detected. Afterwards, the product
        is counted and respawned if it came to rest in the bin or outside the bin below the platform heights (see
        ``num_delivered_products``, ``num_dropped_products``).

        :param mover_setposes: The desired mover poses (numpy array of shape (num_movers, 6) using Euler angles (xyz) in rad),
            [idx_mover,:]: set pose of mover idx_mover. The movers are controlled directly instead of via the platform set poses, e.g. to
            move the movers to the MagBot in the station (see ``get_mover_poses_from_magbot_setpose()`` to control the MagBot).
        :param num_cycles: the number of control cycles for which to apply the same controls, defaults to 40
        :param render_every_cycle: whether to call ``render()`` after each integrator step in the ``step()`` method, defaults to
            False. Rendering every cycle leads to a smoother visualization of the scene, but can also be computationally expensive.
            Thus, this parameter provides the possibility to speed up training and evaluation. Regardless of this parameter, the scene
            is always rendered after ``num_cycles``
        :return: the info dictionary with the collision types of the last executed cycle (see ``_get_info()``) and the product events
            ``product_delivered`` and ``product_dropped``
        """
        assert mover_setposes.shape == (self.num_movers, 6)
        assert num_cycles >= 1

        mover_quats_d = rotations_utils.vec_euler2quat(mover_setposes[:, 3:])
        mover_collision, wall_collision, other_collision, collision_info = False, False, False, None
        for _ in range(0, num_cycles):
            # calculate controls
            self._apply_mover_controls(mover_setposes=mover_setposes, mover_quats_d=mover_quats_d)
            # integration
            mujoco.mj_step(self.model, self.data, nstep=1)
            # render every cycle for a smooth visualization of the movement
            if render_every_cycle:
                self.render()
            # check collisions every cycle to ensure that the collisions are detected and all intermediate mover positions are valid
            # and without collisions
            mover_collision, wall_collision, other_collision, collision_info = self._check_collisions()
            if mover_collision or wall_collision or other_collision:
                break

        product_delivered, product_dropped = self._update_product()
        self.render()

        info = self._get_info(
            mover_collision=mover_collision,
            wall_collision=wall_collision,
            other_collision=other_collision,
            collision_info=collision_info,
        )
        info.update({'product_delivered': product_delivered, 'product_dropped': product_dropped})
        return info


if __name__ == '__main__':
    # Minimal demo: the movers (mover idx_c carries connection idx_c) rotate to the docking yaws at their start positions (rotation
    # points), pick up the MagBot from the station and transport products from the supply rack to the bin until the demo is interrupted
    # (Ctrl+C). More extensive demos are located in the MagBotSim_Solutions repository.
    # start positions of the movers in connection order (rotation points) [m]
    env = ReconfigurationTransportEnv(initial_mover_xy_pos=np.array([[0.72, 0.36], [0.24, 0.36]]))
    num_cycles = 20

    # mover set poses (connection order): in the station, at the station entrances (145 mm along the entry axis of the station), 200 mm
    # in front of the entrances and at the start positions (with the docking yaws and with a yaw of 0)
    station = env.stations[0]
    p_station = env.station_mover_poses[0]
    p_entrance = p_station.copy()
    p_entrance[:, :2] += np.stack([np.zeros(env.num_movers), 0.145 * station.MOVER_ENTRY_SIGNS], axis=1) @ station.rotation_matrix_xy().T
    p_pre_station = p_entrance.copy()
    p_pre_station[:, 1] -= 0.2
    p_start_docking = p_station.copy()
    p_start_docking[:, :2] = env.mover_start_xy_pos
    p_start = p_start_docking.copy()
    p_start[:, 5] = 0.0
    # platform set poses: in the station, taken out of the station by 0.25 m (along the x-axis of the station) and of the transport task
    p_taken_out = p_station.copy()
    p_taken_out[:, :2] += station.rotation_matrix_xy() @ np.array([0.25, 0.0])
    magbot_station = env.get_magbot_setpose_from_mover_poses(mover_poses=p_station)[0]
    magbot_taken_out = env.get_magbot_setpose_from_mover_poses(mover_poses=p_taken_out)[0]
    approach, supply, bin_pose = env.SUPPLY_APPROACH_PLATFORM_POSE, env.SUPPLY_PLATFORM_POSE, env.BIN_PLATFORM_POSE
    pick = supply.copy()
    pick[2] = env.PICK_PLATFORM_Z  # lift the product off the poles of the supply rack
    retract = pick.copy()
    retract[1] = env.RETRACT_PLATFORM_Y  # pull the product out of the supply rack
    bin_retract = bin_pose.copy()
    bin_retract[1] = env.RETRACT_PLATFORM_Y
    dump = bin_pose.copy()
    dump[3] = env.DUMP_PLATFORM_A_ROT  # tilt the platform towards the bin (mover a is at a rotation point)

    def move(start: np.ndarray, end: np.ndarray, magbot: bool = False, duration: float = 1.0, hold: float = 0.25) -> None:
        """Move the movers (or the platform) linearly from start to end and hold the end pose."""
        dt = env.cycle_time * num_cycles
        for s in np.concatenate([np.linspace(0.0, 1.0, int(duration / dt)), np.ones(int(hold / dt))]):
            setposes = start + s * (end - start)
            if magbot:
                setposes = env.get_mover_poses_from_magbot_setpose(magbot_setposes=setposes.reshape((1, 6)))
            info = env.step(mover_setposes=setposes, num_cycles=num_cycles)
            if any(value for key, value in info.items() if not key.startswith('product_')):
                print(f'collision or rotation violation: {info}')

    try:
        # pick up the MagBot and take it out of the station
        move(p_start, p_start_docking)  # rotate at the rotation points
        move(p_start_docking, p_pre_station)
        move(p_pre_station, p_entrance)
        move(p_entrance, p_station)  # enter the station along the entry axis (rail mounts still welded to the station)
        move(magbot_station, magbot_taken_out, magbot=True)  # take out the MagBot (rail mounts welded to the movers)
        move(magbot_taken_out, approach, magbot=True)
        # transport products from the supply rack to the bin
        while True:
            move(approach, supply, magbot=True)  # under the product
            move(supply, pick, magbot=True)
            move(pick, retract, magbot=True)
            move(retract, bin_retract, magbot=True)
            move(bin_retract, bin_pose, magbot=True)
            move(bin_pose, dump, magbot=True, hold=3.0)  # tilt and wait until the product has fallen into the bin
            move(dump, bin_pose, magbot=True)
            move(bin_pose, bin_retract, magbot=True)
            move(bin_retract, approach, magbot=True)
            print(f'delivered products: {env.num_delivered_products}, dropped products: {env.num_dropped_products}')
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

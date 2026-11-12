from collections.abc import Sequence
import itertools
from typing import Any
import numpy as np
import mujoco
from magbotsim import (
    BasicMagBotEnv,
    MoverImpedanceController,
    GripperMagBotsAPM4330,
    GripperMagBotsAPM4330Params,
    GripperStation,
    PlatformStation,
    SixDPlatformMagBotsAPM4330,
    Station,
    StationCollisionProxies,
)
from magbotsim.utils import geometry_2D_utils, rotations_utils


class ReconfigurationEnv(BasicMagBotEnv):
    """Simple benchmark environment for reconfiguration testing. At the beginning, the MagBot is in the station and the movers are
    free on the table at the positions ``initial_mover_xy_pos``.

    Two MagBot types are supported, each with its own station type:

    - ``'gripper'``: Gripper MagBot (3 movers) in a ``GripperStation``
    - ``'sixD_platform'``: 6D-Platform MagBot (2 movers) in a ``PlatformStation``

    The MagBot type is selected at initialization (``magbot_types``). The number of movers in the environment equals the number of
    connections (rail mounts) of the MagBot. Mover ``idx_c`` is attached to connection ``idx_c``, i.e. all mover set poses and station
    mover poses are given in connection order. For the 6D-Platform MagBot, connection 0 (rail mount 0) is attached to the mover that
    controls the platform's beta-rotation (mover b) and connection 1 (rail mount 1) to the mover that controls the alpha-rotation
    (mover a), see ``SixDPlatformMagBotsAPM4330``.

    The scene can be reset (and randomized) without reloading the model using ``reset_scene()``, e.g. from the reset callback of the RL
    variant of this environment (``ReconfigurationRLEnv``), which inherits from this class.

    ``step()`` returns an info dictionary that contains the collision types (all bool), analogous to the RL environments:

    - ``mover_collision``: collision between two movers
    - ``wall_collision``: collision between a mover and a wall
    - ``station_proxy_collision``: contact of any station collision proxy (rail mount proxy, rail rod or station pin) with any other
      geom (see ``get_station_proxy_collisions()``)
    - ``rail_rod_collision``: contact of a rail rod, e.g. an empty mover enters or leaves a station with a sideways misalignment
      larger than the station proxy tolerance (the rail rod hits the jaws of a rail mount)
    - ``station_pin_collision``: contact of a station pin, e.g. a MagBot carried by movers is moved into or out of a station with a
      sideways misalignment larger than the station proxy tolerance (the rail mount hits the station pins)

    :param magbot_types: the MagBot types that can be used (see ``MAGBOT_TYPES``), defaults to both types. If several types are
        specified, the MagBot type is chosen randomly (``self.rng``) at initialization, otherwise the specified type is used. The chosen
        type is stored in ``magbot_type``.
    :param initial_mover_xy_pos: the initial (x, y) positions of the movers on the table [m]. Either a numpy array of shape
        (num_movers, 2), which must fit the chosen MagBot type, or a dict {MagBot type: numpy array of shape (num_movers, 2)}, which is
        required to specify positions if several MagBot types can be chosen. Defaults to None, i.e. the positions are sampled uniformly
        on the free space of the table (no wall, mover or parked rail mount collisions). This also applies to MagBot types that are
        missing in the dict.
    :param station_poses: the poses of the station anchors (numpy array of shape (num_stations, 4): (x, y, z) position in m and yaw in
        rad). The anchor is the m2 mount of a ``GripperStation`` and the corresponding point on the extrusion of a ``PlatformStation``,
        i.e. both station types occupy the same place for the same pose. All stations have the station type of the chosen MagBot type.
        The number of stations can differ from the number of MagBots, but there must be at least one station per MagBot, since MagBot
        i is generated in station i. The rail mounts are welded to the station if a mover leaves the nominal station position (see
        ``Station.get_mover_poses()``) along the entry axis of the station, and welded to the mover if the mover leaves the parked rail
        mount along the take-out axis.
        Defaults to None, i.e. a single station at the top extrusion.
    :param station_switch_tol: the maximum distance between a rail mount and the nominal station position (put-back), or between a
        mover set position and a parked rail mount (pick-up), at which the welds are switched [m], defaults to 0.001
    :param station_proxy_tol: the sideways misalignment with which a mover (or a MagBot carried by movers) can enter a station
        without colliding [m], defaults to 0.002. The station switch is modelled by hidden collision proxies (rail mount proxies and
        station pins, see ``StationCollisionProxies``), which are shown in the viewer by pressing '3' (geom group 3). With 0, the proxies
        match the real geometries exactly, i.e. in practice they always collide.
    :param randomize_magbot_station_pos: whether to shift each MagBot in its station by a random offset along the station x-axis
        (take-out axis, i.e. along the station pins) at each reset, defaults to False. All rail mounts of a MagBot get the same offset.
    :param magbot_station_randomization_x_range: the half-range of the uniform random offset of the MagBots along the station x-axis
        [m], defaults to 0.005. Only used if ``randomize_magbot_station_pos`` is True.
    :param seed: the seed of the RNG used for the randomization (``self.rng``), defaults to None
    :param render_mode: the render mode, defaults to 'human'. Use None to run without rendering (e.g. in tests).
    """

    MAGBOT_TYPES = ('gripper', 'sixD_platform')
    # station type and number of connections (= number of movers) of each MagBot type
    STATION_CLASSES: dict[str, type[Station]] = {'gripper': GripperStation, 'sixD_platform': PlatformStation}
    NUM_CONNECTIONS = {'gripper': GripperMagBotsAPM4330.NUM_CONNECTIONS, 'sixD_platform': SixDPlatformMagBotsAPM4330.NUM_CONNECTIONS}
    DEFAULT_STATION_POSES = np.array([[0.480, 0.720, 0.01, np.deg2rad(-90)]])  # single station at the top extrusion
    # MuJoCo options of the Gripper MagBot (set by GripperMagBotsAPM4330), which are also used for the 6D-Platform MagBot
    OPTION_XML_STR = (
        '\t<option timestep="0.001" cone="elliptic" jacobian="dense" gravity="0 0 -9.81" '
        'integrator="implicitfast"><flag diagexact="enable"/></option>'
    )

    def __init__(
        self,
        magbot_types: Sequence[str] = MAGBOT_TYPES,
        initial_mover_xy_pos: np.ndarray | dict[str, np.ndarray] | None = None,
        station_poses: np.ndarray | None = None,
        station_switch_tol: float = 0.001,
        station_proxy_tol: float = 0.002,
        randomize_magbot_station_pos: bool = False,
        magbot_station_randomization_x_range: float = 0.005,
        seed: int | None = None,
        render_mode: str | None = 'human',
    ) -> None:
        # randomization
        self.rng = np.random.default_rng(seed)
        self.randomize_magbot_station_pos = randomize_magbot_station_pos
        assert magbot_station_randomization_x_range >= 0.0
        self.magbot_station_randomization_x_range = magbot_station_randomization_x_range

        # MagBot type (chosen randomly if several types are specified)
        magbot_types = tuple(dict.fromkeys(magbot_types))  # remove duplicates, keep order
        assert len(magbot_types) > 0, 'at least one MagBot type is required'
        assert set(magbot_types) <= set(self.MAGBOT_TYPES), f'unknown MagBot type, valid types: {self.MAGBOT_TYPES}'
        self.magbot_type = magbot_types[0] if len(magbot_types) == 1 else str(self.rng.choice(magbot_types))
        mover_count = self.NUM_CONNECTIONS[self.magbot_type]

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

        # mover distance m0-m1 of the Gripper MagBot model (configuration in which the MagBot XML is generated)
        self.initial_m0m1_xy_dist = np.array([0.48, 0.0])  # [m]
        self.initial_mover_z_pos = 0.001

        # MagBots
        self.num_magbots = 1  # <-- number of MagBots in this environment
        self.num_connections = mover_count
        # [idx_c]: indices of the movers attached to connection idx_c (one per MagBot)
        self.indices_mover_c = [np.array([idx_c]) for idx_c in range(0, self.num_connections)]
        # hidden collision proxies of the rail mounts and the station pins
        self.station_proxies = StationCollisionProxies(tol=station_proxy_tol)
        if self.magbot_type == 'gripper':
            self.magbot_params = GripperMagBotsAPM4330Params()
            self.magbots = GripperMagBotsAPM4330(
                num_magbots=self.num_magbots,
                indices_mover_c0=self.indices_mover_c[0],
                indices_mover_c1=self.indices_mover_c[1],
                indices_mover_c2=self.indices_mover_c[2],
                magbot_params=self.magbot_params,
                cycle_time=0.001,  # timestep of the MuJoCo model (specified by the MagBot)
                station_proxies=self.station_proxies,
            )
        else:
            # connection 0 (rail mount 0) is attached to mover b, connection 1 (rail mount 1) to mover a
            self.magbots = SixDPlatformMagBotsAPM4330(
                num_magbots=self.num_magbots,
                indices_mover_a=self.indices_mover_c[1],
                indices_mover_b=self.indices_mover_c[0],
                station_proxies=self.station_proxies,
            )

        # initial mover positions (movers free on the table); None: sampled in reset_scene()
        if isinstance(initial_mover_xy_pos, dict):
            assert set(initial_mover_xy_pos.keys()) <= set(self.MAGBOT_TYPES), f'unknown MagBot type, valid types: {self.MAGBOT_TYPES}'
            initial_mover_xy_pos = initial_mover_xy_pos.get(self.magbot_type, None)
        assert initial_mover_xy_pos is None or initial_mover_xy_pos.shape == (mover_count, 2), (
            f'initial_mover_xy_pos must have the shape ({mover_count}, 2) for the {self.magbot_type} MagBot'
        )
        # fixed start positions of the chosen MagBot type (None: the start positions are sampled), e.g. reused at each RL reset
        self.fixed_mover_start_xy_pos = None if initial_mover_xy_pos is None else initial_mover_xy_pos.copy()

        # stations (the station type depends on the MagBot type)
        if station_poses is None:
            station_poses = self.DEFAULT_STATION_POSES
        assert station_poses.ndim == 2 and station_poses.shape[1] == 4
        self.num_stations = station_poses.shape[0]
        assert self.num_stations >= self.num_magbots, 'MagBot i is generated in station i, i.e. at least one station per MagBot required'
        station_class = self.STATION_CLASSES[self.magbot_type]
        self.stations = [
            station_class(pos=station_poses[idx_station, :3], yaw=station_poses[idx_station, 3], name=f'station_{idx_station}')
            for idx_station in range(0, self.num_stations)
        ]
        self.station_switch_tol = station_switch_tol

        # mover set poses at the stations (the equality constraints of the rail mounts are only switched at these positions)
        # shape: (num_stations, num_connections, 6), [:, idx_c, :]: pose of the mover attached to connection idx_c
        self.station_mover_poses = np.array([station.get_mover_poses(z_m=self.initial_mover_z_pos) for station in self.stations])

        # MagBot and connection to which each mover is attached
        self.mover_magbot_indices = np.zeros(mover_count, dtype=int)
        self.mover_connection_indices = np.zeros(mover_count, dtype=int)
        for idx_c, indices_mover in enumerate(self.indices_mover_c):
            self.mover_magbot_indices[indices_mover] = np.arange(0, self.num_magbots)
            self.mover_connection_indices[indices_mover] = idx_c

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

        # impedance controller
        # the rotational damping is computed from the rotational inertia of the movers, which keeps the controller stable regardless of
        # whether a mover carries the MagBot or not, i.e. the same rotational stiffness can be used in both cases.
        # The mover c-rotations of the Gripper MagBot are controlled by the MagBot (see GripperMagBotsAPM4330.control_gripper()), the
        # mover c-rotations of the 6D-Platform MagBot by the impedance controllers.
        if self.magbot_type == 'gripper':
            joint_mask, rotational_stiffness = np.array([1, 1, 1, 1, 1, 0]), np.array([5.0, 5.0, 0.0])
        else:
            joint_mask, rotational_stiffness = np.array([1, 1, 1, 1, 1, 1]), np.array([5.0, 5.0, 5.0])
        self.impedance_controllers = [
            MoverImpedanceController(
                model=self.model,
                mover_joint_name=self.mover_joint_names[mover_idx],
                mover_half_height=self.mover_size[mover_idx, 2],
                joint_mask=joint_mask,
                translational_stiffness=np.array([5000.0, 5000.0, 10.0]),
                rotational_stiffness=rotational_stiffness,
            )
            for mover_idx in range(self.num_movers)
        ]

        # reload model to add MagBots and actuators (includes reset_scene(), i.e. the start positions are sampled if not specified)
        self.reload_model(mover_start_xy_pos=initial_mover_xy_pos)
        self.initial_mover_xy_pos = self.mover_start_xy_pos.copy()

    def _generate_magbot_xml_strings(self, custom_model_xml_strings: dict[str, str]) -> dict[str, str]:
        """Generate the custom model XML strings for the MagBots in the stations (MagBot i in station i) and add them to the
        ``custom_model_xml_strings``-dict. The pose of a MagBot is deduced from the station mover poses.

        :param custom_model_xml_strings: the current ``custom_model_xml_strings``-dict which is modified by this method
        :return: the modified ``custom_model_xml_strings``-dict
        """
        mover_qpos = self.get_mover_qpos(mover_names=self.mover_names, add_noise=False)
        station_mover_xy_pos = self.station_mover_poses[: self.num_magbots, :, :2]  # (num_magbots, num_connections, 2)
        # the MagBot-specific branches check the type of self.magbots (instead of self.magbot_type), so that type checkers can narrow it
        if isinstance(self.magbots, GripperMagBotsAPM4330):
            # the gripper is located in the middle between the movers of connections 0 and 1 and its yaw is given by the direction from
            # connection 0 to connection 1. Connection 0 is located half the initial m0-m1 distance next to the gripper.
            station_pos_xyz_c0 = mover_qpos[self.indices_mover_c[0], :3].copy()
            station_xy_pos_c0, station_xy_pos_c1 = station_mover_xy_pos[:, 0], station_mover_xy_pos[:, 1]
            station_c0c1_xy_dirs = station_xy_pos_c1 - station_xy_pos_c0
            station_magbot_yaws = np.arctan2(station_c0c1_xy_dirs[:, 1], station_c0c1_xy_dirs[:, 0])
            c, s, d = np.cos(station_magbot_yaws), np.sin(station_magbot_yaws), self.initial_m0m1_xy_dist
            m0m1_xy_dists = np.stack([c * d[0] - s * d[1], s * d[0] + c * d[1]], axis=1)  # m0-m1 distance rotated by the MagBot yaw
            station_pos_xyz_c0[:, :2] = 0.5 * (station_xy_pos_c0 + station_xy_pos_c1) - 0.5 * m0m1_xy_dists
            return self.magbots.generate_magbot_xml_strings(
                initial_pos_xyz_mover_c0=station_pos_xyz_c0,
                custom_model_xml_strings=custom_model_xml_strings,
                initial_yaw=station_magbot_yaws,
            )
        # 6D-Platform MagBot: the MagBot base is located at rail mount 0 (connection 0, mover b) and the platform c-rotation is given by
        # the direction from mover a (connection 1) to mover b (connection 0)
        station_pos_xyz_b = mover_qpos[self.indices_mover_c[0], :3].copy()
        station_pos_xyz_b[:, :2] = station_mover_xy_pos[:, 0]
        station_ab_xy_dirs = station_mover_xy_pos[:, 0] - station_mover_xy_pos[:, 1]
        station_platform_yaws = np.arctan2(station_ab_xy_dirs[:, 1], station_ab_xy_dirs[:, 0])
        custom_model_xml_strings = self.magbots.generate_magbot_xml_strings(
            initial_pos_xyz_mover_b=station_pos_xyz_b,
            custom_model_xml_strings=custom_model_xml_strings,
            initial_yaw=station_platform_yaws,
        )
        custom_model_xml_strings['custom_option_xml_str'] = self.OPTION_XML_STR
        return custom_model_xml_strings

    def _custom_xml_string_callback(self, custom_model_xml_strings: dict[str, str] | None = None) -> dict[str, str]:
        """Add the MagBots, stations and actuators for the movers to the MuJoCo model by modifying the ``custom_model_xml_strings``-dict.

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
            if isinstance(self.magbots, GripperMagBotsAPM4330):
                # MagBot gripper and mover c actuators
                actuator_lines.append(self.magbots.generate_actuator_xml_strings())
            else:
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

            # assets application
            custom_assets_xml_str = custom_model_xml_strings['custom_assets_xml_str']
            application_assets_list = [
                '\n\t\t<!-- assets reconfiguration benchmark -->',
                # stations (all stations have the same type)
                type(self.stations[0]).generate_assets_xml_str(),
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

        if self.render_mode is not None:
            self.viewer_collection.reload_model(self.model, self.data)

        self.reset_scene(mover_start_xy_pos=mover_start_xy_pos)

    def reset_scene(
        self,
        mover_start_xy_pos: np.ndarray | None = None,
        magbot_station_x_offsets: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Reset the scene without reloading the model: MagBot i is parked in station i (optionally shifted along the station x-axis) and
        the movers are free on the table. The MuJoCo data, the welds of the rail mounts and the MagBot controller states are reset.
        This method can be reused by the reset callback of an RL variant of this environment.

        :param mover_start_xy_pos: the (x,y) start positions of the movers (numpy array of shape (num_movers,2)) [m], defaults to None.
            If None, the start positions are sampled on the free space of the table (see ``sample_mover_start_xy_pos()``).
        :param magbot_station_x_offsets: the offsets of the MagBots along the x-axis of their stations (numpy array of shape
            (num_magbots,)) [m], defaults to None. If None, the offsets are sampled if ``randomize_magbot_station_pos`` is True (see
            ``sample_magbot_station_x_offsets()``), otherwise they are zero.
        :param rng: the RNG used for sampling, defaults to None, i.e. ``self.rng``. An RL variant can pass ``self.np_random``.
        """
        rng = self.rng if rng is None else rng

        mujoco.mj_resetData(self.model, self.data)
        if isinstance(self.magbots, GripperMagBotsAPM4330):
            self.magbots.reset_yaw_extractor()
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
        # index of the station to which the rail mount of a mover is welded (-1: rail mount welded to the mover)
        self.mover_station_indices = self.mover_magbot_indices.copy()
        for idx_mover in range(0, self.num_movers):
            idx_station = self.mover_station_indices[idx_mover]
            self._set_station_weld_relpose(idx_mover=idx_mover, idx_station=idx_station)
            self.data.eq_active[self.eq_ids_station_weld[idx_mover, idx_station]] = True
            self.data.eq_active[self.eq_ids_mover_weld[idx_mover]] = False

        # mover start positions (movers free on the table)
        if mover_start_xy_pos is None:
            mover_start_xy_pos = self.sample_mover_start_xy_pos(rng=rng)
        else:
            assert mover_start_xy_pos.shape == (self.num_movers, 2)
            assert self._mover_start_xy_pos_is_valid(mover_start_xy_pos), 'invalid mover start positions (collision)'
        self.mover_start_xy_pos = mover_start_xy_pos.copy()
        self.data.qpos[self.mover_qpos_indices] = self._mover_start_qpos(mover_start_xy_pos)
        self.data.qvel[self.mover_qvel_qacc_indices] = 0.0
        mujoco.mj_forward(self.model, self.data)

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
        indices_mover = np.where(self.mover_station_indices >= 0)[0]
        yaws = self.station_mover_poses[self.mover_station_indices[indices_mover], self.mover_connection_indices[indices_mover], 5]
        qpos = np.zeros((len(indices_mover), 7))
        qpos[:, :2] = self.get_rail_mount_xy()[indices_mover]
        qpos[:, 3] = np.cos(0.5 * yaws)
        qpos[:, 6] = np.sin(0.5 * yaws)
        sizes = self.get_c_size_arr(c_size=self.c_size, num_reps=self.num_movers)[indices_mover]
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

        :return: the positions (numpy array of shape (num_movers,2)), [idx_mover,:]: rail mount attached to mover idx_mover [m]
        """
        return self.data.xpos[self.body_ids_rail_mount, :2].copy()

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

    def _check_for_other_collisions_callback(self) -> tuple[bool, dict[str, bool]]:
        """Check for collisions besides mover and wall collisions, i.e. contacts of the station collision proxies (see
        ``get_station_proxy_collisions()``). This method has the signature of the corresponding callback of
        ``BasicMagBotSingleAgentEnv``, i.e. it is also used by the RL variant of this environment.

        :return:
            - whether there is a station proxy collision (bool)
            - a dictionary with the station proxy collision types (see ``get_station_proxy_collisions()``)
        """
        collision_info = self.get_station_proxy_collisions()
        return collision_info['station_proxy_collision'], collision_info

    def _check_collisions(self) -> tuple[bool, bool, bool, dict[str, bool]]:
        """Check for mover, wall and other collisions (station collision proxies) in the current state. Noise is added to the mover
        positions, since it would also occur in a real system.

        :return: whether there is a mover collision, a wall collision or another collision (each bool), and a dictionary with the
            station proxy collision types (see ``get_station_proxy_collisions()``)
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
        ``BasicMagBotSingleAgentEnv``, i.e. it is also used by the RL variant of this environment.

        :param mover_collision: whether there is a collision between two movers
        :param wall_collision: whether there is a collision between a mover and a wall
        :param other_collision: whether there is a station proxy collision (already contained in ``collision_info``)
        :param achieved_goal: not used in this environment, defaults to None
        :param desired_goal: not used in this environment, defaults to None
        :param collision_info: a dictionary with the station proxy collision types (see ``get_station_proxy_collisions()``),
            defaults to None. If None, the station proxy collisions are checked in the current state.
        :return: the info dictionary with keys ``mover_collision``, ``wall_collision``, ``station_proxy_collision``,
            ``rail_rod_collision`` and ``station_pin_collision`` (bool values)
        """
        if collision_info is None:
            collision_info = self.get_station_proxy_collisions()
        info = {'mover_collision': bool(mover_collision), 'wall_collision': bool(wall_collision)}
        info.update({key: bool(value) for key, value in collision_info.items()})
        return info

    def _update_cached_equality_ids(self) -> None:
        """Update the cached IDs of the mover and station welds of the rail mounts and of the bodies connected by the station welds."""
        self.eq_ids_mover_weld = np.zeros(self.num_movers, dtype=int)
        self.eq_ids_station_weld = np.zeros((self.num_movers, self.num_stations), dtype=int)
        self.body_ids_rail_mount = np.zeros(self.num_movers, dtype=int)
        for idx_mover in range(0, self.num_movers):
            idx_magbot, idx_c = self.mover_magbot_indices[idx_mover], self.mover_connection_indices[idx_mover]
            self.eq_ids_mover_weld[idx_mover] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f'mover_weld_c{idx_c}_{idx_magbot}'
            )
            self.body_ids_rail_mount[idx_mover] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f'rail_mount_{idx_c}_{idx_magbot}'
            )
            for idx_station, station in enumerate(self.stations):
                self.eq_ids_station_weld[idx_mover, idx_station] = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f'{station.name}_weld_c{idx_c}_{idx_magbot}'
                )
        self.body_ids_station = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, station.name) for station in self.stations], dtype=int
        )

    def _set_station_weld_relpose(self, idx_mover: int, idx_station: int) -> None:
        """Set the relative pose of the station weld of a rail mount to the current relative pose between the rail mount and the
        station, i.e. the rail mount is held in its current pose when the weld is activated.

        :param idx_mover: the index of the mover whose rail mount is welded to the station
        :param idx_station: the index of the station to which the rail mount is welded
        """
        id_body1 = self.body_ids_rail_mount[idx_mover]
        id_body2 = self.body_ids_station[idx_station]
        # weld (anchor = 0): xpos1 + xmat1 @ relpos = xpos2 and xquat1 * relquat = xquat2
        quat1_inv = np.zeros(4)
        mujoco.mju_negQuat(quat1_inv, self.data.xquat[id_body1])
        relpos = np.zeros(3)
        mujoco.mju_rotVecQuat(relpos, self.data.xpos[id_body2] - self.data.xpos[id_body1], quat1_inv)
        relquat = np.zeros(4)
        mujoco.mju_mulQuat(relquat, quat1_inv, self.data.xquat[id_body2])

        eq_id = self.eq_ids_station_weld[idx_mover, idx_station]
        self.model.eq_data[eq_id, 0:3] = 0.0
        self.model.eq_data[eq_id, 3:6] = relpos
        self.model.eq_data[eq_id, 6:10] = relquat

    def _update_equality_constraints(self, mover_pos_d: np.ndarray) -> None:
        """Switch the welds of the rail mounts between the movers and the stations based on the real rail mount positions. The switch
        happens on the first step in which a mover leaves (i.e. the distance increases), so that the rail mount does not jump.

        - pick-up: the rail mount is parked in a station and the last mover set position is within the station switch tolerance of the
          real rail mount position. If the mover leaves along the take-out axis of the station, the rail mount is welded to the mover.
        - put-back: the rail mount is carried by the mover, the real rail mount position is within the station switch tolerance of the
          nominal station position of its connection and the mover set position moves away from it. If the mover leaves along the
          entry axis of the station, the rail mount is welded to this station.

        Entering a station never switches the welds, since the decision only depends on the direction in which the mover leaves.

        :param mover_pos_d: the current mover set positions (numpy array of shape (num_movers, >=2))
        """
        if self.last_mover_pos_d is not None:
            rail_mount_xy = self.get_rail_mount_xy()
            for idx_mover in range(0, self.num_movers):
                last_xy, curr_xy = self.last_mover_pos_d[idx_mover, :2], mover_pos_d[idx_mover, :2]
                delta = curr_xy - last_xy
                if np.linalg.norm(delta) < 1e-6:  # ignore deltas close to numerical noise
                    continue

                idx_station = self.mover_station_indices[idx_mover]
                if idx_station >= 0:
                    # pick-up: the mover leaves the parked rail mount along the take-out axis -> weld to mover
                    last_dist = np.linalg.norm(last_xy - rail_mount_xy[idx_mover])
                    curr_dist = np.linalg.norm(curr_xy - rail_mount_xy[idx_mover])
                    is_leaving = last_dist < self.station_switch_tol and curr_dist > last_dist
                    if not is_leaving or self.stations[idx_station].is_along_entry_axis(delta_xy=delta):
                        continue
                    self.data.eq_active[self.eq_ids_station_weld[idx_mover, idx_station]] = False
                    self.data.eq_active[self.eq_ids_mover_weld[idx_mover]] = True
                    self.mover_station_indices[idx_mover] = -1
                else:
                    # put-back: the carried rail mount is at the nominal station position of its connection and the mover leaves along
                    # the entry axis -> weld to station
                    idx_c = self.mover_connection_indices[idx_mover]
                    station_xy = self.station_mover_poses[:, idx_c, :2]
                    rail_dists = np.linalg.norm(rail_mount_xy[idx_mover] - station_xy, axis=1)
                    idx_station = int(np.argmin(rail_dists))
                    last_dist = np.linalg.norm(last_xy - station_xy[idx_station])
                    curr_dist = np.linalg.norm(curr_xy - station_xy[idx_station])
                    is_leaving = rail_dists[idx_station] < self.station_switch_tol and curr_dist > last_dist
                    if not is_leaving or not self.stations[idx_station].is_along_entry_axis(delta_xy=delta):
                        continue
                    self._set_station_weld_relpose(idx_mover=idx_mover, idx_station=idx_station)
                    self.data.eq_active[self.eq_ids_station_weld[idx_mover, idx_station]] = True
                    self.data.eq_active[self.eq_ids_mover_weld[idx_mover]] = False
                    self.mover_station_indices[idx_mover] = idx_station

        self.last_mover_pos_d = mover_pos_d.copy()

    def get_mover_poses_from_magbot_setpose(self, magbot_setposes: np.ndarray, single_track: bool | np.ndarray = False) -> np.ndarray:
        """Return the mover set poses resulting from MagBot set poses (inverse kinematics of the MagBot).

        :param magbot_setposes: the desired MagBot set poses (using Euler angles (xyz) in rad). Gripper MagBot: gripper poses (numpy
            array of shape (num_magbots, 7)), the 7th entry is the desired distance between the two gripper fingers in m. 6D-Platform
            MagBot: platform poses (numpy array of shape (num_magbots, 6)).
        :param single_track: whether to calculate the mover set positions in single-track mode, defaults to False. Only used for the
            Gripper MagBot.
        :return: the mover set poses in connection order (numpy array of shape (num_movers, 6) using Euler angles (xyz) in rad)
        """
        if isinstance(self.magbots, GripperMagBotsAPM4330):
            mover_poses = self.magbots.gripperSetPose2MoverSetPose(
                gripper_pose_d=magbot_setposes, z_m=self.initial_mover_zpos, single_track=single_track
            )
        else:
            # [a, b] -> connection order [b, a] (the platform set poses are copied, since they are clipped in-place)
            mover_poses = self.magbots.platformSetPose2MoverSetPose(
                platform_pose_d=magbot_setposes.copy(), mover_z_d=float(self.initial_mover_zpos), use_euler=True
            )[:, ::-1]
        return mover_poses.reshape((-1, 6))

    def get_magbot_setpose_from_mover_poses(self, mover_poses: np.ndarray) -> np.ndarray:
        """Return the MagBot set poses resulting from mover set poses (forward kinematics of the MagBot).

        :param mover_poses: the mover set poses in connection order (numpy array of shape (num_movers, 6) using Euler angles (xyz) in
            rad)
        :return: the MagBot set poses (using Euler angles (xyz) in rad). Gripper MagBot: gripper poses (numpy array of shape
            (num_magbots, 7)). 6D-Platform MagBot: platform poses (numpy array of shape (num_magbots, 6)).
        """
        mover_poses = mover_poses.reshape((self.num_magbots, self.num_connections, 6))
        if isinstance(self.magbots, GripperMagBotsAPM4330):
            return self.magbots.moverSetPose2GripperSetPose(mover_poses=mover_poses)
        # connection order [b, a] -> [a, b]
        return self.magbots.moverSetPose2PlatformSetPose(mover_poses=mover_poses[:, ::-1], use_euler=True)

    def _control_magbot(self, mover_setposes: np.ndarray) -> None:
        """Couple the MagBot mechanics with the movers (gripper rotations and fingers, or platform a,b-rotations).

        :param mover_setposes: the desired mover poses (numpy array of shape (num_movers, 6) using Euler angles (xyz) in rad)
        """
        if isinstance(self.magbots, GripperMagBotsAPM4330):
            current_mover_qpos = self.get_mover_qpos(mover_names=self.mover_names, add_noise=False)
            current_mover_qvels = self.get_mover_qvel(mover_names=self.mover_names, add_noise=False)
            self.magbots.control_gripper(
                model=self.model,
                data=self.data,
                mover_poses_d=mover_setposes,
                current_mover_qpos=current_mover_qpos,
                current_mover_qvels=current_mover_qvels,
            )
        else:
            # the platform a,b-rotations result from the rail mount rotations, which are held by the movers or by the station
            mover_quats = self.magbots.get_mover_quats_from_rail_mounts(data=self.data)
            self.magbots.control_platform_ab_rot(
                model=self.model, data=self.data, mover_a_quats=mover_quats[:, 0], mover_b_quats=mover_quats[:, 1]
            )

    def _get_mover_quats_d(self, mover_setposes: np.ndarray) -> np.ndarray:
        """Return the desired mover rotations for the impedance controllers. The c-rotations of the Gripper MagBot movers are controlled
        by the MagBot (see ``__init__()``), i.e. the impedance controllers only keep the movers level.

        :param mover_setposes: the desired mover poses (numpy array of shape (num_movers, 6) using Euler angles (xyz) in rad)
        :return: the desired mover rotations as quaternions (numpy array of shape (num_movers, 4))
        """
        if self.magbot_type == 'gripper':
            return np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (self.num_movers, 1))
        return rotations_utils.vec_euler2quat(mover_setposes[:, 3:])

    def _apply_mover_controls(self, mover_setposes: np.ndarray, mover_quats_d: np.ndarray | None = None) -> None:
        """Calculate the controls of one control cycle, i.e. switch the welds of the rail mounts, update the mover impedance controllers
        and couple the MagBot mechanics with the movers. This method must be called before each MuJoCo integrator step and is also used
        by the RL variant of this environment.

        :param mover_setposes: the desired mover poses in connection order (numpy array of shape (num_movers, 6) using Euler angles
            (xyz) in rad)
        :param mover_quats_d: the desired mover rotations (numpy array of shape (num_movers, 4)), defaults to None, i.e. they are
            computed from ``mover_setposes`` (see ``_get_mover_quats_d()``)
        """
        quats_d = self._get_mover_quats_d(mover_setposes=mover_setposes) if mover_quats_d is None else mover_quats_d
        # weld the rail mounts to the movers or the station
        self._update_equality_constraints(mover_pos_d=mover_setposes)
        # update controls for mover impedance controllers
        for idx_mover in range(0, self.num_movers):
            # movers whose rail mount is held by the station do not carry the MagBot
            additional_mass = 0.0 if self.mover_station_indices[idx_mover] >= 0 else self.magbots.magbot_masses[0] / 2
            self.impedance_controllers[idx_mover].update(
                model=self.model,
                data=self.data,
                pos_d=mover_setposes[idx_mover, :3],
                quat_d=quats_d[idx_mover],
                additional_mass=additional_mass,
            )
        # control MagBot
        self._control_magbot(mover_setposes=mover_setposes)

    def step(
        self,
        mover_setposes: np.ndarray,
        num_cycles: int = 40,
        render_every_cycle: bool = False,
    ) -> dict[str, Any]:
        """Perform one or multiple simulation steps, including caluclation of mover controls, MuJoCo integrator steps, rendering, and
        collision checking. The remaining cycles are skipped if a collision is detected.

        :param mover_setposes: The desired mover poses in connection order (numpy array of shape (num_movers, 6) using Euler angles (xyz)
            in rad). The movers are controlled directly instead of via the MagBot set poses, e.g. to move the movers away from a MagBot in
            the station (see ``get_mover_poses_from_magbot_setpose()`` to control the MagBot).
        :param num_cycles: the number of control cycles for which to apply the same controls, defaults to 40
        :param render_every_cycle: whether to call ``render()`` after each integrator step in the ``step()`` method, defaults to
            False. Rendering every cycle leads to a smoother visualization of the scene, but can also be computationally expensive.
            Thus, this parameter provides the possibility to speed up training and evaluation. Regardless of this parameter, the scene
            is always rendered after ``num_cycles``
        :return: the info dictionary with the collision types of the last executed cycle (see ``_get_info()``)
        """
        assert mover_setposes.shape == (self.num_movers, 6)
        assert num_cycles >= 1

        mover_quats_d = self._get_mover_quats_d(mover_setposes=mover_setposes)
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

        self.render()

        return self._get_info(
            mover_collision=mover_collision,
            wall_collision=wall_collision,
            other_collision=other_collision,
            collision_info=collision_info,
        )


if __name__ == '__main__':
    # The demo loops through randomly initialized scenarios (MagBot type, station type and MagBot position in the station). For each
    # scenario, the movers pick up the MagBot from the station, take it out of the station, put it back and drive back to their
    # initial positions. The model only depends on the MagBot type (number of movers, MagBot and station type), i.e. a new environment
    # (and viewer window) is only created if the MagBot type changes. Otherwise, the scene of the existing environment is reset.
    single_track = np.array([False])  # Gripper MagBot only
    # MagBot types from which the type of each scenario is chosen randomly
    magbot_types = ('gripper', 'sixD_platform')
    # randomize the position of the MagBot in the station (shift along the station pins)
    randomize_magbot_station_pos = True

    # fixed initial mover positions (movers free on the table) for each MagBot type in connection order; use None to sample random
    # collision-free start positions (note that the straight-line paths of this example are only collision-free for suitable start
    # positions)
    initial_mover_xy_pos = {
        'gripper': np.array([[0.250, 0.150], [0.710, 0.150], [0.480, 0.150]]),  # [m]
        # connection 0 (mover b) is parked at the +x side of the station, connection 1 (mover a) at the -x side
        'sixD_platform': np.array([[0.710, 0.150], [0.250, 0.150]]),  # [m]
    }

    # station at the top extrusion (pose of the station anchor: x, y, z, yaw)
    station_poses = np.array([[0.480, 0.720, 0.01, np.deg2rad(-90)]])

    # the movers enter and leave the station along the entry axis (y-axis of the station body frame) from this distance [m]
    entrance_dist = 0.145
    # the movers carry the MagBot 0.25 m out of the station (center of the tiles)
    magbot_path = np.array([0.25, 0.0])  # [m]

    num_cycles = 20
    # movements defaults
    default_duration = 1.25
    default_hold_after_duration = 0.25

    def run_scenario(env: ReconfigurationEnv) -> None:
        """Pick up the MagBot, take it out of the station, put it back and drive the movers back to their initial positions."""
        station = env.stations[0]
        rot_station = station.rotation_matrix_xy()
        # nominal mover poses in the station (connection order = mover order); the MagBot is put back at these poses
        station_mover_poses = env.station_mover_poses[0].copy()
        center_mover_poses = station_mover_poses.copy()
        center_mover_poses[:, :2] += rot_station @ magbot_path

        station_magbot_pose = env.get_magbot_setpose_from_mover_poses(mover_poses=station_mover_poses)
        center_magbot_pose = env.get_magbot_setpose_from_mover_poses(mover_poses=center_mover_poses)

        # offsets of the station entrances in the station body frame (x-axis: out of the station, y-axis: entry axis of the station),
        # rotated into the world frame; each mover enters from the flat side of its rail mount
        entrance_offsets = np.stack([np.zeros(env.num_movers), entrance_dist * station.MOVER_ENTRY_SIGNS], axis=1) @ rot_station.T

        def station_setposes(station_poses_c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Mover set poses in the station, at the entrance of the station and at the pre-station point for the given mover poses in
            the station. The pre-station point is the entrance shifted by -200 mm in world y, from where the movers drive straight
            into the station.
            """
            entrance_poses_c = station_poses_c.copy()
            entrance_poses_c[:, :2] += entrance_offsets
            pre_station_poses_c = entrance_poses_c.copy()
            pre_station_poses_c[:, 1] -= 0.2  # [m]
            return station_poses_c, entrance_poses_c, pre_station_poses_c

        # put-back: nominal station poses
        station_mover_setposes, entrance_mover_setposes, pre_station_mover_setposes = station_setposes(station_mover_poses)

        # the movers start with a yaw of 0 and rotate to the station yaw on their way to the entrance
        initial_mover_setposes = station_mover_setposes.copy()
        initial_mover_setposes[:, :2] = env.mover_start_xy_pos  # start positions of the current scene
        initial_mover_setposes[:, 5] = 0.0

        dt = env.cycle_time * num_cycles  # [s] duration of one env step

        def interpolate(start: np.ndarray, end: np.ndarray, duration: float) -> list[np.ndarray]:
            """Linearly interpolate between two set poses, i.e. one set pose per env step."""
            return [start + s * (end - start) for s in np.linspace(0.0, 1.0, int(duration / dt))]

        def move_movers(
            start: np.ndarray, end: np.ndarray, duration: float = default_duration, hold_after_duration: float = default_hold_after_duration
        ) -> None:
            for mover_setposes in interpolate(start, end, duration=duration) + interpolate(end, end, duration=hold_after_duration):
                env.step(mover_setposes=mover_setposes, num_cycles=num_cycles)

        def move_magbot(
            start: np.ndarray, end: np.ndarray, duration: float = default_duration, hold_after_duration: float = default_hold_after_duration
        ) -> None:
            for magbot_setposes in interpolate(start, end, duration=duration) + interpolate(end, end, duration=hold_after_duration):
                mover_setposes = env.get_mover_poses_from_magbot_setpose(magbot_setposes=magbot_setposes, single_track=single_track)
                env.step(mover_setposes=mover_setposes, num_cycles=num_cycles)

        # pick-up: the movers drive to the real positions of the parked rail mounts
        pickup_poses_c = station_mover_poses.copy()
        pickup_poses_c[:, :2] = env.get_rail_mount_xy()
        pickup_mover_setposes, pickup_entrance_mover_setposes, pickup_pre_station_mover_setposes = station_setposes(pickup_poses_c)
        pickup_magbot_pose = env.get_magbot_setpose_from_mover_poses(mover_poses=pickup_poses_c)

        # 1) the movers drive from their initial positions to the pre-station point (200 mm in front of the station)
        move_movers(initial_mover_setposes, pickup_pre_station_mover_setposes)
        # 2) the movers drive straight into the station (+y)
        move_movers(pickup_pre_station_mover_setposes, pickup_entrance_mover_setposes)
        # 3) the movers enter the station along the entry axis (rail mounts still welded to the station)
        move_movers(pickup_entrance_mover_setposes, pickup_mover_setposes)
        # 4) the movers take the MagBot and move it along the take-out axis to the center (rail mounts welded to the movers)
        move_magbot(pickup_magbot_pose, center_magbot_pose)
        # 5) the movers move the MagBot back into the station (nominal station pose)
        move_magbot(center_magbot_pose, station_magbot_pose)
        # 6) the movers leave the station along the entry axis (rail mounts welded to the station)
        move_movers(station_mover_setposes, entrance_mover_setposes)
        # 7) the movers drive back to their pre station positions
        move_movers(entrance_mover_setposes, pre_station_mover_setposes)
        # 8) the movers drive back to their initial positions
        move_movers(pre_station_mover_setposes, initial_mover_setposes)

    rng = np.random.default_rng(seed=0)
    env: ReconfigurationEnv | None = None
    try:
        for idx_scenario in itertools.count():
            magbot_type = str(rng.choice(magbot_types))
            if env is not None and env.magbot_type == magbot_type and env.window_viewer_is_running():
                # same model: reset the scene in place (new MagBot offset if randomized), i.e. the viewer window stays open
                env.reset_scene(mover_start_xy_pos=env.fixed_mover_start_xy_pos, rng=rng)
            else:
                # new MagBot type (or viewer window closed by the user): new model -> new environment and viewer window
                if env is not None:
                    env.close()
                env = ReconfigurationEnv(
                    magbot_types=(magbot_type,),
                    initial_mover_xy_pos=initial_mover_xy_pos,
                    station_poses=station_poses,
                    randomize_magbot_station_pos=randomize_magbot_station_pos,
                    seed=idx_scenario,
                )
            print(
                f'scenario {idx_scenario}: {env.magbot_type} MagBot in a {type(env.stations[0]).__name__}, MagBot offset along the station '
                f'x-axis: {env.magbot_station_x_offsets * 1000} mm'
            )
            run_scenario(env)
    except KeyboardInterrupt:
        pass
    finally:
        if env is not None:
            env.close()

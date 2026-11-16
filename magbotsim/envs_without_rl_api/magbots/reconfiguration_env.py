from collections.abc import Sequence
import itertools
from pathlib import Path
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
from magbotsim.magbots.mover_coupling import NO_INDEX
from magbotsim.utils import geometry_2D_utils, rotations_utils


class ReconfigurationEnv(BasicMagBotEnv):
    """Simple benchmark environment for reconfiguration testing. At the beginning, the MagBot is in the station and the movers are
    free on the table at the positions ``initial_mover_xy_pos``.

    Two MagBot types are supported, each with its own station type:

    - ``'gripper'``: Gripper MagBot (3 connections) in a ``GripperStation``
    - ``'sixD_platform'``: 6D-Platform MagBot (2 connections) in a ``PlatformStation``

    The MagBot type is selected at initialization (``magbot_types``). The environment contains at least as many movers as the MagBot has
    connections (rail mounts), see ``num_movers``. All movers carry the same mover adapter, i.e. any mover can pick up any rail mount of
    the MagBot (see ``MagBotMoverCoupling``). The assignment of the movers to the connections is not fixed, but determined at runtime:
    a mover is attached to a rail mount when it picks up the rail mount from the station and detached when it puts the rail mount back
    (see ``_update_equality_constraints()``). The current assignment is given by ``connection_mover_indices`` (and
    ``mover_magbot_indices``, ``mover_connection_indices``), the station in which a rail mount is parked by ``connection_station_indices``.

    The mover adapter has a direction, i.e. a mover docked at a connection that is entered from the other side is rotated by pi
    (Gripper MagBot: connection 0, 6D-Platform MagBot: connection 1, see ``MOVER_YAW_OFFSETS`` of the MagBots). The station mover poses
    (``station_mover_poses``) and the mover set poses computed from MagBot set poses (``get_mover_poses_from_magbot_setpose()``) contain
    these rotations. For the 6D-Platform MagBot, connection 0 (rail mount 0) is attached to the mover that controls the platform's
    beta-rotation (mover b) and connection 1 (rail mount 1) to the mover that controls the alpha-rotation (mover a), see
    ``SixDPlatformMagBotsAPM4330``.

    The movers are controlled by impedance controllers. The c-rotation of a mover attached to a connection of the Gripper MagBot is
    controlled by the MagBot (coupling of the mover rotations with the gripper rotations and the finger, see
    ``GripperMagBotsAPM4330.control_gripper()``), the c-rotations of all other movers by the impedance controllers. The rotation control
    is switched when a mover is attached or detached.

    The scene can be reset (and randomized) without reloading the model using ``reset_scene()``, e.g. from the reset callback of the RL
    variant of this environment (``ReconfigurationRLEnv``), which inherits from this class.

    At initialization, the full model XML (``model_xml_str``) is saved next to this script, e.g. ``reconfiguration_env_gripper.xml`` (see
    ``model_xml_path``), e.g. to inspect the generated model or to load it in the MuJoCo viewer. The file is overwritten by every new
    environment instance with the same MagBot type (incl. the RL variants of this environment).

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
    :param num_movers: the number of movers, defaults to None, i.e. the number of connections of the chosen MagBot type. Must not be less
        than the number of connections of the chosen MagBot type.
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
    # station type and number of connections of each MagBot type
    STATION_CLASSES: dict[str, type[Station]] = {'gripper': GripperStation, 'sixD_platform': PlatformStation}
    NUM_CONNECTIONS = {'gripper': GripperMagBotsAPM4330.NUM_CONNECTIONS, 'sixD_platform': SixDPlatformMagBotsAPM4330.NUM_CONNECTIONS}
    DEFAULT_STATION_POSES = np.array([[0.480, 0.720, 0.01, np.deg2rad(-90)]])  # single station at the top extrusion
    # MuJoCo options of the Gripper MagBot (set by GripperMagBotsAPM4330), which are also used for the 6D-Platform MagBot
    OPTION_XML_STR = (
        '\t<option timestep="0.001" cone="elliptic" jacobian="dense" gravity="0 0 -9.81" '
        'integrator="implicitfast"><flag diagexact="enable"/></option>'
    )
    # joint masks of the impedance controllers: all DoFs, or all DoFs except the c-rotation (mover attached to a Gripper MagBot)
    JOINT_MASK_FREE = np.array([1, 1, 1, 1, 1, 1])
    JOINT_MASK_GRIPPER_ATTACHED = np.array([1, 1, 1, 1, 1, 0])

    def __init__(
        self,
        magbot_types: Sequence[str] = MAGBOT_TYPES,
        num_movers: int | None = None,
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
        self.num_connections = self.NUM_CONNECTIONS[self.magbot_type]
        mover_count = self.num_connections if num_movers is None else int(num_movers)
        assert mover_count >= self.num_connections, f'the {self.magbot_type} MagBot requires at least {self.num_connections} movers'

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

        # MagBots (all movers carry a mover adapter, the movers are attached to the connections at runtime)
        self.num_magbots = 1  # <-- number of MagBots in this environment
        # hidden collision proxies of the rail mounts and the station pins
        self.station_proxies = StationCollisionProxies(tol=station_proxy_tol)
        self.magbots: GripperMagBotsAPM4330 | SixDPlatformMagBotsAPM4330
        if self.magbot_type == 'gripper':
            self.magbot_params = GripperMagBotsAPM4330Params()
            self.magbots = GripperMagBotsAPM4330(
                num_magbots=self.num_magbots,
                num_movers=mover_count,
                magbot_params=self.magbot_params,
                cycle_time=0.001,  # timestep of the MuJoCo model (specified by the MagBot)
                station_proxies=self.station_proxies,
            )
        else:
            self.magbots = SixDPlatformMagBotsAPM4330(
                num_magbots=self.num_magbots, num_movers=mover_count, station_proxies=self.station_proxies
            )

        # initial mover positions (movers free on the table); None: sampled in reset_scene()
        if isinstance(initial_mover_xy_pos, dict):
            assert set(initial_mover_xy_pos.keys()) <= set(self.MAGBOT_TYPES), f'unknown MagBot type, valid types: {self.MAGBOT_TYPES}'
            initial_mover_xy_pos = initial_mover_xy_pos.get(self.magbot_type, None)
        assert initial_mover_xy_pos is None or initial_mover_xy_pos.shape == (mover_count, 2), (
            f'initial_mover_xy_pos must have the shape ({mover_count}, 2) for {mover_count} movers'
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
        # shape: (num_stations, num_connections, 6), [:, idx_c, :]: pose of the mover docked at connection idx_c
        self.station_mover_poses = np.array([station.get_mover_poses(z_m=self.initial_mover_z_pos) for station in self.stations])

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
        # whether a mover carries the MagBot or not, i.e. the same rotational stiffness can be used in both cases. The c-rotations of the
        # movers attached to a Gripper MagBot are controlled by the MagBot, i.e. the c-rotation is removed from the joint mask while a
        # mover is attached (see _set_mover_rotation_control()).
        self.impedance_controllers = [
            MoverImpedanceController(
                model=self.model,
                mover_joint_name=self.mover_joint_names[mover_idx],
                mover_half_height=self.mover_size[mover_idx, 2],
                joint_mask=self.JOINT_MASK_FREE.copy(),
                translational_stiffness=np.array([5000.0, 5000.0, 10.0]),
                rotational_stiffness=np.array([5.0, 5.0, 5.0]),
            )
            for mover_idx in range(self.num_movers)
        ]

        # reload model to add MagBots and actuators (includes reset_scene(), i.e. the start positions are sampled if not specified)
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

    def _set_mover_rotation_control(self, idx_mover: int, attached: bool) -> None:
        """Switch the control of the c-rotation of a mover: the c-rotation of a mover attached to a Gripper MagBot is controlled by the
        MagBot, the c-rotations of all other movers by the impedance controllers.

        :param idx_mover: the index of the mover
        :param attached: whether the mover is attached to a connection
        """
        gripper_attached = attached and isinstance(self.magbots, GripperMagBotsAPM4330)
        joint_mask = self.JOINT_MASK_GRIPPER_ATTACHED if gripper_attached else self.JOINT_MASK_FREE
        self.impedance_controllers[idx_mover].set_joint_mask(joint_mask.copy(), data=self.data)

    def _attach_mover(self, idx_mover: int, idx_magbot: int, idx_c: int) -> None:
        """Pick up a parked rail mount, i.e. weld the rail mount to the mover instead of the station and switch the rotation control of
        the mover.

        :param idx_mover: the index of the mover
        :param idx_magbot: the index of the MagBot
        :param idx_c: the index of the connection (the rail mount must be parked)
        """
        assert self.connections_parked[idx_magbot, idx_c]
        idx_station = self.connection_station_indices[idx_magbot, idx_c]
        self.data.eq_active[self.eq_ids_station_weld[idx_magbot, idx_c, idx_station]] = False
        self.magbots.attach_mover(data=self.data, idx_mover=idx_mover, idx_magbot=idx_magbot, idx_c=idx_c)
        self.connection_station_indices[idx_magbot, idx_c] = NO_INDEX
        self._set_mover_rotation_control(idx_mover=idx_mover, attached=True)

    def _detach_mover(self, idx_magbot: int, idx_c: int, idx_station: int) -> None:
        """Put back a carried rail mount, i.e. weld the rail mount to the station (in its current pose) instead of the mover and switch the
        rotation control of the mover.

        :param idx_magbot: the index of the MagBot
        :param idx_c: the index of the connection (the rail mount must be carried by a mover)
        :param idx_station: the index of the station
        """
        self._set_station_weld_relpose(idx_magbot=idx_magbot, idx_c=idx_c, idx_station=idx_station)
        self.data.eq_active[self.eq_ids_station_weld[idx_magbot, idx_c, idx_station]] = True
        idx_mover = self.magbots.detach_mover(data=self.data, idx_magbot=idx_magbot, idx_c=idx_c)
        self.connection_station_indices[idx_magbot, idx_c] = idx_station
        self._set_mover_rotation_control(idx_mover=idx_mover, attached=False)

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
        # the MagBot-specific branches check the type of self.magbots (instead of self.magbot_type), so that type checkers can narrow it
        if isinstance(self.magbots, GripperMagBotsAPM4330):
            # the gripper is located in the middle between the movers of connections 0 and 1 and its yaw is given by the direction from
            # connection 0 to connection 1. Connection 0 is located half the initial m0-m1 distance next to the gripper.
            station_xy_pos_c0, station_xy_pos_c1 = station_mover_xy_pos[:, 0], station_mover_xy_pos[:, 1]
            station_c0c1_xy_dirs = station_xy_pos_c1 - station_xy_pos_c0
            station_magbot_yaws = np.arctan2(station_c0c1_xy_dirs[:, 1], station_c0c1_xy_dirs[:, 0])
            c, s, d = np.cos(station_magbot_yaws), np.sin(station_magbot_yaws), self.initial_m0m1_xy_dist
            m0m1_xy_dists = np.stack([c * d[0] - s * d[1], s * d[0] + c * d[1]], axis=1)  # m0-m1 distance rotated by the MagBot yaw
            station_pos_xyz[:, :2] = 0.5 * (station_xy_pos_c0 + station_xy_pos_c1) - 0.5 * m0m1_xy_dists
            return self.magbots.generate_magbot_xml_strings(
                initial_pos_xyz_mover_c0=station_pos_xyz,
                custom_model_xml_strings=custom_model_xml_strings,
                initial_yaw=station_magbot_yaws,
            )
        # 6D-Platform MagBot: the MagBot base is located at rail mount 0 (connection 0, mover b) and the platform c-rotation is given by
        # the direction from mover a (connection 1) to mover b (connection 0)
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

        if self.render_mode is not None:
            self.viewer_collection.reload_model(self.model, self.data)

        self.reset_scene(mover_start_xy_pos=mover_start_xy_pos)

    ###################################################
    # Scene                                           #
    ###################################################

    @property
    def model_xml_path(self) -> Path:
        """The path of the file to which the full model XML is saved (see ``save_model_xml()``): next to this script, named after the
        MagBot type, e.g. ``reconfiguration_env_gripper.xml``.
        """
        return Path(__file__).resolve().with_name(f'reconfiguration_env_{self.magbot_type}.xml')

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
        """Reset the scene without reloading the model: MagBot i is parked in station i (optionally shifted along the station x-axis) and
        all movers are free on the table (not attached to any connection). The MuJoCo data, the welds of the rail mounts, the rotation
        control of the movers and the MagBot controller states are reset. This method can be reused by the reset callback of an RL
        variant of this environment.

        :param mover_start_xy_pos: the (x,y) start positions of the movers (numpy array of shape (num_movers,2)) [m], defaults to None.
            If None, the start positions are sampled on the free space of the table (see ``sample_mover_start_xy_pos()``).
        :param magbot_station_x_offsets: the offsets of the MagBots along the x-axis of their stations (numpy array of shape
            (num_magbots,)) [m], defaults to None. If None, the offsets are sampled if ``randomize_magbot_station_pos`` is True (see
            ``sample_magbot_station_x_offsets()``), otherwise they are zero.
        :param rng: the RNG used for sampling, defaults to None, i.e. ``self.rng``. An RL variant can pass ``self.np_random``.
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
        # all movers are free, i.e. their rotations are controlled by the impedance controllers
        for idx_mover in range(0, self.num_movers):
            self._set_mover_rotation_control(idx_mover=idx_mover, attached=False)

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

    def get_mover_poses_from_magbot_setpose(self, magbot_setposes: np.ndarray, single_track: bool | np.ndarray = False) -> np.ndarray:
        """Return the mover set poses resulting from MagBot set poses (inverse kinematics of the MagBot), incl. the yaw offsets of the
        docked movers (see ``MOVER_YAW_OFFSETS`` of the MagBots). The poses are returned in connection order, i.e. the poses of the movers
        attached to the connections (see ``connection_mover_indices``) have to be set to these poses, e.g.
        ``mover_setposes[env.connection_mover_indices.flatten()] = poses``.

        :param magbot_setposes: the desired MagBot set poses (using Euler angles (xyz) in rad). Gripper MagBot: gripper poses (numpy
            array of shape (num_magbots, 7)), the 7th entry is the desired distance between the two gripper fingers in m. 6D-Platform
            MagBot: platform poses (numpy array of shape (num_magbots, 6)).
        :param single_track: whether to calculate the mover set positions in single-track mode, defaults to False. Only used for the
            Gripper MagBot.
        :return: the mover set poses in connection order (numpy array of shape (num_magbots*num_connections, 6) using Euler angles (xyz)
            in rad)
        """
        if isinstance(self.magbots, GripperMagBotsAPM4330):
            mover_poses = self.magbots.gripperSetPose2MoverSetPose(
                gripper_pose_d=magbot_setposes, z_m=self.initial_mover_zpos, single_track=single_track
            )
        else:
            # [a, b] -> connection order [b, a] (the platform set poses are copied, since they are clipped in-place)
            mover_poses = self.magbots.platformSetPose2MoverSetPose(
                platform_pose_d=magbot_setposes.copy(), mover_z_d=float(self.initial_mover_zpos), use_euler=True
            )[:, ::-1].copy()
        mover_poses[:, :, 5] += self.magbots.MOVER_YAW_OFFSETS[np.newaxis, :]
        return mover_poses.reshape((-1, 6))

    def get_magbot_setpose_from_mover_poses(self, mover_poses: np.ndarray) -> np.ndarray:
        """Return the MagBot set poses resulting from mover set poses (forward kinematics of the MagBot). The yaw offsets of the docked
        movers (see ``MOVER_YAW_OFFSETS`` of the MagBots) are removed.

        :param mover_poses: the mover set poses in connection order (numpy array of shape (num_magbots*num_connections, 6) using Euler
            angles (xyz) in rad)
        :return: the MagBot set poses (using Euler angles (xyz) in rad). Gripper MagBot: gripper poses (numpy array of shape
            (num_magbots, 7)). 6D-Platform MagBot: platform poses (numpy array of shape (num_magbots, 6)).
        """
        mover_poses = mover_poses.reshape((self.num_magbots, self.num_connections, 6)).copy()
        mover_poses[:, :, 5] -= self.magbots.MOVER_YAW_OFFSETS[np.newaxis, :]
        if isinstance(self.magbots, GripperMagBotsAPM4330):
            return self.magbots.moverSetPose2GripperSetPose(mover_poses=mover_poses)
        # connection order [b, a] -> [a, b]
        return self.magbots.moverSetPose2PlatformSetPose(mover_poses=mover_poses[:, ::-1], use_euler=True)

    def _control_magbot(self, mover_setposes: np.ndarray) -> None:
        """Couple the MagBot mechanics with the attached movers (gripper rotations and fingers, or platform a,b-rotations).

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
        """Return the desired mover rotations for the impedance controllers. The c-rotations of the movers attached to a Gripper MagBot
        are controlled by the MagBot, i.e. the impedance controllers ignore the c-rotation of these movers (see
        ``_set_mover_rotation_control()``).

        :param mover_setposes: the desired mover poses (numpy array of shape (num_movers, 6) using Euler angles (xyz) in rad)
        :return: the desired mover rotations as quaternions (numpy array of shape (num_movers, 4))
        """
        return rotations_utils.vec_euler2quat(mover_setposes[:, 3:])

    def _apply_mover_controls(self, mover_setposes: np.ndarray, mover_quats_d: np.ndarray | None = None) -> None:
        """Calculate the controls of one control cycle, i.e. switch the welds of the rail mounts, update the mover impedance controllers
        and couple the MagBot mechanics with the movers. This method must be called before each MuJoCo integrator step and is also used
        by the RL variant of this environment.

        :param mover_setposes: the desired mover poses (numpy array of shape (num_movers, 6) using Euler angles (xyz) in rad)
        :param mover_quats_d: the desired mover rotations (numpy array of shape (num_movers, 4)), defaults to None, i.e. they are
            computed from ``mover_setposes`` (see ``_get_mover_quats_d()``)
        """
        quats_d = self._get_mover_quats_d(mover_setposes=mover_setposes) if mover_quats_d is None else mover_quats_d
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
        self._control_magbot(mover_setposes=mover_setposes)

    def step(
        self,
        mover_setposes: np.ndarray,
        num_cycles: int = 40,
        render_every_cycle: bool = False,
    ) -> dict[str, Any]:
        """Perform one or multiple simulation steps, including caluclation of mover controls, MuJoCo integrator steps, rendering, and
        collision checking. The remaining cycles are skipped if a collision is detected.

        :param mover_setposes: The desired mover poses (numpy array of shape (num_movers, 6) using Euler angles (xyz) in rad),
            [idx_mover,:]: set pose of mover idx_mover. The movers are controlled directly instead of via the MagBot set poses, e.g. to
            move the movers to a MagBot in the station (see ``get_mover_poses_from_magbot_setpose()`` to control the MagBot). The yaws
            of movers attached to a Gripper MagBot must be continuous (not wrapped).
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
    # The demo loops through randomly initialized scenarios (MagBot type, MagBot position in the station and movers that carry the MagBot).
    # The environment contains more movers than the MagBot has connections. For each scenario, a random subset of the movers picks up the
    # MagBot from the station, takes it out of the station, puts it back and drives back to the initial positions, while the other movers
    # wait. Any mover can carry any connection; a mover docked at a connection that is entered from the other side is rotated by 180 deg.
    # After the put-back, the movers rotate back under the control of their impedance controllers. The model only depends on the MagBot
    # type, i.e. a new environment (and viewer window) is only created if the MagBot type changes. Otherwise, the scene is reset.
    single_track = np.array([False])  # Gripper MagBot only
    # MagBot types from which the type of each scenario is chosen randomly
    magbot_types = ('gripper', 'sixD_platform')
    # randomize the position of the MagBot in the station (shift along the station pins)
    randomize_magbot_station_pos = False

    # initial mover positions (movers free on the table, a row at the bottom of the table)
    initial_mover_xy_pos = {
        'gripper': np.array([[0.100, 0.100], [0.290, 0.100], [0.480, 0.100], [0.670, 0.100], [0.860, 0.100]]),  # [m]
        'sixD_platform': np.array([[0.130, 0.100], [0.373, 0.100], [0.617, 0.100], [0.860, 0.100]]),  # [m]
    }
    num_movers = {magbot_type: xy_pos.shape[0] for magbot_type, xy_pos in initial_mover_xy_pos.items()}
    # the movers that carry the MagBot first drive straight up to this lane, so that they do not hit the waiting movers
    lane_y = 0.28  # [m]

    # station at the top extrusion (pose of the station anchor: x, y, z, yaw)
    station_poses = np.array([[0.480, 0.720, 0.01, np.deg2rad(-90)]])

    # the movers enter and leave the station along the entry axis (y-axis of the station body frame) from this distance [m]
    entrance_dist = 0.145
    # the movers carry the MagBot 0.25 m out of the station (center of the tiles)
    magbot_path = np.array([0.25, 0.0])  # [m]
    # distance between the pre-station points and the staging points, at which the movers rotate in place [m]
    staging_dist = 0.15

    num_cycles = 20
    # movements defaults
    default_duration = 0.75
    default_hold_after_duration = 0.25

    def run_scenario(env: ReconfigurationEnv, rng: np.random.Generator) -> None:
        """Pick up the MagBot with a random subset of the movers, take it out of the station, put it back and drive the movers back to
        their initial positions.
        """
        station = env.stations[0]
        rot_station = station.rotation_matrix_xy()
        # nominal mover poses in the station (connection order); the MagBot is put back at these poses
        station_mover_poses = env.station_mover_poses[0].copy()
        center_mover_poses = station_mover_poses.copy()
        center_mover_poses[:, :2] += rot_station @ magbot_path

        station_magbot_pose = env.get_magbot_setpose_from_mover_poses(mover_poses=station_mover_poses)
        center_magbot_pose = env.get_magbot_setpose_from_mover_poses(mover_poses=center_mover_poses)

        # offsets of the station entrances in the station body frame (x-axis: out of the station, y-axis: entry axis of the station),
        # rotated into the world frame; each mover enters from the flat side of its rail mount
        entrance_offsets = np.stack([np.zeros(env.num_connections), entrance_dist * station.MOVER_ENTRY_SIGNS], axis=1) @ rot_station.T

        def station_setposes(station_poses_c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Mover set poses (connection order) in the station, at the entrance of the station and at the pre-station point for the
            given mover poses in the station. The pre-station point is the entrance shifted by -200 mm in world y, from where the movers
            drive straight into the station.
            """
            entrance_poses_c = station_poses_c.copy()
            entrance_poses_c[:, :2] += entrance_offsets
            pre_station_poses_c = entrance_poses_c.copy()
            pre_station_poses_c[:, 1] -= 0.2  # [m]
            return station_poses_c, entrance_poses_c, pre_station_poses_c

        # put-back: nominal station poses
        station_mover_setposes, entrance_mover_setposes, pre_station_mover_setposes = station_setposes(station_mover_poses)

        # movers that carry the MagBot: a random subset of the movers, assigned to the connections in the order of the x-positions of
        # the pre-station points, so that the paths of the movers do not cross
        connection_movers = np.zeros(env.num_connections, dtype=int)
        connection_movers[np.argsort(pre_station_mover_setposes[:, 0])] = np.sort(
            rng.choice(env.num_movers, size=env.num_connections, replace=False)
        )
        print(f'  movers carrying connections {list(range(env.num_connections))}: {connection_movers.tolist()}')

        # all movers start with a yaw of 0, the movers that carry the MagBot rotate to the docking yaws on their way to the station
        initial_setposes = np.zeros((env.num_movers, 6))
        initial_setposes[:, :2] = env.mover_start_xy_pos  # start positions of the current scene
        initial_setposes[:, 2] = env.initial_mover_z_pos
        lane_poses_c = initial_setposes[connection_movers].copy()
        lane_poses_c[:, 1] = lane_y

        def setposes(poses_c: np.ndarray) -> np.ndarray:
            """Set poses of all movers: the movers that carry the MagBot at the given poses (connection order), all other movers at their
            initial poses.
            """
            mover_setposes = initial_setposes.copy()
            mover_setposes[connection_movers] = poses_c
            return mover_setposes

        dt = env.cycle_time * num_cycles  # [s] duration of one env step

        def interpolate(start: np.ndarray, end: np.ndarray, duration: float) -> list[np.ndarray]:
            """Linearly interpolate between two set poses, i.e. one set pose per env step."""
            return [start + s * (end - start) for s in np.linspace(0.0, 1.0, int(duration / dt))]

        def move_movers(
            start: np.ndarray, end: np.ndarray, duration: float = default_duration, hold_after_duration: float = default_hold_after_duration
        ) -> None:
            for poses_c in interpolate(start, end, duration=duration) + interpolate(end, end, duration=hold_after_duration):
                env.step(mover_setposes=setposes(poses_c), num_cycles=num_cycles)

        def move_magbot(
            start: np.ndarray, end: np.ndarray, duration: float = default_duration, hold_after_duration: float = default_hold_after_duration
        ) -> None:
            for magbot_setposes in interpolate(start, end, duration=duration) + interpolate(end, end, duration=hold_after_duration):
                poses_c = env.get_mover_poses_from_magbot_setpose(magbot_setposes=magbot_setposes, single_track=single_track)
                env.step(mover_setposes=setposes(poses_c), num_cycles=num_cycles)

        # pick-up: the movers drive to the real positions of the parked rail mounts
        pickup_poses_c = station_mover_poses.copy()
        pickup_poses_c[:, :2] = env.get_rail_mount_xy()
        pickup_mover_setposes, pickup_entrance_mover_setposes, pickup_pre_station_mover_setposes = station_setposes(pickup_poses_c)
        pickup_magbot_pose = env.get_magbot_setpose_from_mover_poses(mover_poses=pickup_poses_c)
        # staging points: the pre-station points shifted by staging_dist towards the station anchor (x), where the movers rotate in
        # place between a yaw of 0 and the docking yaws (the pre-station points of the outer connections are too close to the walls)
        staging_poses_c = pickup_pre_station_mover_setposes.copy()
        staging_poses_c[:, 0] += staging_dist * np.sign(station.pos[0] - staging_poses_c[:, 0])
        staging_yaw0_poses_c = staging_poses_c.copy()
        staging_yaw0_poses_c[:, 5] = 0.0

        # 1) the movers drive straight up to the lane, to the staging points, rotate to the docking yaws and drive to the pre-station
        #    points (200 mm in front of the station)
        move_movers(initial_setposes[connection_movers], lane_poses_c, duration=0.75)
        move_movers(lane_poses_c, staging_yaw0_poses_c)
        move_movers(staging_yaw0_poses_c, staging_poses_c, duration=1.0)
        move_movers(staging_poses_c, pickup_pre_station_mover_setposes, duration=0.75)
        # 2) the movers drive straight into the station (+y)
        move_movers(pickup_pre_station_mover_setposes, pickup_entrance_mover_setposes)
        # 3) the movers enter the station along the entry axis (rail mounts still welded to the station)
        move_movers(pickup_entrance_mover_setposes, pickup_mover_setposes)
        # 4) the movers take the MagBot and move it along the take-out axis to the center (rail mounts welded to the movers)
        move_magbot(pickup_magbot_pose, center_magbot_pose)
        print(f'  carried by movers: {env.connection_mover_indices[0].tolist()}')
        # 5) the movers move the MagBot back into the station (nominal station pose)
        move_magbot(center_magbot_pose, station_magbot_pose)
        # 6) the movers leave the station along the entry axis (rail mounts welded to the station)
        move_movers(station_mover_setposes, entrance_mover_setposes)
        # 7) the movers drive back to their pre-station positions
        move_movers(entrance_mover_setposes, pre_station_mover_setposes)
        # 8) the movers drive back to the staging points, rotate back to a yaw of 0 (impedance controllers) and drive back to the lane and
        #    to their initial positions
        move_movers(pre_station_mover_setposes, staging_poses_c, duration=0.75)
        move_movers(staging_poses_c, staging_yaw0_poses_c, duration=1.0)
        move_movers(staging_yaw0_poses_c, lane_poses_c)
        move_movers(lane_poses_c, initial_setposes[connection_movers], duration=0.75)

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
                    num_movers=num_movers[magbot_type],
                    initial_mover_xy_pos=initial_mover_xy_pos,
                    station_poses=station_poses,
                    randomize_magbot_station_pos=randomize_magbot_station_pos,
                    seed=idx_scenario,
                )
            print(
                f'scenario {idx_scenario}: {env.magbot_type} MagBot in a {type(env.stations[0]).__name__}, MagBot offset along the station '
                f'x-axis: {env.magbot_station_x_offsets * 1000} mm'
            )
            run_scenario(env, rng)
    except KeyboardInterrupt:
        pass
    finally:
        if env is not None:
            env.close()

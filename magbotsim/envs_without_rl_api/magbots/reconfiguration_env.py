from math import sin

import numpy as np
import mujoco
from magbotsim import (
    BasicMagBotEnv,
    MoverImpedanceController,
    GripperMagBotsAPM4330,
    GripperMagBotsAPM4330Params,
    GripperStation,
    StationCollisionProxies,
)


class ReconfigurationEnv(BasicMagBotEnv):
    """Simple benchmark environment for reconfiguration testing. At the beginning, the MagBot is in the station and the movers are
    free on the table at the positions ``initial_mover_xy_pos``.

    :param initial_mover_xy_pos: the initial (x, y) positions of the movers on the table (numpy array of shape (num_movers, 2)) [m]
    :param station_poses: the poses of the m2 mounts of the stations (numpy array of shape (num_stations, 4): (x, y, z) position in m
        and yaw in rad). The number of stations can differ from the number of MagBots, but there must be at least one station per
        MagBot, since MagBot i is generated in station i. The corresponding mover set positions (see ``GripperStation.get_mover_poses()``)
        are the positions at which the equality constraints of the rail mounts are switched: if a mover leaves a station position along
        the entry axis of the station, the rail mount is welded to the station; otherwise, the rail mount is welded to the mover.
        Defaults to None, i.e. a single station at the top extrusion.
    :param station_switch_tol: the maximum distance between a mover set position and the mover's station position at which the mover
        is considered to be at the station [m], defaults to 0.001
    :param station_proxy_tol: the sideways misalignment with which a mover (or a MagBot carried by movers) can enter a station
        without colliding [m], defaults to 0.002. The station switch is modelled by hidden collision proxies (rail mount proxies and
        station pins, see ``StationCollisionProxies``), which are shown in the viewer by pressing '3' (geom group 3). With 0, the proxies
        match the real geometries exactly, i.e. in practice they always collide.
    :param render_mode: the render mode, defaults to 'human'. Use None to run without rendering (e.g. in tests).
    """

    def __init__(
        self,
        initial_mover_xy_pos: np.ndarray,
        station_poses: np.ndarray,
        station_switch_tol: float = 0.001,
        station_proxy_tol: float = 0.002,
        render_mode: str | None = 'human',
    ) -> None:
        mover_count = 3

        # mover parameters, collision parameters and initial start positions
        mover_mass = np.array([1.264] * mover_count)
        bumper_mass = np.array([0.05] * mover_count)
        mover_params = {
            'shape': 'mesh',
            'mesh': {'mover_stl_path': ['beckhoff_apm4330_mover'] * mover_count, 'bumper_stl_path': ['beckhoff_apm4330_bumper'] * mover_count},
            'mass': mover_mass - bumper_mass,
            'bumper_mass': bumper_mass,
        }
        collision_params = {'shape': 'box', 'size': np.array([[0.155 / 2, 0.155 / 2]] * mover_count) + 0.001}

        # mover distance m0-m1 of the MagBot model (configuration in which the MagBot XML is generated)
        self.initial_m0m1_xy_dist = np.array([0.48, 0.0])  # [m]
        self.initial_mover_z_pos = 0.001

        # MagBots
        self.num_magbots = 1  # <-- number of MagBots in this environment
        self.indices_mover_c0 = np.array([0])  # <-- movers attached to connection 0
        self.indices_mover_c1 = np.array([1])  # <-- movers attached to connection 1
        self.indices_mover_c2 = np.array([2])  # <-- movers attached to connection 2
        self.magbot_params = GripperMagBotsAPM4330Params()
        # hidden collision proxies of the rail mounts and the station pins
        self.station_proxies = StationCollisionProxies(tol=station_proxy_tol)
        self.magbots = GripperMagBotsAPM4330(
            num_magbots=self.num_magbots,
            indices_mover_c0=self.indices_mover_c0,
            indices_mover_c1=self.indices_mover_c1,
            indices_mover_c2=self.indices_mover_c2,
            magbot_params=self.magbot_params,
            cycle_time=0.001,  # timestep of the MuJoCo model (specified by the MagBot)
            station_proxies=self.station_proxies,
        )

        # initial mover positions (movers free on the table)
        assert initial_mover_xy_pos.shape == (mover_count, 2)
        self.initial_mover_xy_pos = initial_mover_xy_pos.copy()

        assert station_poses.ndim == 2 and station_poses.shape[1] == 4
        self.num_stations = station_poses.shape[0]
        assert self.num_stations >= self.num_magbots, 'MagBot i is generated in station i, i.e. at least one station per MagBot required'
        self.stations = [
            GripperStation(pos=station_poses[idx_station, :3], yaw=station_poses[idx_station, 3], name=f'station_{idx_station}')
            for idx_station in range(0, self.num_stations)
        ]
        self.station_switch_tol = station_switch_tol

        # mover set poses at the stations (the equality constraints of the rail mounts are only switched at these positions)
        # shape: (num_stations, 3, 6), [:, idx_c, :]: pose of the mover attached to connection idx_c
        self.station_mover_poses = np.array([station.get_mover_poses(z_m=self.initial_mover_z_pos) for station in self.stations])

        # MagBot and connection to which each mover is attached
        self.mover_magbot_indices = np.zeros(mover_count, dtype=int)
        self.mover_connection_indices = np.zeros(mover_count, dtype=int)
        for idx_c, indices_mover in enumerate([self.indices_mover_c0, self.indices_mover_c1, self.indices_mover_c2]):
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
            initial_mover_start_xy_pos=self.initial_mover_xy_pos,
            custom_model_xml_strings=None,
            render_mode=render_mode,
            use_mj_passive_viewer=True,
        )

        # impedance controller
        # the rotational damping is computed from the rotational inertia of the movers, which keeps the controller stable regardless of
        # whether a mover carries the MagBot or not, i.e. the same rotational stiffness can be used in both cases
        self.impedance_controllers = [
            MoverImpedanceController(
                model=self.model,
                mover_joint_name=self.mover_joint_names[mover_idx],
                mover_half_height=self.mover_size[mover_idx, 2],
                joint_mask=np.array([1, 1, 1, 1, 1, 0]),
                translational_stiffness=np.array([5000.0, 5000.0, 10.0]),
                rotational_stiffness=np.array([5.0, 5.0, 0.0]),
                rotational_damping_from_inertia=True,
            )
            for mover_idx in range(self.num_movers)
        ]

        # reload model to add MagBots and actuators
        self.reload_model(mover_start_xy_pos=self.initial_mover_xy_pos)

    def _generate_extrusion_xml_str(self) -> str:
        """Generate the XML string for the aluminum extrusions surrounding the tiles. The extrusions are mounted directly at the tiles
        and arranged as a pinwheel, i.e. each side starts at the inner face of the previous side and fills the next corner
        (counterclockwise). Each side consists of (n-1) 240 mm extrusions and one 290 mm extrusion at the corner,
        where n is the number of tiles along that side. The open ends at the corners are closed by covers.

        :return: the XML string of the body containing the extrusions and covers
        """
        lengths = {240: 0.24, 290: 0.29}  # [m]
        profile_width = 0.05  # [m]
        profile_height = 0.1  # [m]
        z_bottom = -0.04  # [m]

        # outer edges of the tiles (= inner faces of the extrusions)
        active_x_pos = self.x_pos_tiles[self.layout_tiles == 1]
        active_y_pos = self.y_pos_tiles[self.layout_tiles == 1]
        x_min, x_max = np.min(active_x_pos) - self.tile_size[0], np.max(active_x_pos) + self.tile_size[0]
        y_min, y_max = np.min(active_y_pos) - self.tile_size[1], np.max(active_y_pos) + self.tile_size[1]
        num_tiles_x = int(np.round((x_max - x_min) / (2 * self.tile_size[0])))
        num_tiles_y = int(np.round((y_max - y_min) / (2 * self.tile_size[1])))

        # (start point at the inner face, direction, number of tiles) for each side (counterclockwise)
        sides = [
            (np.array([x_min, y_min]), np.array([1.0, 0.0]), num_tiles_x),  # bottom
            (np.array([x_max, y_min]), np.array([0.0, 1.0]), num_tiles_y),  # right
            (np.array([x_max, y_max]), np.array([-1.0, 0.0]), num_tiles_x),  # top
            (np.array([x_min, y_max]), np.array([0.0, -1.0]), num_tiles_y),  # left
        ]

        lines = ['\n\t\t<!-- aluminum extrusions and covers -->', '\t\t<body name="aluminum_extrusions" pos="0 0 0" gravcomp="1">']
        for idx_side, (start, direction, num_tiles) in enumerate(sides):
            outward = np.array([direction[1], -direction[0]])
            # extrusion mesh frame: x-axis = profile width (centered), y-axis = profile height (centered), z-axis = extrusion direction
            xyaxes_extrusion = f'{outward[0]:.0f} {outward[1]:.0f} 0 0 0 1'
            s = 0.0
            for idx_extrusion, length in enumerate([240] * (num_tiles - 1) + [290]):
                pos = start + (s + lengths[length]) * direction + 0.5 * profile_width * outward
                lines.append(
                    f'\t\t\t<geom name="alu_extrusion_{idx_side}_{idx_extrusion}" type="mesh" mesh="alu_extrusion_{length}_mesh" '
                    f'mass="0.2" pos="{pos[0]:.5f} {pos[1]:.5f} {z_bottom + 0.5 * profile_height:.5f}" '
                    f'xyaxes="{xyaxes_extrusion}" material="gray"/>'
                )
                s += lengths[length]
            # cover at the open end (cover mesh frame: x-axis = profile width, y-axis = thickness, z-axis = profile height)
            pos = start + s * direction
            lines.append(
                f'\t\t\t<geom name="cover_geom_{idx_side}" type="mesh" mesh="cover_mesh" mass="0.1" '
                f'pos="{pos[0]:.5f} {pos[1]:.5f} {z_bottom:.5f}" '
                f'xyaxes="{outward[0]:.0f} {outward[1]:.0f} 0 {direction[0]:.0f} {direction[1]:.0f} 0" material="black"/>'
            )
        lines.append('\t\t</body>')
        return '\n'.join(lines)

    def _custom_xml_string_callback(self, custom_model_xml_strings: dict[str, str] | None = None) -> dict[str, str]:
        """Add the MagBots and actuators for the movers to the MuJoCo model by modifying the ``custom_model_xml_strings``-dict.

        :param custom_model_xml_strings: the current ``custom_model_xml_strings``-dict which is modified by this callback, defaults to None
        :return: the modified ``custom_model_xml_strings``-dict
        """
        if custom_model_xml_strings is None:
            custom_model_xml_strings = {}
            custom_model_xml_strings.update({'custom_mover_body_xml_str_list': [None] * self.num_movers})

        if hasattr(self, 'mover_joint_names'):
            # generate custom model XML strings for the MagBots in the stations (MagBot i in station i). The pose of the MagBot is
            # deduced from the station mover poses of connections 0 and 1: the gripper is located in the middle between both movers and
            # its yaw is given by the direction from connection 0 to connection 1. Connection 0 is located half the initial m0-m1
            # distance next to the gripper.
            mover_qpos = self.get_mover_qpos(mover_names=self.mover_names, add_noise=False)
            station_pos_xyz_c0 = mover_qpos[self.indices_mover_c0, :3].copy()
            station_xy_pos_c0 = self.station_mover_poses[: self.num_magbots, 0, :2]
            station_xy_pos_c1 = self.station_mover_poses[: self.num_magbots, 1, :2]
            station_c0c1_xy_dirs = station_xy_pos_c1 - station_xy_pos_c0
            station_magbot_yaws = np.arctan2(station_c0c1_xy_dirs[:, 1], station_c0c1_xy_dirs[:, 0])
            c, s, d = np.cos(station_magbot_yaws), np.sin(station_magbot_yaws), self.initial_m0m1_xy_dist
            m0m1_xy_dists = np.stack([c * d[0] - s * d[1], s * d[0] + c * d[1]], axis=1)  # m0-m1 distance rotated by the MagBot yaw
            station_pos_xyz_c0[:, :2] = 0.5 * (station_xy_pos_c0 + station_xy_pos_c1) - 0.5 * m0m1_xy_dists
            custom_model_xml_strings = self.magbots.generate_magbot_xml_strings(
                initial_pos_xyz_mover_c0=station_pos_xyz_c0,
                custom_model_xml_strings=custom_model_xml_strings,
                initial_yaw=station_magbot_yaws,
            )

            # add actuators
            actuator_lines = ['\n\n\t<actuator>']
            for idx_mover in range(0, self.num_movers):
                actuator_lines.append(self.impedance_controllers[idx_mover].generate_actuator_xml_string(idx_mover=idx_mover))
            # MagBot gripper and mover c actuators
            actuator_lines.append(self.magbots.generate_actuator_xml_strings())
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
                # extrusions (STL files in mm, extrusion direction along the z-axis)
                '\n\t\t<!-- assets reconfiguration benchmark -->',
                '\t\t<mesh name="alu_extrusion_240_mesh" file="./misc/reconfiguration_benchmark/50x100_240.stl" '
                'scale="0.001 0.001 0.001"/>',
                '\t\t<mesh name="alu_extrusion_290_mesh" file="./misc/reconfiguration_benchmark/50x100_290.stl" '
                'scale="0.001 0.001 0.001"/>',
                '\t\t<mesh name="cover_mesh" file="./misc/gripper_application_example/cover.stl" scale="1 1 1"/>',
                # stations
                GripperStation.generate_assets_xml_str(),
            ]
            application_assets_xml_str = '\n'.join(application_assets_list)
            custom_assets_xml_str += application_assets_xml_str
            custom_model_xml_strings['custom_assets_xml_str'] = custom_assets_xml_str

            # center lines of the extrusions at the bottom and right side (supply platforms and product are placed relative to them)
            y_extrusion_bottom = np.min(self.y_pos_tiles[self.layout_tiles == 1]) - self.tile_size[1] - 0.025
            x_extrusion_right = np.max(self.x_pos_tiles[self.layout_tiles == 1]) + self.tile_size[0] + 0.025

            # z-pos of the rail mount origin of a parked MagBot (= z-pos of the mover adapter: mover z-pos + mover half height + 0.006)
            rail_mount_z = self.initial_mover_z_pos + float(np.max(self.mover_size[:, 2])) + 0.006

            custom_worldbody_xml_str = custom_model_xml_strings['custom_worldbody_xml_str']
            application_bodies_list = [
                # extrusions
                self._generate_extrusion_xml_str(),
                # stations
                *[station.generate_body_xml_str(proxies=self.station_proxies, rail_mount_z=rail_mount_z) for station in self.stations],
            ]
            application_bodies_xml_str = '\n'.join(application_bodies_list)
            custom_worldbody_xml_str += application_bodies_xml_str
            custom_model_xml_strings['custom_worldbody_xml_str'] = custom_worldbody_xml_str

        return custom_model_xml_strings

    def reload_model(self, mover_start_xy_pos: np.ndarray) -> None:
        """Generate a new model XML string with new start positions for movers.

        :param mover_start_xy_pos: None or a numpy array of shape (num_movers,2) containing the (x,y) starting positions of each mover
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
        # IDs of all station collision proxies (rail mount proxies and station pins)
        self.geom_ids_proxy = np.array(
            [
                idx_geom
                for idx_geom in range(0, self.model.ngeom)
                if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, idx_geom) or '').startswith(
                    StationCollisionProxies.PROXY_NAME_PREFIX
                )
            ],
            dtype=int,
        )

        # the MagBots are in their stations after reloading (MagBot i in station i), i.e. all rail mounts are welded to the stations
        # index of the station to which the rail mount of a mover is welded (-1: rail mount welded to the mover)
        self.mover_station_indices = self.mover_magbot_indices.copy()
        self.last_mover_pos_d = None
        for idx_mover in range(0, self.num_movers):
            idx_station = self.mover_station_indices[idx_mover]
            self._set_station_weld_relpose(idx_mover=idx_mover, idx_station=idx_station)
            self.data.eq_active[self.eq_ids_station_weld[idx_mover, idx_station]] = True
            self.data.eq_active[self.eq_ids_mover_weld[idx_mover]] = False

        # render the environment after reloading
        if self.render_mode is not None:
            self.viewer_collection.reload_model(self.model, self.data)
        self.render()

    def check_station_proxy_collision(self) -> bool:
        """Check whether a station collision proxy (rail mount proxy or station pin) is in contact with any other
        geom, e.g. because a mover or a MagBot entered a station with a sideways misalignment larger than the station proxy
        tolerance. If the station is entered correctly, the proxies are not in contact with anything.

        :return: True if a collision proxy is in contact with any other geom, False otherwise
        """
        contact_geoms = self.data.contact.geom[: self.data.ncon]
        return bool(np.isin(contact_geoms, self.geom_ids_proxy).any())

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
        """Switch the welds of the rail mounts between the movers and the stations. The welds are only switched if a mover leaves the
        station position of its connection at one of the stations, i.e. the last mover set position is within the station switch
        tolerance and the distance to the station position increases. The switch happens on the first step away from the station
        position, so that the rail mount does not jump. If the mover leaves along the entry axis of the station, the rail mount is
        welded to this station; otherwise, the rail mount is welded to the mover. Entering a station never switches the welds, since
        the decision only depends on the direction in which the mover leaves.

        :param mover_pos_d: the current mover set positions (numpy array of shape (num_movers, >=2))
        """
        if self.last_mover_pos_d is not None:
            for idx_mover in range(0, self.num_movers):
                # distances of the last and current set position to the closest station position of the mover's connection
                idx_c = self.mover_connection_indices[idx_mover]
                last_dists = np.linalg.norm(self.last_mover_pos_d[idx_mover, :2] - self.station_mover_poses[:, idx_c, :2], axis=1)
                idx_station = int(np.argmin(last_dists))
                last_dist = last_dists[idx_station]
                curr_dist = np.linalg.norm(mover_pos_d[idx_mover, :2] - self.station_mover_poses[idx_station, idx_c, :2])

                # entering: the mover moves into the station switch tolerance -> no switch
                # leaving: the mover was within the station switch tolerance and moves away from the station position -> switch

                is_leaving = last_dist < self.station_switch_tol and curr_dist > last_dist # = not entering
                if not is_leaving:
                    continue
                delta = mover_pos_d[idx_mover, :2] - self.last_mover_pos_d[idx_mover, :2]
                if np.linalg.norm(delta) < 1e-6: # ignore deltas close to numerical noise
                    continue
                to_station = self.stations[idx_station].is_along_entry_axis(delta_xy=delta)
                if to_station == (self.mover_station_indices[idx_mover] == idx_station):
                    continue

                if to_station:
                    self._set_station_weld_relpose(idx_mover=idx_mover, idx_station=idx_station)
                self.data.eq_active[self.eq_ids_station_weld[idx_mover, idx_station]] = to_station
                self.data.eq_active[self.eq_ids_mover_weld[idx_mover]] = not to_station
                self.mover_station_indices[idx_mover] = idx_station if to_station else -1

        self.last_mover_pos_d = mover_pos_d.copy()

    def get_mover_poses_from_gripper_setPose(
        self,
        gripper_setposes: np.ndarray,
        single_track: bool = False,
        ) ->np.ndarray:
        """Returns the mover poses resulting from a gripper set pose.

        :param gripper_setposes (np.ndarray): the desired target positions of the MagBot grippers (numpy array of shape (num_magbots, 7) using Euler
            angles (xyz) in rad). The 7th entry is the desired distance between the two gripper fingers in m. Ignored (can be None) if
            ``mover_setposes`` is specified.
        :param single_track (bool, optional): whether to calculate the mover set positions in single-track mode, defaults to False

        Returns:
            np.ndarray: _description_
        """
        mover_poses = self.magbots.gripperSetPose2MoverSetPose(
        gripper_pose_d=gripper_setposes, z_m=self.initial_mover_zpos, single_track=single_track
        )
        mover_poses = mover_poses.reshape((-1, 6))
        return mover_poses

    def step(
        self,
        mover_setposes: np.ndarray,
        num_cycles: int = 40,
        render_every_cycle: bool = False,
    ) -> None:
        """Perform one or multiple simulation steps, including caluclation of mover controls, MuJoCo integrator steps, rendering, and
        collision checking. The remaining cycles are skipped if a collision is detected.

        :param mover_setposes: The desired mover poses (numpy array of shape (num_movers, 6) using Euler angles (xyz) in rad). 
            The movers are controlled directly instead of via the gripper set poses, e.g. to move the movers away from a MagBot in the station.
        :param num_cycles: the number of control cycles for which to apply the same controls, defaults to 40
        :param render_every_cycle: whether to call ``render()`` after each integrator step in the ``step()`` method, defaults to
            False. Rendering every cycle leads to a smoother visualization of the scene, but can also be computationally expensive.
            Thus, this parameter provides the possibility to speed up training and evaluation. Regardless of this parameter, the scene
            is always rendered after ``num_cycles``
        """
        
        assert mover_setposes.shape == (self.num_movers, 6)

        collision = False
        for _ in range(0, num_cycles):
            # calculate controls
            
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
                    quat_d=np.array([1, 0, 0, 0]),
                    additional_mass=additional_mass,
                )
            # control gripper
            current_mover_qpos = self.get_mover_qpos(mover_names=self.mover_names, add_noise=False)
            current_mover_qvels = self.get_mover_qvel(mover_names=self.mover_names, add_noise=False)

            self.magbots.control_gripper(
                model=self.model,
                data=self.data,
                mover_poses_d=mover_setposes,
                current_mover_qpos=current_mover_qpos,
                current_mover_qvels=current_mover_qvels,
            )

            # integration
            mujoco.mj_step(self.model, self.data, nstep=1)
            # render every cycle for a smooth visualization of the movement
            if render_every_cycle:
                self.render()
            # check wall and mover collision every cycle to ensure that the collisions are detected and
            # all intermediate mover positions are valid and without collisions
            wall_collision = self.check_wall_collision(
                mover_names=self.mover_names,
                c_size=self.c_size,
                add_safety_offset=False,
                mover_qpos=None,
                add_qpos_noise=True,  # would also occur in a real system
            ).any()
            mover_collision = self.check_mover_collision(
                mover_names=self.mover_names,
                c_size=self.c_size,
                add_safety_offset=False,
                mover_qpos=None,
                add_qpos_noise=True,  # would also occur in a real system
            )
            station_proxy_collision = self.check_station_proxy_collision()
            collision = bool(mover_collision or wall_collision or station_proxy_collision)
            if collision:
                break

        self.render()


if __name__ == '__main__':
    single_track = np.array([False])

    # initial mover positions (movers free on the table)
    initial_mover_xy_pos = np.array(
        [[0.250, 0.150], 
         [0.710, 0.150],
         [0.480, 0.150], 
         ])  # [m]
    
    # station at the top extrusion (pose of the m2 mount: x, y, z, yaw)
    station_poses = np.array([[0.480, 0.720, 0.01, np.deg2rad(-90)]])
    env = ReconfigurationEnv(initial_mover_xy_pos=initial_mover_xy_pos, station_poses=station_poses)

    # hardcoded offsets in the station body frame (x-axis: out of the station, y-axis: entry axis of the station), rotated into the world frame
    entrance_offsets = np.array([[0.0, -0.145], [0.0, 0.145], [0.0, 0.145]])  # [m]
    
    # the movers carry the MagBot 0.25 m out of the station (center of the tiles)
    magbot_path = np.array([0.25, 0.0])  # [m]

    station = env.stations[0]

    rot_station = station.rotation_matrix_xy()
    station_mover_poses = station.get_mover_poses(z_m=env.initial_mover_z_pos)
    entrance_mover_poses = station_mover_poses.copy()
    entrance_mover_poses[:, :2] += entrance_offsets @ rot_station.T
    center_mover_poses = station_mover_poses.copy()
    center_mover_poses[:, :2] += rot_station @ magbot_path

    station_gripper_pose = env.magbots.moverSetPose2GripperSetPose(mover_poses=station_mover_poses[np.newaxis])
    center_gripper_pose = env.magbots.moverSetPose2GripperSetPose(mover_poses=center_mover_poses[np.newaxis])

    # mover set poses in the station, at the entrance of the station and at the initial positions
    station_mover_setposes = np.zeros((env.num_movers, 6))
    entrance_mover_setposes = np.zeros((env.num_movers, 6))
    for idx_c, indices_mover in enumerate([env.indices_mover_c0, env.indices_mover_c1, env.indices_mover_c2]):
        station_mover_setposes[indices_mover[0]] = station_mover_poses[idx_c]
        entrance_mover_setposes[indices_mover[0]] = entrance_mover_poses[idx_c]

    # the movers start with a yaw of 0 and rotate to the station yaw on their way to the entrance
    initial_mover_setposes = station_mover_setposes.copy()
    initial_mover_setposes[:, :2] = initial_mover_xy_pos
    initial_mover_setposes[:, 5] = 0.0

    # pre-station point: the station mover poses shifted by -200 mm in world y, from where the movers drive straight into the station
    pre_station_mover_setposes = entrance_mover_setposes.copy()
    pre_station_mover_setposes[:, 1] -= 0.2  # [m]

    num_cycles = 20
    dt = env.magbots.cycle_time * num_cycles  # [s] duration of one env step
    def interpolate(start: np.ndarray, end: np.ndarray, duration: float) -> list[np.ndarray]:
        """Linearly interpolate between two set poses, i.e. one set pose per env step."""
        return [start + s * (end - start) for s in np.linspace(0.0, 1.0, int(duration / dt))]

    # movements defaults
    default_duration = 1.25
    default_hold_after_duration = 0.25
    def move_movers(start: np.ndarray, end: np.ndarray, duration: float = default_duration, hold_after_duration: float = default_hold_after_duration) -> None:
        for mover_setposes in interpolate(start, end, duration=duration):
            env.step(mover_setposes=mover_setposes, num_cycles=num_cycles)
        for mover_setposes in interpolate(end, end, duration=hold_after_duration):
            env.step(mover_setposes=mover_setposes, num_cycles=num_cycles)

    def move_gripper(start: np.ndarray, end: np.ndarray, duration: float = default_duration, hold_after_duration: float = default_hold_after_duration) -> None:
        for gripper_setposes in interpolate(start, end, duration=duration):
            mover_setposes = env.get_mover_poses_from_gripper_setPose(gripper_setposes=gripper_setposes, single_track=single_track)
            env.step(mover_setposes=mover_setposes, num_cycles=num_cycles)

        for gripper_setposes in interpolate(end, end, duration=hold_after_duration):
            mover_setposes = env.get_mover_poses_from_gripper_setPose(gripper_setposes=gripper_setposes, single_track=single_track)
            env.step(mover_setposes=mover_setposes, num_cycles=num_cycles)

    try:
        while True:
            # move_movers(initial_mover_setposes, initial_mover_setposes)
            
            # # 1) the movers drive from their initial positions to the pre-station point (200 mm in front of the station)
            move_movers(initial_mover_setposes, pre_station_mover_setposes)

            # 2) the movers drive straight into the station (+y)
            move_movers(pre_station_mover_setposes, entrance_mover_setposes)

            # 3) the movers enter the station along the x-axis (rail mounts still welded to the station)
            move_movers(entrance_mover_setposes, station_mover_setposes)

            # 4) the movers take the MagBot and move it along the y-axis to the center (rail mounts welded to the movers)
            move_gripper(station_gripper_pose, center_gripper_pose)

            # 5) the movers move the MagBot back into the station. 
            move_gripper(center_gripper_pose, station_gripper_pose)

            # 6) the movers leave the station along the x-axis (rail mounts welded to the station)
            move_movers(station_mover_setposes, entrance_mover_setposes)

            # 7) the movers drive back to their pre station positions
            move_movers(entrance_mover_setposes, pre_station_mover_setposes)

            # 8) the movers drive back to their initial positions
            move_movers(pre_station_mover_setposes, initial_mover_setposes)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()

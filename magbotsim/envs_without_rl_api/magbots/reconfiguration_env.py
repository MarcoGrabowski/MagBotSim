import numpy as np
import mujoco
from magbotsim import BasicMagBotEnv, MoverImpedanceController, GripperMagBotsAPM4330, GripperMagBotsAPM4330Params


class ReconfigurationEnv(BasicMagBotEnv):
    """Simple benchmark environment for reconfiguration testing.

    :param initial_gripper_pose: the initial gripper poses of the MagBots (numpy array of shape (num_magbots, 7) using Euler angles
        (xyz) in rad). The 7th entry is the desired distance between the two gripper fingers in m. The mover start positions are
        calculated from these poses using the inverse kinematics of the MagBot.
    """

    # the origins of the station meshes are not centered, i.e. the parts are misaligned without these corrections
    STATION_M0M1_X_OFFSET = 0.05665  # [m] offset of the m0/m1 station mesh in its x-direction (mesh frame)
    STATION_M2_CORRECTION = 0.025  # [m] correction of the m2 station and m2 mount meshes in their y-direction (mesh frame)

    def __init__(self, initial_gripper_pose: np.ndarray) -> None:
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

        # mover distances of the MagBot model (configuration in which the MagBot XML is generated)
        self.initial_m0m1_xy_dist = np.array([0.48, 0.0])  # [m]
        self.initial_m0m2_xy_dist = np.array([0.24, -0.12])  # [m]
        self.initial_mover_z_pos = 0.001

        # MagBots
        self.num_magbots = 1  # <-- number of MagBots in this environment
        self.indices_mover_c0 = np.array([0])  # <-- movers attached to connection 0
        self.indices_mover_c1 = np.array([1])  # <-- movers attached to connection 1
        self.indices_mover_c2 = np.array([2])  # <-- movers attached to connection 2
        self.magbot_params = GripperMagBotsAPM4330Params()
        self.magbots = GripperMagBotsAPM4330(
            num_magbots=self.num_magbots,
            indices_mover_c0=self.indices_mover_c0,
            indices_mover_c1=self.indices_mover_c1,
            indices_mover_c2=self.indices_mover_c2,
            magbot_params=self.magbot_params,
            cycle_time=0.001,  # timestep of the MuJoCo model (specified by the MagBot)
        )

        # mover start positions
        assert initial_gripper_pose.shape == (self.num_magbots, 7)
        self.initial_gripper_pose = initial_gripper_pose
        self.initial_mover_start_xy_pos = self._get_initial_mover_start_xy_pos(gripper_pose=self.initial_gripper_pose)

        # init BasicMagBotEnv
        super().__init__(
            layout_tiles=np.ones((4, 3)),
            num_movers=mover_count,
            mover_params=mover_params,
            initial_mover_zpos=self.initial_mover_z_pos,
            table_height=0.2,
            collision_params=collision_params,
            initial_mover_start_xy_pos=self.initial_mover_start_xy_pos,
            custom_model_xml_strings=None,
            use_mj_passive_viewer=True,
        )

        # impedance controller
        self.impedance_controllers = [
            MoverImpedanceController(
                model=self.model,
                mover_joint_name=self.mover_joint_names[mover_idx],
                mover_half_height=self.mover_size[mover_idx, 2],
                joint_mask=np.array([1, 1, 1, 1, 1, 0]),
                translational_stiffness=np.array([150.0, 150.0, 10.0]),
                rotational_stiffness=np.array([5.0, 5.0, 0.0]),
            )
            for mover_idx in range(self.num_movers)
        ]

        # reload model to add MagBots and actuators
        self.reload_model(mover_start_xy_pos=self.initial_mover_start_xy_pos)

    def _get_initial_mover_start_xy_pos(self, gripper_pose: np.ndarray) -> np.ndarray:
        """Calculate the mover start positions from the gripper poses using the inverse kinematics of the MagBot.

        :param gripper_pose: the gripper poses (numpy array of shape (num_magbots, 7) using Euler angles (xyz) in rad)
        :return: the mover start positions (numpy array of shape (num_movers, 2))
        """
        # inverse kinematics (as in GripperMagBotsAPM4330.gripperSetPose2MoverSetPose(), which requires a compiled model)
        z_d = np.maximum(gripper_pose[:, 2], self.magbot_params.z_d_min)
        a_d = np.minimum(np.maximum(gripper_pose[:, 3], np.deg2rad(-180.0)), np.deg2rad(0.0))
        d_m01, d_m2p = self.magbots._get_desired_mover_distances(
            z_d=z_d, a_d=a_d, z_m=np.array([self.initial_mover_z_pos] * self.num_magbots), single_track=np.array([False] * self.num_magbots)
        )
        x_m0, y_m0, x_m1, y_m1, _, _ = self.magbots._get_desired_mover_xy_positions(
            x_d=gripper_pose[:, 0], y_d=gripper_pose[:, 1], a_d=a_d, c_d=gripper_pose[:, 5], d_m01=d_m01, d_m2p=d_m2p
        )

        # center the fixed mover configuration at the platform position (center between mover c0 and mover c1)
        xy_pos_c0 = 0.5 * np.stack([x_m0 + x_m1, y_m0 + y_m1], axis=1) - 0.5 * self.initial_m0m1_xy_dist
        mover_start_xy_pos = np.zeros((len(self.indices_mover_c0) * 3, 2))
        mover_start_xy_pos[self.indices_mover_c0] = xy_pos_c0
        mover_start_xy_pos[self.indices_mover_c1] = xy_pos_c0 + self.initial_m0m1_xy_dist
        mover_start_xy_pos[self.indices_mover_c2] = xy_pos_c0 + self.initial_m0m2_xy_dist
        return mover_start_xy_pos

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

    def _generate_station_xml_str(self, pos: np.ndarray, yaw: float = 0.0) -> str:
        """Generate the XML string for the station consisting of the m0, m1 and m2 station parts and the m2 mount. The parts have a fixed
        relative position to each other. The station m2 (including its mount) is used as the reference, i.e. it is located at the origin
        of the station body frame.

        :param pos: the (x, y, z) position of the station m2 in the world frame [m]
        :param yaw: the rotation of the whole station about the world z-axis [rad], defaults to 0.0
        :return: the XML string of the body containing the station
        """
        # positions of the m0 and m1 station relative to the m2 station (station body frame)
        # the m0/m1 station meshes are rotated by -90 deg, i.e. the mesh x-direction corresponds to the -y-direction of the station body
        rel_pos_m0 = np.array([0.0, 0.24452294 + self.STATION_M0M1_X_OFFSET, 0.0045])  # [m]
        rel_pos_m1 = np.array([0.0, -0.24452294 + self.STATION_M0M1_X_OFFSET, 0.0045])  # [m]
        rel_pos_m2 = np.array([0.0, self.STATION_M2_CORRECTION, 0.0])  # [m]

        lines = [
            '\n\t\t<!-- station -->',
            f'\t\t<body name="station" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" euler="0 0 {yaw:.7f}">',
            # m2 station (reference)
            f'\t\t\t<geom name="station_mount_m2_geom" type="mesh" mesh="station_mount_m2_mesh" mass="0.001" 'f'pos="{rel_pos_m2[0]:.5f} {rel_pos_m2[1]:.5f} {rel_pos_m2[2]:.5f}" material="gray"/>',
            f'\t\t\t<geom name="station_m2_geom" type="mesh" mesh="station_m2_mesh" mass="0.001" 'f'pos="{rel_pos_m2[0]:.5f} {rel_pos_m2[1]:.5f} {rel_pos_m2[2]:.5f}" material="black" contype="0" conaffinity="0"/>',
            # m0 station
            f'\t\t\t<geom name="station_m0_geom" type="mesh" mesh="station_m0m1_mesh" mass="0.001" 'f'pos="{rel_pos_m0[0]:.5f} {rel_pos_m0[1]:.5f} {rel_pos_m0[2]:.5f}" euler="0 0 -1.5707963" material="black" contype="0" conaffinity="0"/>',
            # m1 station
            f'\t\t\t<geom name="station_m1_geom" type="mesh" mesh="station_m0m1_mesh" mass="0.001" 'f'pos="{rel_pos_m1[0]:.5f} {rel_pos_m1[1]:.5f} {rel_pos_m1[2]:.5f}" euler="0 0 -1.5707963" material="black" contype="0" conaffinity="0"/>',
            '\t\t</body>',
        ]
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
            mover_qpos = self.get_mover_qpos(mover_names=self.mover_names, add_noise=False)
            # generate custom model XML strings for the MagBot
            custom_model_xml_strings = self.magbots.generate_magbot_xml_strings(
                initial_pos_xyz_mover_c0=mover_qpos[self.indices_mover_c0, :3], custom_model_xml_strings=custom_model_xml_strings
            )

            # add actuators
            actuator_lines = ['\n\n\t<actuator>']
            for idx_mover in range(0, self.num_movers):
                actuator_lines.append(self.impedance_controllers[idx_mover].generate_actuator_xml_string(idx_mover=idx_mover))
            # MagBot gripper and mover c actuators
            actuator_lines.append(self.magbots.generate_actuator_xml_strings())
            # join
            actuator_lines.append('\n\t</actuator>')
            custom_outworldbody_xml_str = custom_model_xml_strings.get('custom_outworldbody_xml_str', None)
            custom_model_xml_strings['custom_outworldbody_xml_str'] = custom_outworldbody_xml_str + ''.join(actuator_lines)

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
                '\t\t<mesh name="station_m0m1_mesh" file="./misc/reconfiguration_benchmark/station_m0m1.stl" scale="1 1 1"/>',
                '\t\t<mesh name="station_mount_m2_mesh" file="./misc/reconfiguration_benchmark/station_mount_m2.stl" scale="1 1 1"/>',
                '\t\t<mesh name="station_m2_mesh" file="./misc/reconfiguration_benchmark/station_m2.stl" scale="1 1 1"/>'
            ]
            application_assets_xml_str = '\n'.join(application_assets_list)
            custom_assets_xml_str += application_assets_xml_str
            custom_model_xml_strings['custom_assets_xml_str'] = custom_assets_xml_str

            # center lines of the extrusions at the bottom and right side (supply platforms and product are placed relative to them)
            y_extrusion_bottom = np.min(self.y_pos_tiles[self.layout_tiles == 1]) - self.tile_size[1] - 0.025
            x_extrusion_right = np.max(self.x_pos_tiles[self.layout_tiles == 1]) + self.tile_size[0] + 0.025

            custom_worldbody_xml_str = custom_model_xml_strings['custom_worldbody_xml_str']
            application_bodies_list = [
                # extrusions
                self._generate_extrusion_xml_str(),
                # station
                self._generate_station_xml_str(pos=np.array([0.480, 0.720, 0.01]), yaw=np.deg2rad(-90)),
            ]
            application_bodies_xml_str = '\n'.join(application_bodies_list)
            custom_worldbody_xml_str += application_bodies_xml_str
            custom_model_xml_strings['custom_worldbody_xml_str'] = custom_worldbody_xml_str

        return custom_model_xml_strings

    def reload_model(self, mover_start_xy_pos: np.ndarray) -> None:
        """Generate a new model XML string with new start positions for movers.

        :param mover_start_xy_pos: None or a numpy array of shape (num_movers,2) containing the (x,y) starting positions of each mover
        """
        assert np.allclose(mover_start_xy_pos[1, :] - mover_start_xy_pos[0, :], self.initial_m0m1_xy_dist)
        assert np.allclose(mover_start_xy_pos[2, :] - mover_start_xy_pos[0, :], self.initial_m0m2_xy_dist)
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

        # render the environment after reloading
        if self.render_mode is not None:
            self.viewer_collection.reload_model(self.model, self.data)
        self.render()

    def step(
        self, gripper_setposes: np.ndarray, single_track: np.ndarray | bool, num_cycles: int = 40, render_every_cycle: bool = False
    ) -> None:
        """Perform one or multiple simulation steps, including caluclation of mover controls, MuJoCo integrator steps, rendering, and
        collision checking.

        :param gripper_setposes: the desired target positions of the MagBot grippers (numpy array of shape (num_magbots, 7) using Euler
            angles (xyz) in rad). The 7th entry is the desired distance between the two gripper fingers in m.
        :param single_track: whether to calculate the mover set positions in single-track mode, defaults to False
        :param num_cycles: the number of control cycles for which to apply the same controls, defaults to 40
        :param render_every_cycle: whether to call ``render()`` after each integrator step in the ``step()`` method, defaults to
            False. Rendering every cycle leads to a smoother visualization of the scene, but can also be computationally expensive.
            Thus, this parameter provides the possibility to speed up training and evaluation. Regardless of this parameter, the scene
            is always rendered after ``num_cycles``.
        """
        assert gripper_setposes.shape == (self.num_magbots, 7)

        for _ in range(0, num_cycles):
            # calculate controls
            # gripper set pose to mover set pose
            mover_poses = self.magbots.gripperSetPose2MoverSetPose(
                gripper_pose_d=gripper_setposes, z_m=self.initial_mover_zpos, single_track=single_track
            )
            mover_poses = mover_poses.reshape((-1, 6))
            # update controls for mover impedance controllers
            for idx_mover in range(0, self.num_movers):
                self.impedance_controllers[idx_mover].update(
                    model=self.model,
                    data=self.data,
                    pos_d=mover_poses[idx_mover, :3],
                    quat_d=np.array([1, 0, 0, 0]),
                    additional_mass=self.magbots.magbot_masses[0] / 2,
                )
            # control gripper
            current_mover_qpos = self.get_mover_qpos(mover_names=self.mover_names, add_noise=False)
            current_mover_qvels = self.get_mover_qvel(mover_names=self.mover_names, add_noise=False)

            self.magbots.control_gripper(
                model=self.model,
                data=self.data,
                mover_poses_d=mover_poses,
                current_mover_qpos=current_mover_qpos,
                current_mover_qvels=current_mover_qvels,
            )

            # print("mover poses", mover_poses)
            # print("diff m2-m1", mover_poses[2] - mover_poses[1])
            # print("diff m2-m0", mover_poses[2] - mover_poses[0])

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
            if mover_collision or wall_collision:
                break
        self.render()


if __name__ == '__main__':
    magbot_params = GripperMagBotsAPM4330Params()

    v_x = 0.0   # m/s
    v_y = 0.0001 # m/s

    z_offset = (magbot_params.z_p_max - 0.01 - magbot_params.z_p_min) / 2
    fR = 0.08  # radius of the circle
    fTime = 0.0
    single_track = np.array([False])

    initial_z_pos = z_offset + magbot_params.z_p_min + magbot_params.n + 0.001
    # the 7th entry is the desired distance between the two gripper fingers in m
    finger_distance = 0.0

    gripper_offset = np.array([0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0])
    gripper_setposes = np.array(
        [
            [0.480, 0.6349, 0.4057, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), finger_distance],  # default mode
        ]
    ) 

    env = ReconfigurationEnv(initial_gripper_pose=gripper_setposes - gripper_offset)

    num_cycles = 20
    try:
        while True:
            # gripper_setposes[:, 0] = gripper_setposes[:, 0] + v_x * fTime
            # gripper_setposes[:, 1] = gripper_setposes[:, 1] + v_y * fTime
            # gripper_setposes[:, 2] = gripper_setposes[:, 2]

            env.step(gripper_setposes=gripper_setposes, single_track=single_track, num_cycles=num_cycles, render_every_cycle=False)
            fTime += 0.001 * num_cycles
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

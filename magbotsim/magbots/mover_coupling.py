import numpy as np
from mujoco import MjData, MjModel

# entry of an index array that does not refer to any object, e.g. a connection without a mover in ``connection_mover_indices`` or a
# rail mount carried by a mover in ``ReconfigurationEnv.connection_station_indices``. Use the boolean masks (``connections_carried``,
# ``ReconfigurationEnv.connections_parked``) or ``get_mover_attachments()`` instead of comparing with this value.
NO_INDEX = -1


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    """Wrap angles to (-pi, pi].

    :param angle: the angle(s) [rad]
    :return: the wrapped angle(s) [rad]
    """
    return np.angle(np.exp(1j * np.asarray(angle, dtype=np.float64)))


class MagBotMoverCoupling:
    """Mixin for MagBot classes that implements the coupling between the movers and the connections (rail mounts) of the MagBots.

    Each mover that can carry a rail mount has a mover adapter (body ``m_{idx_mover}_rail``), whose rail site
    (``m_{idx_mover}_rail_site``) can be welded to the rail mount site of any connection of any MagBot
    (``rail_mount_{idx_c}_site_{idx_magbot}``). For each pair of a mover with an adapter and a rail mount, the model contains a weld
    (``mover_weld_m{idx_mover}_c{idx_c}_{idx_magbot}``). At most one weld per rail mount and per mover is active, i.e. a rail mount is
    either carried by exactly one mover or not carried at all (e.g. parked in a station). The current assignment is stored in
    ``connection_mover_indices``.

    Two modes are supported, which use the same mechanism:

    In both modes, the number of movers in the environment is specified at initialization.

    - static: the movers attached to the connections are specified at initialization and are attached from the beginning (the
      corresponding welds are active in the model). Only these movers get an adapter, each with the adapter orientation of its
      connection, i.e. the movers have the same rotation as the MagBot, as in the kinematics of the MagBot (``MOVER_YAW_OFFSETS`` do not
      apply).
    - dynamic: all movers get the standard adapter orientation. No mover is
      attached at the beginning. Movers are attached to and detached from the connections at runtime (``attach_mover()``,
      ``detach_mover()``), e.g. to pick up a MagBot from a station and to put it back. A mover attached to a connection that is entered
      from the other side has a rotation offset of pi relative to the kinematics of the MagBot (see ``MOVER_YAW_OFFSETS``).

    Yaw conventions (rotations about the z-axis, all in connection order):

    - kinematic connection yaw: the mover c-rotation used in the kinematics of the MagBot (inverse and forward kinematics, coupling). In the
      default configuration of the MagBot, all kinematic connection yaws are equal to the c-rotation of the MagBot.
    - ``RAIL_MOUNT_YAW_OFFSETS``: kinematic connection yaw - yaw of the rail mount body (given by the body frames of the MagBot model)
    - ``RAIL_MOUNT_SITE_YAWS``: yaw of the rail mount site in the rail mount body frame
    - mover adapter yaw: yaw of the mover adapter body in the mover frame, either ``STANDARD_ADAPTER_YAW`` or ``FLIPPED_ADAPTER_YAW``. The
      rail site is located at ``MOVER_RAIL_SITE_YAW`` in the adapter frame, i.e. its yaw relative to the mover is 0 (standard adapter) or
      pi (flipped adapter).
    - the weld of mover i and connection k enforces: mover yaw = rail mount yaw + ``RAIL_MOUNT_SITE_YAWS[k]`` - rail site yaw of mover i,
      i.e. the yaw offset of the pair (mover yaw - kinematic connection yaw) is ``RAIL_MOUNT_SITE_YAWS[k]`` - rail site yaw of mover i -
      ``RAIL_MOUNT_YAW_OFFSETS[k]`` (see ``get_mover_yaw_offset()``).
    - ``MOVER_YAW_OFFSETS``: the yaw offsets of movers with the standard adapter, which are used in dynamic mode

    Subclasses define ``NUM_CONNECTIONS``, ``RAIL_MOUNT_YAW_OFFSETS``, ``RAIL_MOUNT_SITE_YAWS``, ``FLIPPED_ADAPTER_CONNECTIONS``
    (whether the connection is entered from the side of the flipped adapter), ``MOVER_WELD_ATTRS``, ``MOVER_EXCLUDE_BODIES`` and the XML
    of the mover adapter (``_mover_adapter_xml_str()``).
    """

    NUM_CONNECTIONS: int = 0
    RAIL_MOUNT_YAW_OFFSETS: np.ndarray = np.zeros(0)
    RAIL_MOUNT_SITE_YAWS: np.ndarray = np.zeros(0)
    FLIPPED_ADAPTER_CONNECTIONS: np.ndarray = np.zeros(0, dtype=bool)
    # additional attributes of the mover welds (e.g. solref and solimp)
    MOVER_WELD_ATTRS: str = ''
    # bodies of a mover that do not collide with the rail mounts ('{idx_mover}' is replaced by the index of the mover)
    MOVER_EXCLUDE_BODIES: tuple[str, ...] = ('m_{idx_mover}_rail',)

    STANDARD_ADAPTER_YAW = np.pi / 2  # [rad] yaw of the standard mover adapter in the mover frame
    FLIPPED_ADAPTER_YAW = -np.pi / 2  # [rad] yaw of the flipped mover adapter in the mover frame
    MOVER_RAIL_SITE_YAW = -np.pi / 2  # [rad] yaw of the rail site in the mover adapter frame

    @classmethod
    def mover_yaw_offsets(cls) -> np.ndarray:
        """Return the yaw offsets (mover yaw - kinematic connection yaw) of movers with the standard adapter, i.e. the offsets used in
        dynamic mode (0 or pi).

        :return: the yaw offsets in connection order (numpy array of shape (NUM_CONNECTIONS,)) [rad]
        """
        rail_site_yaw = cls.STANDARD_ADAPTER_YAW + cls.MOVER_RAIL_SITE_YAW
        return np.abs(wrap_angle(cls.RAIL_MOUNT_SITE_YAWS - rail_site_yaw - cls.RAIL_MOUNT_YAW_OFFSETS))

    @staticmethod
    def mover_weld_name(idx_mover: int, idx_c: int, idx_magbot: int) -> str:
        """Return the name of the weld between the adapter of a mover and the rail mount of a connection.

        :param idx_mover: the index of the mover
        :param idx_c: the index of the connection
        :param idx_magbot: the index of the MagBot
        :return: the name of the weld
        """
        return f'mover_weld_m{idx_mover}_c{idx_c}_{idx_magbot}'

    def _init_mover_coupling(self, num_magbots: int, num_movers: int, connection_mover_indices: np.ndarray | None) -> None:
        """Initialize the coupling in static mode (``connection_mover_indices`` specified) or dynamic mode (``connection_mover_indices`` is
        None).

        :param num_magbots: the number of MagBots
        :param num_movers: the number of movers in the environment
        :param connection_mover_indices: static mode: the indices of the movers attached to the connections (numpy array of shape
            (num_magbots, NUM_CONNECTIONS)), only these movers get an adapter. Dynamic mode: None, all movers get an adapter.
        """
        self.num_movers = num_movers
        self.dynamic_mover_coupling = connection_mover_indices is None
        if connection_mover_indices is None:
            self.mover_adapter_yaws = np.full(num_movers, self.STANDARD_ADAPTER_YAW)
            self.initial_connection_mover_indices = np.full((num_magbots, self.NUM_CONNECTIONS), NO_INDEX, dtype=int)
        else:
            connection_mover_indices = np.asarray(connection_mover_indices, dtype=int)
            assert connection_mover_indices.shape == (num_magbots, self.NUM_CONNECTIONS)
            assert len(np.unique(connection_mover_indices)) == connection_mover_indices.size, 'each mover can only carry one rail mount'
            self.mover_adapter_yaws = np.full(num_movers, np.nan)
            for idx_c in range(0, self.NUM_CONNECTIONS):
                adapter_yaw = self.FLIPPED_ADAPTER_YAW if self.FLIPPED_ADAPTER_CONNECTIONS[idx_c] else self.STANDARD_ADAPTER_YAW
                self.mover_adapter_yaws[connection_mover_indices[:, idx_c]] = adapter_yaw
            self.initial_connection_mover_indices = connection_mover_indices.copy()

        # movers with an adapter, i.e. movers that can carry a rail mount
        self.adapter_mover_indices = np.where(~np.isnan(self.mover_adapter_yaws))[0]
        # yaw offsets (mover yaw - kinematic connection yaw) of each pair of a mover and a connection, [idx_mover, idx_c] (NaN: no adapter)
        rail_site_yaws = self.mover_adapter_yaws + self.MOVER_RAIL_SITE_YAW
        self.pair_yaw_offsets = wrap_angle(
            self.RAIL_MOUNT_SITE_YAWS[np.newaxis, :] - rail_site_yaws[:, np.newaxis] - self.RAIL_MOUNT_YAW_OFFSETS[np.newaxis, :]
        )
        self.pair_yaw_offsets[np.isnan(self.mover_adapter_yaws)] = np.nan
        # [idx_magbot, idx_c]: index of the mover attached to connection idx_c of MagBot idx_magbot (NO_INDEX: no mover attached, see
        # connections_carried)
        self.connection_mover_indices = self.initial_connection_mover_indices.copy()

    ###################################################
    # Attachments                                     #
    ###################################################

    @property
    def connections_carried(self) -> np.ndarray:
        """Whether a mover is attached to a connection (numpy array of shape (num_magbots, NUM_CONNECTIONS), bool)."""
        return self.connection_mover_indices != NO_INDEX

    def get_carried_connections(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the connections to which a mover is attached and the attached movers.

        :return: three numpy arrays of shape (num_carried,): the indices of the MagBots, the indices of the connections and the indices
            of the attached movers
        """
        indices_magbot, indices_c = np.nonzero(self.connections_carried)
        return indices_magbot, indices_c, self.connection_mover_indices[indices_magbot, indices_c]

    def get_mover_attachments(self) -> list[tuple[int, int] | None]:
        """Return the connection to which each mover is attached.

        :return: a list with one entry per mover: (index of the MagBot, index of the connection) if the mover is attached to a connection,
            None if the mover is free
        """
        attachments: list[tuple[int, int] | None] = [None] * self.num_movers
        for idx_magbot, idx_c, idx_mover in zip(*self.get_carried_connections(), strict=True):
            attachments[idx_mover] = (int(idx_magbot), int(idx_c))
        return attachments

    def get_mover_yaw_offset(self, idx_mover: int, idx_c: int) -> float:
        """Return the yaw offset of a mover attached to a connection, i.e. the mover yaw - the kinematic connection yaw (0 or pi).

        :param idx_mover: the index of the mover (must have an adapter)
        :param idx_c: the index of the connection
        :return: the yaw offset [rad]
        """
        offset = self.pair_yaw_offsets[idx_mover, idx_c]
        assert not np.isnan(offset), f'mover {idx_mover} has no adapter'
        return float(offset)

    def attach_mover(self, data: MjData, idx_mover: int, idx_magbot: int, idx_c: int) -> None:
        """Attach a mover to a connection, i.e. activate the weld between the mover adapter and the rail mount. The rail mount is aligned
        with the mover adapter (the weld snaps the rail mount to the adapter), i.e. the mover should be located at the rail mount with the
        docking yaw (see ``get_docking_mover_yaw()``). The yaw of the mover is not checked.

        :param data: mjData of the MuJoCo environment
        :param idx_mover: the index of the mover (must have an adapter and must not be attached)
        :param idx_magbot: the index of the MagBot
        :param idx_c: the index of the connection (no mover must be attached)
        """
        assert not np.isnan(self.mover_adapter_yaws[idx_mover]), f'mover {idx_mover} has no adapter'
        assert not self.connections_carried[idx_magbot, idx_c], f'connection {idx_c} of MagBot {idx_magbot} is already carried'
        assert not (self.connection_mover_indices == idx_mover).any(), f'mover {idx_mover} is already attached'
        data.eq_active[self.eq_ids_mover_weld[idx_mover, idx_magbot, idx_c]] = 1
        self.connection_mover_indices[idx_magbot, idx_c] = idx_mover
        self._on_mover_attached(idx_magbot=idx_magbot, idx_c=idx_c)

    def detach_mover(self, data: MjData, idx_magbot: int, idx_c: int) -> int:
        """Detach the mover from a connection, i.e. deactivate the weld between the mover adapter and the rail mount. The rail mount should
        be held otherwise, e.g. by a station.

        :param data: mjData of the MuJoCo environment
        :param idx_magbot: the index of the MagBot
        :param idx_c: the index of the connection (a mover must be attached)
        :return: the index of the detached mover
        """
        assert self.connections_carried[idx_magbot, idx_c], f'no mover is attached to connection {idx_c} of MagBot {idx_magbot}'
        idx_mover = int(self.connection_mover_indices[idx_magbot, idx_c])
        data.eq_active[self.eq_ids_mover_weld[idx_mover, idx_magbot, idx_c]] = 0
        self.connection_mover_indices[idx_magbot, idx_c] = NO_INDEX
        self._on_mover_detached(idx_magbot=idx_magbot, idx_c=idx_c)
        return idx_mover

    def reset_mover_coupling(self, data: MjData | None = None) -> None:
        """Reset the attachments to the initial attachments (static mode: the movers specified at initialization, dynamic mode: no mover
        attached) and reset the state of the MagBot controllers.

        :param data: mjData of the MuJoCo environment, defaults to None. If specified, the welds are activated according to the initial
            attachments (after ``mujoco.mj_resetData()``, the welds are already in this state).
        """
        self.connection_mover_indices = self.initial_connection_mover_indices.copy()
        if data is not None:
            data.eq_active[self.eq_ids_mover_weld[self.adapter_mover_indices]] = 0
            indices_magbot, indices_c, indices_mover = self.get_carried_connections()
            data.eq_active[self.eq_ids_mover_weld[indices_mover, indices_magbot, indices_c]] = 1
        self._on_mover_coupling_reset()

    def _on_mover_attached(self, idx_magbot: int, idx_c: int) -> None:
        """Hook that is called after a mover is attached to a connection (e.g. to reset controller states).

        :param idx_magbot: the index of the MagBot
        :param idx_c: the index of the connection
        """

    def _on_mover_detached(self, idx_magbot: int, idx_c: int) -> None:
        """Hook that is called after a mover is detached from a connection.

        :param idx_magbot: the index of the MagBot
        :param idx_c: the index of the connection
        """

    def _on_mover_coupling_reset(self) -> None:
        """Hook that is called after the attachments are reset (e.g. to reset controller states)."""

    ###################################################
    # MuJoCo data                                     #
    ###################################################

    def _update_cached_coupling_ids(self, model: MjModel) -> None:
        """Update the cached IDs of the rail mount bodies and the mover welds.

        :param model: mjModel of the MuJoCo environment
        """
        # [idx_magbot, idx_c]: rail mount body of connection idx_c
        self.rail_mount_body_ids = np.array(
            [
                [model.body(f'rail_mount_{idx_c}_{idx_magbot}').id for idx_c in range(0, self.NUM_CONNECTIONS)]
                for idx_magbot in range(0, self.num_magbots)
            ],
            dtype=np.int32,
        )
        # [idx_mover, idx_magbot, idx_c]: weld between the adapter of mover idx_mover and the rail mount (NO_INDEX: mover without adapter,
        # see adapter_mover_indices)
        self.eq_ids_mover_weld = np.full((self.num_movers, self.num_magbots, self.NUM_CONNECTIONS), NO_INDEX, dtype=np.int32)
        for idx_mover in self.adapter_mover_indices:
            for idx_magbot in range(0, self.num_magbots):
                for idx_c in range(0, self.NUM_CONNECTIONS):
                    self.eq_ids_mover_weld[idx_mover, idx_magbot, idx_c] = model.equality(
                        self.mover_weld_name(idx_mover=idx_mover, idx_c=idx_c, idx_magbot=idx_magbot)
                    ).id

    def get_rail_mount_yaws(self, data: MjData) -> np.ndarray:
        """Return the current yaws of the rail mount bodies. Requires ``update_cached_mujoco_data()``.

        :param data: mjData of the MuJoCo environment
        :return: the yaws (numpy array of shape (num_magbots, NUM_CONNECTIONS)) [rad]
        """
        xmat = data.xmat[self.rail_mount_body_ids].reshape((self.num_magbots, self.NUM_CONNECTIONS, 3, 3))
        return np.arctan2(xmat[:, :, 1, 0], xmat[:, :, 0, 0])

    def get_docking_mover_yaw(self, data: MjData, idx_mover: int, idx_magbot: int, idx_c: int) -> float:
        """Return the docking yaw of a mover at a connection, i.e. the yaw at which the adapter of the mover is aligned with the rail
        mount. Requires ``update_cached_mujoco_data()``.

        :param data: mjData of the MuJoCo environment
        :param idx_mover: the index of the mover (must have an adapter)
        :param idx_magbot: the index of the MagBot
        :param idx_c: the index of the connection
        :return: the docking yaw, wrapped to (-pi, pi] [rad]
        """
        kinematic_yaw = self.get_rail_mount_yaws(data)[idx_magbot, idx_c] + self.RAIL_MOUNT_YAW_OFFSETS[idx_c]
        return float(wrap_angle(kinematic_yaw + self.get_mover_yaw_offset(idx_mover=idx_mover, idx_c=idx_c)))

    def get_docking_mover_yaws(self, data: MjData) -> np.ndarray:
        """Return the docking yaws of the connections, i.e. the yaws at which the adapter of a mover with the standard adapter is aligned
        with the rail mounts. Requires ``update_cached_mujoco_data()``.

        :param data: mjData of the MuJoCo environment
        :return: the docking yaws, wrapped to (-pi, pi] (numpy array of shape (num_magbots, NUM_CONNECTIONS)) [rad]
        """
        kinematic_yaws = self.get_rail_mount_yaws(data) + self.RAIL_MOUNT_YAW_OFFSETS[np.newaxis, :]
        return wrap_angle(kinematic_yaws + self.mover_yaw_offsets()[np.newaxis, :])

    ###################################################
    # XML                                             #
    ###################################################

    def _mover_adapter_xml_str(self, idx_mover: int, adapter_yaw: float) -> str:
        """Generate the XML string of the mover adapter, which is added to the mover body. The adapter body (``m_{idx_mover}_rail``) must
        be rotated by ``adapter_yaw`` and contain the rail site (``m_{idx_mover}_rail_site``) at ``MOVER_RAIL_SITE_YAW``.

        :param idx_mover: the index of the mover
        :param adapter_yaw: the yaw of the adapter in the mover frame [rad]
        :return: the XML string
        """
        raise NotImplementedError

    def _add_mover_adapters_xml(self, custom_model_xml_strings: dict) -> None:
        """Add the mover adapters to the mover bodies of all movers with an adapter (``custom_mover_body_xml_str_list``).

        :param custom_model_xml_strings: the current ``custom_model_xml_strings``-dict which is modified by this method
        """
        mover_body_xml_strs = custom_model_xml_strings.get('custom_mover_body_xml_str_list')
        assert isinstance(mover_body_xml_strs, list), 'custom_mover_body_xml_str_list must be a list'
        assert len(mover_body_xml_strs) == self.num_movers, 'custom_mover_body_xml_str_list must contain an entry for each mover'
        for idx_mover in self.adapter_mover_indices:
            mover_body_xml_strs[idx_mover] = self._mover_adapter_xml_str(
                idx_mover=int(idx_mover), adapter_yaw=float(self.mover_adapter_yaws[idx_mover])
            )
        custom_model_xml_strings['custom_mover_body_xml_str_list'] = mover_body_xml_strs

    def _rail_mount_site_xml_str(self, idx_c: int, idx_magbot: int) -> str:
        """Generate the XML string of the rail mount site of a connection (welded to the rail sites of the mover adapters).

        :param idx_c: the index of the connection
        :param idx_magbot: the index of the MagBot
        :return: the XML string
        """
        return (
            f'<site name="rail_mount_{idx_c}_site_{idx_magbot}" pos="0 0 0" euler="0 0 {self.RAIL_MOUNT_SITE_YAWS[idx_c]:.9f}" '
            'size="0.001" type="sphere"/>'
        )

    def _mover_weld_xml_lines(self, idx_magbot: int) -> list[str]:
        """Generate the XML strings of the welds between all mover adapters and the rail mounts of a MagBot. Only the welds of the
        initially attached movers are active.

        :param idx_magbot: the index of the MagBot
        :return: a list of XML strings (one per weld)
        """
        lines = []
        for idx_mover in self.adapter_mover_indices:
            for idx_c in range(0, self.NUM_CONNECTIONS):
                active = 'true' if self.initial_connection_mover_indices[idx_magbot, idx_c] == idx_mover else 'false'
                lines.append(
                    f'\t\t<weld name="{self.mover_weld_name(idx_mover=idx_mover, idx_c=idx_c, idx_magbot=idx_magbot)}" '
                    f'site1="m_{idx_mover}_rail_site" site2="rail_mount_{idx_c}_site_{idx_magbot}"{self.MOVER_WELD_ATTRS} '
                    f'active="{active}"/>'
                )
        return lines

    def _mover_exclude_xml_lines(self, idx_magbot: int) -> list[str]:
        """Generate the XML strings of the contact excludes between all movers with an adapter and the rail mounts of a MagBot.

        :param idx_magbot: the index of the MagBot
        :return: a list of XML strings (one per exclude)
        """
        lines = []
        for idx_c in range(0, self.NUM_CONNECTIONS):
            for idx_mover in self.adapter_mover_indices:
                for body in self.MOVER_EXCLUDE_BODIES:
                    lines.append(f'\t\t<exclude body1="rail_mount_{idx_c}_{idx_magbot}" body2="{body.format(idx_mover=idx_mover)}"/>')
        return lines

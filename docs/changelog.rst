Changelog
=========

Unreleased
----------

General
^^^^^^^
1. Flexible coupling between movers and MagBots (``MagBotMoverCoupling``, used by ``GripperMagBotsAPM4330`` and
   ``SixDPlatformMagBotsAPM4330``):
    - Both MagBot classes require the number of movers in the environment (``num_movers``, second argument).
    - Dynamic mode (no mover indices): all movers carry the same mover adapter, and any mover can be attached to and detached from any
      connection (rail mount) at runtime (``attach_mover()``, ``detach_mover()``). A mover docked at a connection that is entered from
      the other side is rotated by pi (``MOVER_YAW_OFFSETS``).
    - Static mode (``indices_mover_c0..c2`` / ``indices_mover_a/b``): the given movers are attached at initialization using the same
      mechanism; the behavior is unchanged.
    - ``GripperMagBotsAPM4330.control_gripper()`` couples only the attached movers. The kinematic yaw of a connection without a mover is
      held, i.e. a parked gripper no longer moves when free movers rotate. The mover c-rotation actuators of the Gripper MagBot are named
      ``magbot_mover_actuator_c_{idx_mover}``, and the mover welds ``mover_weld_m{idx_mover}_c{idx_c}_{idx_magbot}``.
2. ``ReconfigurationEnv`` and its RL variants:
    - Support more movers than connections (``num_movers``). The movers are assigned to the connections at runtime when they pick up a
      rail mount (``connection_mover_indices``, ``connection_station_indices``). A mover is only attached if its adapter points in the
      direction of the rail mount (mover yaw within +/-90 deg of the docking yaw), since a mover rotated by pi fits through the jaws.
    - The c-rotation of a mover attached to a Gripper MagBot is controlled by the MagBot, the c-rotations of all other movers by the
      impedance controllers. The control is switched when a mover is attached or detached.
    - ``ReconfigurationRLEnv``: new observation layout (values per mover incl. a one-hot encoding of the attached connection, followed
      by values per connection), see ``_get_obs()``.
    - At initialization, the full model XML is saved next to ``reconfiguration_env.py``, e.g. ``reconfiguration_env_gripper.xml`` (see
      ``save_model_xml()``, ``model_xml_path``).
3. ``MoverImpedanceController.set_joint_mask()`` optionally sets the controls of disabled DoFs to zero (``data``) and no longer validates
   the joint mask.
4. New ``ReconfigurationTransportEnv`` (``envs_without_rl_api``), a standalone sibling of ``ReconfigurationEnv`` with the same tile
   layout, profile frame and station, two movers and one 6D-Platform MagBot:
    - Infinite transport task: the movers pick up the MagBot from the station, take products from the supply rack of the 6D-Platform
      application example and drop them into its bin by tilting the platform. The supply rack and the bin are located next to each
      other at the bottom long side, i.e. the MagBot does not have to turn.
    - A product that rests in the bin is counted as delivered (``num_delivered_products``), a product that rests outside the bin below
      the platform heights as dropped (``num_dropped_products``). In both cases, it is respawned in the supply rack
      (``respawn_product()``). ``step()`` reports both events in the info dict (``product_delivered``, ``product_dropped``).
    - Scripted demo movements (``run_magbot_pickup()``, ``run_transport_cycle()``), used by the demo of the script and by the tests.

Pre-Release v1.4.0a4 (2026-11-12)
---------------------------------

General
^^^^^^^
1. Added optional, collision-enabled profile frames around exposed tile edges,
   including inner holes, through ``use_profile_frame``. The profile and cover tops are 0.065 m (6.5 cm) above the tile surface.
2. Updated ``StateBasedGlobalPushingEnv``:
    - Added H-, E-, and O-shaped objects and equilateral triangular prisms with configurable dimensions and mass.
    - Added random object type selection on reset from a configured list, with explicit reset overrides taking precedence.
    - Fixed pose reward calculations to use the geometry stored with each transition, independently of the current episode’s object type.

Bug Fixes
^^^^^^^^^
1. Fixed initialization of ``BasicMagBotMultiAgentEnv``.
2. Fixed some minor discontinuity issues in edge cases:
    - Numerical rounding issue for z_p_min (in ``GripperMagBotsAPM4330Params``)
    - Finger distance to angle calculation now became c2 smooth (in ``GripperMagBotsAPM4330``)
3. Fixed stale rendering resources in ``MujocoViewerCollection`` after model reloads, which could 
4. Fix damping calculation for rotational component together with corresponding env examples (in ``MoverImpedanceController``)
   cause missing objects or incorrectly displayed meshes.

Release v1.3.0 (2026-09-11)
---------------------------

General
^^^^^^^
1. Add Gripper MagBot including inverse kinematics controller
2. Update mujoco utils for compatibility with latest MuJoCo versions (3.12.0 and 3.13.0)

Release v1.2.0 (2026-07-18)
---------------------------

Bug Fixes
^^^^^^^^^
1. Fixed wrong joint position in 6D-Platform MagBot model.
2. Fixed wrong offset in 6D-Platform MagBot model.
3. Updated stl files for 6D-Platform MagBot model.

Release v1.1.0 (2026-05-08)
---------------------------

General
^^^^^^^
1. The ``_step_callback`` of the ``BasicMagBotSingleAgentEnv`` now returns the possibly modified action instead of None.
2. Added support for Python 3.14
3. The ``StateBasedGlobalPushingEnv`` and all related pushing environments, such as ``StateBasedPushBoxEnv``, ``StateBasedPushTEnv``,
   ``StateBasedPushLEnv``, and ``StateBasedPushXEnv`` now also support the rotation of the movers around the 
   z-axis (:math:`\gamma`-rotation) to establish more stable contacts with the object. It is possible, but not necessary, to specify 
   a limit for the :math:`\gamma`-rotation.

Performance
^^^^^^^^^^^
1. Added spatial hashing for mover-mover collision detection.
2. Precompute per-tile wall-segment lookup table for wall collision checks.
3. Added fast paths for uniform mover sizes and complete tile layouts in ``qpos_is_valid``.
4. Cached episode-level Shapely polygons in ``StateBasedGlobalPushingEnv`` to avoid rebuilding them on every step.
5. Removed ``scipy`` dependency and optimize rectangle vertex calculation in ``geometry_2D_utils``.

Bug Fixes
^^^^^^^^^
1. Fixed wrong mover bumper mass and corresponding examples in the documentation
2. Fixed wrong gainprm in all RL environments
3. Ensured correct maximum dynamics of the actions (RL environments). This is fixed by using the ``_step_callback``.

------------------------------------------------------------------------------------------------------------------------------------

Release v1.0.3 (2026-03-25)
---------------------------

General
^^^^^^^
1. Fixed bug in ``StateBasedGlobalPushingEnv-v0``
2. Updated RL examples

------------------------------------------------------------------------------------------------------------------------------------

Release v1.0.2 (2026-02-27)
---------------------------

General
^^^^^^^
1. Added the 6D-Platform MagBot (``SixDPlatformMagBotsAPM4330``)
2. Added tutorial on how to add MagBots to custom environments
3. Added example environments with MagBots (``SixDPlatformMagBotApplicationExampleEnv`` and ``SixDPlatformMagBotExampleEnv``)

------------------------------------------------------------------------------------------------------------------------------------

Release v1.0.1 (2025-11-24)
---------------------------

General
^^^^^^^
1. Used MuJoCo functions in ``MoverImpedanceController`` instead of ``scipy.spatial.transform.Rotation`` (better performance, but ``scipy.spatial.transform.Rotation`` is not completely replaced)
2. Added benchmark environments
3. Added ``StateBasedPushXEnv-v0`` and ``StateBasedPushLEnv-v0``
4. Updated throughput calculation in pushing environments and update benchmark results accordingly

------------------------------------------------------------------------------------------------------------------------------------

Release v1.0.0 (2025-11-08)
---------------------------

General
^^^^^^^
1.  Added the basic environments: ``BasicMagBotEnv``, ``BasicMagBotSingleAgentEnv``, ``BasicMagBotMultiAgentEnv``
2.  Added the following example environments for trajectory planning and object manipulation:
    ``LongHorizonGlobalTrajectoryPlanningEnv-v0``, ``StateBasedStaticObstaclePushingEnv-v0``, ``StateBasedGlobalPushingEnv-v0``,
    ``StateBasedPushBoxEnv-v0``, ``StateBasedPushTEnv-v0``
3.  Added a ``MoverImpedanceController`` that solves a position and orientation task
4.  Added the ``MujocoOffScreenViewer``: an extension of the Gymnasium OffScreenViewer that allows to also specify the groups
    of geoms to be rendered by the off-screen renderer
5.  Added the ``MujocoViewerCollection``: a manager for all renderers in a MuJoCo environment
6.  Added the ``Matplotlib2DViewer``: a simple viewer that displays the tile and mover configuration together with the mover
    collision offsets for debugging and analyzing planning tasks
7.  Added auxiliary functions for MuJoCo, collision checking and rotations
8.  Added benchmarks and latest results for trajectory planning and object pushing
9.  Added tutorials and examples on how to use MagBotSim

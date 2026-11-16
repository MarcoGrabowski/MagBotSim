.. _mover_coupling_api_reference:

Mover Coupling
==============

The coupling between the movers and the connections (rail mounts) of the MagBots is shared by all MagBot classes. In static mode, the movers
attached to the connections are specified at initialization. In dynamic mode, any mover can be attached to and detached from any
connection at runtime, e.g. to pick up a MagBot from a station and to put it back (see ``ReconfigurationEnv``).

.. autoclass:: magbotsim.magbots.mover_coupling.MagBotMoverCoupling
  :members:
  :no-index:

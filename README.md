# spmd-reflection-sim

Frequency-domain reflection/transfer simulation for SPMD mixing segments with optional Touchstone S-parameter nodes.

## Scope (initial)
- AC-domain solver (no LTspice dependency)
- Global S2P applied to all nodes
- RL/IL plots and optional CSV export
- JSON and CLI configuration

## Planned structure
- `spmd_reflection/` core package
- `cli.py` entry point
- `examples/` sample configs and notes

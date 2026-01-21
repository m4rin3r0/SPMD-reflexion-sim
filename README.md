# spmd-reflection-sim

Frequency-domain reflection/transfer simulation for SPMD mixing segments with Touchstone S-parameter nodes.

## Scope (v0)
- AC-domain solver (no LTspice dependency)
- Global S2P applied to all nodes
- RL/IL plots
- JSON + CLI configuration

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

Editable install (recommended for dev):
```bash
pip install -e .
```

## Quick start
```bash
python cli.py --json examples/basic.json --s2p /path/to/node.s2p
```

To save plots:
```bash
python cli.py --json examples/basic.json --s2p /path/to/node.s2p --plot results.png
```

## Configuration
See `examples/basic.json` for the minimal schema. The `s2p` field is required.

## Notes
This solver treats differential ports as a single port with `z0` (default 100 ohm).

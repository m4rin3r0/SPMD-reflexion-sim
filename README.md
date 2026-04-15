# spmd-reflection-sim

Frequency-domain reflection/transfer simulation for SPMD mixing segments with Touchstone S-parameter nodes.

## Scope (v0)
- AC-domain solver (no LTspice dependency)
- RX drop S2P used as shunt 1-port loads
- TX jumped S2P used as inline 2-port
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
python cli.py --json examples/basic.json --s2p /path/to/rx_drop.s2p --jumped-s2p /path/to/tx_jumped.s2p
```

To save plots:
```bash
python cli.py --json examples/basic.json --s2p /path/to/rx_drop.s2p --jumped-s2p /path/to/tx_jumped.s2p --plot results.png
```

## Configuration
See `examples/basic.json` for the minimal schema. The fields `s2p` and `jumped_s2p` are required.

## Notes
This solver treats the RX drop measurement as a one-port load using its differential `S11`.
The TX node is modeled as an inline differential two-port from the jumped measurement.

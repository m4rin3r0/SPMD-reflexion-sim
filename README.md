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
In addition to the global bus `S11`/`S21`, the plots include a local RL/IL evaluation for every RX drop on the mixing segment:
solid lines use the left side of the branch, dashed lines use the right side.
The RL plots also overlay the 10BASE-T1M compliance limit, using `N_UNIT = nodes`.

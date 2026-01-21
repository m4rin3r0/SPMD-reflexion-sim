# Design Draft (v0)

## Goal
- AC-domain solver in Python, no LTspice dependency
- Global `.s2p` applied to all nodes
- RL/IL plots (S11/S21)
- JSON + CLI configuration

## Architecture & Data Flow
1) `cli.py`
   - Parse CLI args (e.g. `--json`, `--s2p`, `--freq_start`, `--freq_stop`, `--npoints`).
   - Hand off to `config.py`.
2) `config.py`
   - Load defaults + JSON.
   - Validate required fields.
   - Produce canonical config.
3) `topology.py`
   - Build trunk, drops, nodes, terminations.
   - Provide node/port indices for solver.
4) `touchstone.py`
   - Parse S2P (frequency, S-matrix, Z0).
   - Interpolate to solver frequency grid.
   - Convert S -> Y for stamping.
5) `solver_ac.py`
   - Stamp all elements into a Y matrix per frequency.
   - Solve MNA system.
   - Compute S11/S21.
6) `plots.py`
   - RL/IL plots (dB vs f).
   - Optional CSV export.

## Modeling Assumptions (v0)
- Differential ports are treated as a single port with `Z0_diff` (default 100 ohm).
- Node S2P represents a 2-port between drop end and PHY port.
- Node load is assumed included in the S2P; no extra rnode/cnode.
- Drops remain explicit cable elements.

## Solver Approach
- Stamp 2-port Y-parameters into a global Y matrix.
- Use a Norton source with reference impedance `Z0` at the TX port.
- Derive S11/S21 from solved port voltages and currents.

## Cable Model (v0)
- Series impedance per meter:
  - `Z = rdc + rskin*sqrt(f) + j*2*pi*f*l`
- Shunt admittance per meter:
  - `Y = j*2*pi*f*c`
- Convert per-length Z/Y to ABCD, then to Y-parameters.

## JSON Schema (minimal)
```json
{
  "analysis": "ac",
  "freq_start": 1e5,
  "freq_stop": 4e7,
  "npoints": 400,
  "z0": 100,
  "nodes": 16,
  "length": 100,
  "attach_points": null,
  "drop_max": 0.02,
  "random_drop": false,
  "s2p": "path/to/node.s2p",
  "cable_model": {
    "rdc": 0.0094,
    "l": 20.6435e-9,
    "c": 2.25026e-12,
    "rskin": 1.134268e-5,
    "ref_length": 0.05
  },
  "termination": { "rterm": 100 }
}
```

## Risks / Open Points
- S2P format and Z0 need validation with real files.
- Mixed-mode vs single-ended conventions may require mapping.
- S21 normalization must be validated against a known reference.



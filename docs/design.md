# Design Draft (v0)

## Goal
- AC-domain solver in Python, no LTspice dependency
- RX drop `.s2p` used as one-port shunt load
- TX jumped `.s2p` used as inline two-port
- RL/IL plots (S11/S21)
- JSON + CLI configuration

## Architecture & Data Flow
1) `cli.py`
   - Parse CLI args (e.g. `--json`, `--s2p`, `--jumped-s2p`, `--freq_start`, `--freq_stop`, `--npoints`).
   - Hand off to `config.py`.
2) `config.py`
   - Load defaults + JSON.
   - Validate required fields.
   - Produce canonical config.
3) `topology.py`
   - Build trunk plus inline/shunt node placements.
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
- RX node S2P already represents the full measured drop and node under realistic conditions.
- RX nodes are stamped as one-port shunt admittances using the differential `S11`; no extra drop cable or termination is added.
- TX node is stamped as an inline two-port using the jumped measurement.

## Solver Approach
- Stamp trunk cable segments and the TX jumped two-port into a global Y matrix.
- Stamp RX nodes as shunt one-port admittances at the trunk attachment points.
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
  "s2p": "path/to/rx_drop.s2p",
  "jumped_s2p": "path/to/tx_jumped.s2p",
  "cable_model": {
    "rdc": 0.0094,
    "l": 20.6435e-9,
    "c": 2.25026e-12,
    "rskin": 1.134268e-5,
    "ref_length": 0.05
  }
}
```

## Risks / Open Points
- S2P format and Z0 need validation with real files.
- Mixed-mode vs single-ended conventions may require mapping.
- S21 normalization must be validated against a known reference.


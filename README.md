# spmd_reflection

Simulation tool for analyzing signal integrity in **10BASE-T1M Automotive Ethernet multi-drop bus systems** with Power over Data Line (PoDL).

---

## Overview

`spmd_reflection` builds a nodal admittance matrix model of a multi-drop bus from:
- **Measured S-parameters** of individual drop PCBs (via Touchstone files)
- **Analytical cable model** based on distributed RLC parameters
- **Bus topology** defined in a YAML configuration file

It computes return loss, insertion loss, and TCI quantities per IEEE 802.3da Clause 188/189, and checks conformance against the standard's mask functions.

---

## Features

- Nodal admittance matrix solver with Norton source excitation
- Drop PCBs modeled as 2-port networks (jumped measurement + external PHY load)
- IEEE 802.3da Clause 188 mask checks: TCI IL, TCI RL, Mixing Segment IL
- IEEE 802.3da Clause 189 mask checks: MPI IL, MPI RL (PoDL)
- TX sweep: every drop simulated as transmitter in turn
- Jupyter notebook for visualization and compliance reporting

---

## Project Structure

```
spmd_reflection/
├── topology/           # Bus topology (nodes, segments, drops, terminations)
├── cable/              # Distributed cable model (telegrapher's equations)
├── config/             # YAML config loading and validation
├── drop/               # Touchstone loading and drop data
├── solver/             # Nodal admittance matrix solver
├── postprocess/        # IL/RL extraction and IEEE mask functions
└── cli.py              # Top-level pipeline: run() and run_tx_sweep()

notebooks/
└── analysis.ipynb      # Visualization and compliance checks

docs/
└── methodology.md      # Derivation of all computed quantities

tests/                  # 89+ pytest tests
measurements/           # Touchstone files (not included in repo)
examples/               # Touchstone files used for unittests and showcase purposes
config.yaml             # Simulation configuration
```

---

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/m4rin3r0/spmd_reflection.git
cd spmd_reflection
pip install -e .
```

Dependencies are declared in `pyproject.toml` and include `numpy`, `scikit-rf`, and `pyyaml`.

---

## Configuration

Create a `config.yaml` in the project root:

```yaml
frequency:
  start_hz: 300000.0
  stop_hz: 40000000.0
  n_points: 1001

topology:
  drop_positions_m: [1.0, 3.0, 5.0]
  bus_start_m: 0.0
  bus_end_m: 7.0
  tx_drop_index: 0
  termination_ohm: 100.0

drop:
  touchstone: measurements/your_drop_measurement.s2p  # jumped measurement
  phy_load_ohm: 20000.0

cable:
  l_per_m: 413e-9    # H/m
  c_per_m: 45e-12    # F/m
  rdc_per_m: 0.19    # Ω/m
  rskin_per_m: 5e-7  # Ω/(m·√Hz)
```

---

## Usage

### Single simulation

```python
from pathlib import Path
from spmd_reflection.cli import run

results = run(Path("config.yaml"))

# results.rl_db            – Bus RL at TX port,     shape (n_freq,)
# results.rx_to_tx_db        – RX/TX at RX PHY ports,    shape (n_freq, n_rx_drops)
# results.il_tci_db        – TCI IL per drop,        shape (n_freq, n_drops)
# results.rl_tci_db        – TCI RL per drop,        shape (n_freq, n_drops)
# results.ms_il_db         – Mixing segment IL,      shape (n_freq,)
```

### TX sweep

```python
from spmd_reflection.cli import run_tx_sweep

sweep = run_tx_sweep(Path("config.yaml"))
# sweep[i] = BusResults with drop i as TX
```

### Mask checks

```python
from spmd_reflection.postprocess.masks import check_tci_il, check_tci_rl

ok_il = check_tci_il(results.frequency_hz, results.il_tci_db)
ok_rl = check_tci_rl(results.frequency_hz, results.rl_tci_db)
# True = conformant, False = violates mask
```

### Notebook

Open `notebooks/analysis.ipynb` in VSCode or JupyterLab for plots and
compliance summaries.

---

## Running Tests

```bash
pytest -v
```

89+ tests covering all modules. Some tests require the measurement files to be
present in the `examples/` directory; they are skipped automatically if the
files are not found.

---

## Methodology

See [`docs/methodology.md`](docs/methodology.md) for a full derivation of all
computed quantities, modeling assumptions, and known limitations.

Key assumptions:
- Drop PCB connected as **parallel shunt** to the trunk (Case A)
- All drops use the same Touchstone file (uniform population)
- Edge terminations are frequency-independent resistors

---

## IEEE 802.3da Compliance

The following mask functions are implemented:

| Standard | Equation | Quantity |
|----------|----------|----------|
| Clause 188 | Eq. 188-3 | Mixing Segment IL |
| Clause 188 | Eq. 188-5 | TCI IL |
| Clause 188 | Eq. 188-6 | TCI RL |
| Clause 189 | Eq. 189-1 | MPI RL (PoDL) |
| Clause 189 | Eq. 189-2 | MPI IL (PoDL) |

---

## License

This project was developed for academic research purposes.

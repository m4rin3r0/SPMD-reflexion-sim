# Simulation Methodology

This document describes how the physical quantities computed by `spmd_reflection`
are derived from measured S-parameters and the nodal admittance matrix solver.

---

## 1. Modeling Assumptions

### 1.1 Bus Topology

The 10BASE-T1M multi-drop bus is modeled as a linear transmission line (the
trunk) with $N$ drop PCBs connected in parallel at discrete positions. Each drop
PCB connects the trunk to a PHY device. The trunk is terminated at both ends
with resistive edge terminations of impedance $Z_T$ (typically 100 Ω).

### 1.2 Drop PCB Model (Parallel Shunt — Case A)

Each drop PCB is assumed to connect to the trunk as a **parallel shunt element**.
That is, TC1 and TC2 (the two trunk-side connectors of the TCI) are directly
connected on the PCB, and the drop circuitry (CMC, ESD protection, connectors)
hangs in parallel between this trunk connection and the PHY port.

This assumption — referred to as **Case A** throughout this document — is valid
for PCB layouts where the signal path runs directly from TC1 to TC2, with the
PHY stub branching off in parallel. It must be verified against the actual PCB
schematic before applying the derived formulas.

Under Case A, the drop PCB is fully described as a **2-port network**:

- **Port 0**: PHY side
- **Port 1**: Trunk side (TC1 and TC2 shorted together)

### 1.3 Measurement Basis: Jumped PCB

All drop PCBs are measured with the PHY input impedance **bypassed** (jumped),
using a 4×2 pin header on the PCB. This yields a 2-port S-parameter file
describing only the passive PCB circuitry, without the PHY load.

The same Touchstone file is used for both TX and RX drops. The role-specific
termination is applied externally by the solver:

- **TX drop**: Norton source at the PHY-side port
- **RX drop**: PHY load resistance (approximated with 20 kΩ) at the PHY-side port

### 1.4 Norton Source Model

The transmitting PHY is modeled as a **Norton equivalent source**: a current
source $I_s = 1/Z_0$ in parallel with a source admittance $Y_s = 1/Z_0$, where
$Z_0 = 100\,\Omega$ is the differential bus characteristic impedance. This gives
a Thévenin equivalent open-circuit voltage of 1 V and a source impedance of
100 Ω, consistent with the wave reference impedance used throughout the
simulation.

### 1.5 Reference Impedance

All S-parameter files are renormalized to a **differential reference impedance
of 100 Ω** before use. This is consistent with the bus characteristic impedance
and with the IEEE 802.3da measurement reference.

### 1.6 Frequency-Independent Terminations

Edge terminations are modeled as ideal resistors, independent of frequency.
It is not intended to model AC-coupled terminations.

### 1.7 Uniform Drop Population

All drop positions use the same Touchstone file. Per-drop variation
(manufacturing tolerances, individual measurements) is not modeled.
This is a deliberate simplification to reduce measurement effort; the
architecture supports per-drop files as a future extension.

---

## 2. Nodal Admittance Matrix Solver

### 2.1 System Assembly

The bus is described by a **nodal admittance matrix** $\mathbf{Y}$ of size
$N_\text{total} \times N_\text{total}$, where:

$$N_\text{total} = N_\text{topology} + N_\text{RX} + 1$$

- $N_\text{topology}$: number of trunk nodes (from topology)
- $N_\text{RX}$: one internal PHY node per RX drop
- $+1$: one internal PHY node for the TX drop

Node layout:

```
[0 .. N_topology-1]      Trunk nodes
[N_topology .. -2]       RX-PHY nodes (one per RX drop)
[-1]                     TX-PHY node
```

The following elements are stamped into $\mathbf{Y}$:

| Element | Stamping |
|---------|----------|
| Trunk segment of length $l$ | 2-port Y-matrix from telegrapher's equations |
| RX drop | 2-port Y-matrix between trunk node and RX-PHY node; PHY load $Y_L = 1/R_\text{PHY}$ at RX-PHY node |
| TX drop | 2-port Y-matrix between trunk node and TX-PHY node |
| Edge termination $Z_T$ | Shunt $1/Z_T$ at termination node |

### 2.2 Norton Source Excitation

The Norton source is placed at the TX-PHY node. The source admittance
$Y_s = 1/Z_0$ is added to the diagonal entry $Y[\text{tx\_phy}, \text{tx\_phy}]$.
The injected current vector has $I_s = Y_s$ at the TX-PHY node and zero
elsewhere.

The nodal system $\mathbf{Y} \cdot \mathbf{v} = \mathbf{i}$ is solved for the
node voltage vector $\mathbf{v}$ at each frequency independently.

### 2.3 Cable Model

Trunk segments are modeled using the **distributed-parameter telegrapher's
equations**. For a segment of length $l$, the 2-port Y-matrix is:

$$Y_{11} = Y_{22} = \frac{\cosh(\gamma l)}{Z_0 \sinh(\gamma l)}, \quad
Y_{12} = Y_{21} = \frac{-1}{Z_0 \sinh(\gamma l)}$$

where the propagation constant is:

$$\gamma = \sqrt{(R' + j\omega L')(j\omega C')}$$

and $Z_0 = \sqrt{(R' + j\omega L') / (j\omega C')}$ is the complex characteristic
impedance. The skin-effect resistance is modeled as
$R'(f) = R'_\text{DC} + R'_\text{skin} \sqrt{f}$.

---

## 3. Bus Return Loss (RL_Bus)

The **bus return loss at the TX port** is extracted from the voltage and current
at the TX-PHY node after solving the nodal system.

The incident and reflected wave amplitudes at the TX-PHY port are:

$$a_1 = V_\text{tx} + I_\text{tx} \cdot Z_0, \quad b_1 = V_\text{tx} - I_\text{tx} \cdot Z_0$$

where the port current is:

$$I_\text{tx} = Y_s - Y_s \cdot V_\text{tx} = Y_s (1 - V_\text{tx})$$

The reflection coefficient and return loss are:

$$S_{11} = \frac{b_1}{a_1}, \quad \text{RL}_\text{Bus}(f) = -20 \log_{10}(|S_{11}|) \quad [\text{dB}]$$

A higher RL value indicates less reflected energy, so better impedance matching.

---

## 4. PHY Insertion Loss (IL_PHY)

The **PHY insertion loss** quantifies how much signal reaches the PHY of each
RX drop relative to the available source power.

The RX-PHY node voltage $V_\text{rx,i}$ is read directly from the solved node
voltage vector. The reference level is the voltage that would appear across a
matched 100 Ω load driven by the Norton source with 1 V open-circuit voltage:

$$V_\text{ref} = \frac{1\,\text{V}}{2} = 0.5\,\text{V}$$

The PHY insertion loss for RX drop $i$ is:

$$\text{IL}_\text{PHY,i}(f) = -20 \log_{10}\left(\frac{|V_\text{rx,i}|}{0.5}\right) \quad [\text{dB}]$$

A value of 0 dB means the RX PHY receives the same power as a directly connected
matched load. Positive values indicate signal loss.

**Note:** IL_PHY is a system-level quantity that captures the combined effect of
cable losses, reflections from all drops, and the impedance loading of the PHY
input. It is not directly specified by IEEE 802.3da but is the most practically
relevant figure of merit for receiver signal quality.

---

## 5. TCI Insertion Loss (IL_TCI)

The **TCI insertion loss** is defined by IEEE 802.3da Clause 188 (Eq. 188-5) as
the transmission loss between TC1 and TC2 of a single TCI, measured into 100 Ω
with the station load present.

### 5.1 Derivation Under Case A

Under the parallel-shunt assumption (Case A), the drop PCB presents a shunt
admittance $Y_\text{drop}(f)$ to the trunk. The transmission from TC1 to TC2
through a shunt element between two 100 Ω lines is:

$$S_{21,\text{TCI}} = \frac{2}{2 + Z_0 \cdot Y_\text{drop}(f)}$$

The TCI insertion loss is then:

$$\text{IL}_\text{TCI}(f) = -20 \log_{10}(|S_{21,\text{TCI}}|) \quad [\text{dB}]$$

### 5.2 Trunk-Side Input Admittance

$Y_\text{drop}$ is derived from the measured 2-port Y-matrix by terminating
Port 0 (PHY side) with the PHY load admittance $Y_L = 1/R_\text{PHY}$:

$$Y_\text{drop}(f) = Y_{11}^\text{(trunk)} - \frac{Y_{10} \cdot Y_{01}}{Y_{00} + Y_L}$$

where the indices refer to the port numbering: Port 0 = PHY, Port 1 = trunk.

### 5.3 Validity

This derivation is **exact under Case A**. If the PCB routes the signal through
the CMC in series between TC1 and TC2 (Case B), a separate TC1→TC2 measurement
would be required, and the formula above would not apply.

---

## 6. TCI Return Loss (RL_TCI)

The **TCI return loss** quantifies the impedance match of the drop at the trunk
port. It is defined by IEEE 802.3da Clause 188 (Eq. 188-6).

The trunk-side reflection coefficient is:

$$\Gamma_\text{TC}(f) = \frac{Y_0 - Y_\text{drop}(f)}{Y_0 + Y_\text{drop}(f)}$$

where $Y_0 = 1/Z_0 = 0.01\,\text{S}$ and $Y_\text{drop}$ is the trunk-side
input admittance defined in Section 5.2. The TCI return loss is:

$$\text{RL}_\text{TCI}(f) = -20 \log_{10}(|\Gamma_\text{TC}|) \quad [\text{dB}]$$

A higher RL_TCI value indicates better impedance matching at the trunk port,
meaning the drop causes fewer reflections on the bus.

**Note on interpretation:** For a high-impedance PHY load ($R_\text{PHY} \gg Z_0$),
the trunk-side admittance $Y_\text{drop}$ is small compared to $Y_0$, so
$\Gamma_\text{TC} \approx 1$ and RL_TCI is small. This is physically correct:
a weakly-coupled drop reflects most of the incident wave, but also disturbs the
bus very little. The IEEE RL_TCI mask (Eq. 188-6) sets a lower bound on RL_TCI
to ensure the drop does not cause excessive reflections.

---

## 7. Mixing Segment Insertion Loss (IL_MS)

The **mixing segment insertion loss** is defined by IEEE 802.3da Clause 188
(Eq. 188-3) as the transmission loss between the two edge termination reference
planes of the entire bus, measured into 100 Ω with all station loads attached.
The edge terminators are replaced by the measurement probes during this test.

### 7.1 Simulation Setup

To compute IL_MS, a separate simulation is performed with:

- A Norton source (100 Ω) placed at the **left bus-end node**
- A 100 Ω load at the **right bus-end node**
- All drops modeled as in the standard simulation (2-port + PHY load)
- **Edge terminations not stamped** (replaced by source and load impedances)

### 7.2 S₂₁ Extraction

The mixing segment S₂₁ is extracted as:

$$S_{21,\text{MS}} = \frac{2 \cdot V_\text{right}}{a_1}$$

where $a_1 = V_\text{left} + I_\text{left} \cdot Z_0$ is the incident wave
amplitude at the left port. The mixing segment insertion loss is:

$$\text{IL}_\text{MS}(f) = -20 \log_{10}(|S_{21,\text{MS}}|) \quad [\text{dB}]$$

---

## 8. IEEE 802.3da Mask Functions

The following mask functions from IEEE 802.3da are implemented:

| Mask | Equation | Quantity | Direction |
|------|----------|----------|-----------|
| TCI IL limit | 188-5 | IL_TCI | ≤ limit |
| TCI RL limit | 188-6 | RL_TCI | ≥ limit |
| Mixing segment IL limit | 188-3 | IL_MS | ≤ limit |
| MPI RL limit (PoDL) | 189-1 | RL_TCI | ≥ limit |
| MPI IL limit (PoDL) | 189-2 / 188-5 | IL_TCI | ≤ limit |

All mask functions accept frequency in Hz and return limit values in dB. Points
outside the valid frequency range (0.3–40 MHz) return NaN and are treated as
conformant in the compliance checks.

The MPI masks (Clause 189) apply when the TCI also implements PoDL (Power over
Data Line). The parameter $N_\text{UNIT}$ in Eq. 189-1 is the MPD unit load
value; the default of 16 applies to MPSEs. The correct value for a specific
configuration requires verification against the Clause 189 definitions.

---

## 9. Limitations and Open Points

| Topic | Status |
|-------|--------|
| Case A assumption | Must be verified against PCB schematic |
| Per-drop variation | Not modeled; single Touchstone for all drops |
| AC-coupled terminations | Not modeled; resistive only |
| N_UNIT for MPI masks | Default 16; requires Clause 189 verification |
| TX sweep | Implemented; each drop simulated as TX in turn |
| Mixing segment IL conformance check | Implemented |
| Validation against bus measurements | Pending |

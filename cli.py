"""Command-line entry point for spmd-reflection-sim."""

from __future__ import annotations

import argparse
import numpy as np

from spmd_reflection.config import load_config
from spmd_reflection.topology import build_topology
from spmd_reflection.touchstone import parse_s2p, interpolate_s_params, s_to_y
from spmd_reflection.solver_ac import run_ac_sim
from spmd_reflection.plots import plot_results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPMD reflection AC simulator")
    parser.add_argument("--json", type=str, default=None, help="Path to JSON config")
    parser.add_argument("--s2p", type=str, default=None, help="Global S2P file")
    parser.add_argument("--freq_start", type=float, default=None)
    parser.add_argument("--freq_stop", type=float, default=None)
    parser.add_argument("--npoints", type=int, default=None)
    parser.add_argument("--z0", type=float, default=None)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--length", type=float, default=None)
    parser.add_argument("--drop_max", type=float, default=None)
    parser.add_argument("--random_drop", action="store_true")
    parser.add_argument("--random_attach", action="store_true")
    parser.add_argument("--separation_min", type=float, default=None)
    parser.add_argument("--start_pad", type=float, default=None)
    parser.add_argument("--end_pad", type=float, default=None)
    parser.add_argument("--start_attach", type=int, default=None)
    parser.add_argument("--end_attach", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tx_node", type=int, default=None)
    parser.add_argument("--rterm", type=float, default=None)
    parser.add_argument("--plot", type=str, default=None, help="Save plot to file")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    overrides = {}
    if args.s2p:
        overrides["s2p"] = args.s2p
    if args.freq_start is not None:
        overrides["freq_start"] = args.freq_start
    if args.freq_stop is not None:
        overrides["freq_stop"] = args.freq_stop
    if args.npoints is not None:
        overrides["npoints"] = args.npoints
    if args.z0 is not None:
        overrides["z0"] = args.z0
    if args.nodes is not None:
        overrides["nodes"] = args.nodes
    if args.length is not None:
        overrides["length"] = args.length
    if args.drop_max is not None:
        overrides["drop_max"] = args.drop_max
    if args.random_drop:
        overrides["random_drop"] = True
    if args.random_attach:
        overrides["random_attach"] = True
    if args.separation_min is not None:
        overrides["separation_min"] = args.separation_min
    if args.start_pad is not None:
        overrides["start_pad"] = args.start_pad
    if args.end_pad is not None:
        overrides["end_pad"] = args.end_pad
    if args.start_attach is not None:
        overrides["start_attach"] = args.start_attach
    if args.end_attach is not None:
        overrides["end_attach"] = args.end_attach
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.tx_node is not None:
        overrides["tx_node"] = args.tx_node
    if args.rterm is not None:
        overrides.setdefault("termination", {})["rterm"] = args.rterm

    config = load_config(args.json, overrides)

    freq = np.linspace(config.freq_start, config.freq_stop, config.npoints)
    topo = build_topology(config.data)

    touchstone = parse_s2p(config.data["s2p"])
    s_interp = interpolate_s_params(touchstone, freq)
    y_s2p = s_to_y(s_interp, touchstone.z0)

    results = run_ac_sim(
        topology=topo,
        cable_model=config.data["cable_model"],
        y_s2p=y_s2p,
        frequency=freq,
        z0=config.z0,
        rterm=config.data["termination"]["rterm"],
    )

    plot_results(results, output_path=args.plot)


if __name__ == "__main__":
    main()

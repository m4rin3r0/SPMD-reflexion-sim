"""Command-line entry point for spmd-reflection-sim"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from spmd_reflection.config import load_config
from spmd_reflection.topology import build_topology
from spmd_reflection.touchstone_parser.touchstone import interpolate_s_params, parse_s2p, s11_to_y, s_to_y, write_s2p
from spmd_reflection.solver_ac import run_ac_sim
from spmd_reflection.plots import plot_results
from spmd_reflection.touchstone_parser.touchstone_wrapper import build_differential_s_params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SPMD reflection AC simulator")
    parser.add_argument("--json", type=str, default=None, help="Path to JSON config")
    parser.add_argument("--s2p", type=str, default=None, help="RX drop S2P file")
    parser.add_argument(
        "--s2p-zip",
        type=str,
        default=None,
        help="ZIP archive or directory containing the RX drop measurements",
    )
    parser.add_argument("--jumped-s2p", type=str, default=None, help="TX jumped S2P file")
    parser.add_argument(
        "--jumped-s2p-zip",
        type=str,
        default=None,
        help="ZIP archive or directory containing the TX jumped measurements",
    )
    parser.add_argument("--freq_start", type=float, default=None)
    parser.add_argument("--freq_stop", type=float, default=None)
    parser.add_argument("--npoints", type=int, default=None)
    parser.add_argument("--z0", type=float, default=None)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--length", type=float, default=None)
    parser.add_argument("--tx_node", type=int, default=None)
    parser.add_argument("--plot", type=str, default=None, help="Save plot to file")
    return parser


def _parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = _parse_args()
    overrides = {}
    if args.s2p:
        overrides["s2p"] = args.s2p
    if args.s2p_zip:
        overrides["s2p_zip"] = args.s2p_zip
    if args.jumped_s2p:
        overrides["jumped_s2p"] = args.jumped_s2p
    if args.jumped_s2p_zip:
        overrides["jumped_s2p_zip"] = args.jumped_s2p_zip
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
    if args.tx_node is not None:
        overrides["tx_node"] = args.tx_node

    config = load_config(args.json, overrides)

    if config.data.get("s2p_zip"):
        file_path = config.data["s2p_zip"]
        rx_touchstone = build_differential_s_params(file_path)
    else:
        rx_touchstone = parse_s2p(config.data["s2p"])

    if config.data.get("jumped_s2p_zip"):
        file_path = config.data["jumped_s2p_zip"]
        tx_touchstone = build_differential_s_params(file_path)
    else:
        tx_touchstone = parse_s2p(config.data["jumped_s2p"])

    freq = np.linspace(config.freq_start, config.freq_stop, config.npoints)
    topo = build_topology(config.data)

    rx_interp = interpolate_s_params(rx_touchstone, freq)
    tx_interp = interpolate_s_params(tx_touchstone, freq)
    # RX stubs are reduced to a 1-port shunt seen from differential port 2,
    # the receive side connected to the trunk.
    rx_shunt_y = s11_to_y(rx_interp[:, 1, 1], rx_touchstone.z0)
    tx_y = s_to_y(tx_interp, tx_touchstone.z0)

    results = run_ac_sim(
        topology=topo,
        cable_model=config.data["cable_model"],
        rx_shunt_y=rx_shunt_y,
        tx_y=tx_y,
        frequency=freq,
        z0=config.z0,
    )

    plot_results(results, output_path=args.plot)


if __name__ == "__main__":
    main()

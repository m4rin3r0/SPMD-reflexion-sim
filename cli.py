"""Command-line entry point for spmd-reflection-sim"""

from __future__ import annotations

import argparse
from pathlib import Path


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
    parser.add_argument(
        "--export-s2p",
        type=str,
        default=None,
        help="Optional output path for the differential RX drop S2P generated from --s2p-zip",
    )
    parser.add_argument(
        "--export-jumped-s2p",
        type=str,
        default=None,
        help="Optional output path for the differential TX jumped S2P generated from --jumped-s2p-zip",
    )
    parser.add_argument(
        "--plot-repeated",
        type=str,
        default=None,
        help="Optional output path for a comparison plot of repeated single-ended S-parameter measurements from --s2p-zip",
    )
    parser.add_argument("--plot", type=str, default=None, help="Save plot to file")
    return parser


def _parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    import numpy as np

    from spmd_reflection.config import load_config
    from spmd_reflection.measurement_import import (
        build_single_ended_4port,
        convert_single_ended_to_differential,
        load_measurement_archive,
    )
    from spmd_reflection.topology import build_topology
    from spmd_reflection.touchstone import (
        interpolate_s_params,
        parse_s2p,
        s11_to_y,
        s_to_y,
        write_s2p,
    )
    from spmd_reflection.solver_ac import run_ac_sim
    from spmd_reflection.plots import plot_repeated_measurement_traces, plot_results

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
        archive = load_measurement_archive(config.data["s2p_zip"])
        if args.plot_repeated:
            plot_repeated_measurement_traces(archive, output_path=args.plot_repeated)
            print(f"Exported repeated-trace comparison plot to: {args.plot_repeated}")

        single_ended = build_single_ended_4port(archive)
        rx_touchstone = convert_single_ended_to_differential(single_ended)
        export_path = args.export_s2p
        if export_path is None:
            zip_path = Path(config.data["s2p_zip"])
            export_path = str(zip_path.with_name(f"{zip_path.stem}_differential.s2p"))
        write_s2p(export_path, rx_touchstone)
        print(f"Exported differential S2P to: {export_path}")
    else:
        rx_touchstone = parse_s2p(config.data["s2p"])

    if config.data.get("jumped_s2p_zip"):
        jumped_archive = load_measurement_archive(config.data["jumped_s2p_zip"])
        jumped_single_ended = build_single_ended_4port(jumped_archive)
        tx_touchstone = convert_single_ended_to_differential(jumped_single_ended)
        export_jumped_path = args.export_jumped_s2p
        if export_jumped_path is None:
            zip_path = Path(config.data["jumped_s2p_zip"])
            export_jumped_path = str(zip_path.with_name(f"{zip_path.stem}_differential.s2p"))
        write_s2p(export_jumped_path, tx_touchstone)
        print(f"Exported jumped differential S2P to: {export_jumped_path}")
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

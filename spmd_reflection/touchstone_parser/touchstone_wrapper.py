from spmd_reflection.touchstone_parser.measurement_import import build_single_ended_4port, convert_single_ended_to_differential, load_measurement_archive
from spmd_reflection.touchstone_parser.touchstone import TouchstoneData, write_s2p


def build_differential_s_params(file_path:str) -> TouchstoneData:
    archive = load_measurement_archive(file_path)
    single_ended = build_single_ended_4port(archive)
    differential = convert_single_ended_to_differential(single_ended)
    return differential

def export_differential_s_params(file_path:str, touchstone:TouchstoneData):
    write_s2p(file_path, touchstone)
    print(f"Exported jumped differential S2P to: {file_path}")
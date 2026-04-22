"""Tests for topology.build.

Covers positive cases (correct topology construction for various bus layouts)
and negative cases (validation errors for invalid configurations).
"""

import pytest
from spmd_reflection.topology.build import build_topology
from spmd_reflection.topology.models import TrunkSegment, DropAttachment, Termination

"""Table of content
positive tests:
    1. Test: general topology build
        P1: symmetric bus, all drops in the middle
    2. Test: merging of termination and drop
        P2: merging at left end
        P3: merging at right end
        P4: merging at both ends
    3. Test: minimal bus
        P5: just one drop
        P6: extreme case, just one drop and its at the end
    4. Test: TX in the middle
        P7: TX not at the beginning of mixing segment
    5. Test: asymmetric bus
        P8: asymmetric lengths
negative tests:
    6. Test: no drop error raise
        N1: empty drop list
    7. Test: indices
        N2: bus-length not positive
        N3: drop before bus start
        N4: drop after bus end
        N5: duplicated bus-positions
        N6: unsorted bus-positions
        N7: negative tx-drop index
        N8: tx-drop index outside range
    8. Test: termination
        N9: negative termination

"""

# ---------------------------------------------------------------------------
# 1. Test: general topology build
# ---------------------------------------------------------------------------

def test_simple_bus_with_drops_in_middle():
    """Three drops between the bus ends, TX is the leftmost drop."""
    
    topo = build_topology(
        drop_positions_m = [1.0, 3.0, 5.0],
        bus_start_m = 0.0,
        bus_end_m = 7.0,
        tx_drop_index = 0,
        termination_ohm = 100.0)

    # Five nodes: left term, three drops, right term.
    assert topo.n_nodes == 5

    # Four cable segments connecting the five nodes in sequence.
    assert topo.trunk_segments == (
        TrunkSegment(node_a=0, node_b=1, length_m=1.0),
        TrunkSegment(node_a=1, node_b=2, length_m=2.0),
        TrunkSegment(node_a=2, node_b=3, length_m=2.0),
        TrunkSegment(node_a=3, node_b=4, length_m=2.0))

    # First drop is TX, others are RX. Each sits on its own trunk node.
    assert topo.drops == (
        DropAttachment(trunk_node=1, role="tx"),
        DropAttachment(trunk_node=2, role="rx"),
        DropAttachment(trunk_node=3, role="rx"))

    # Terminations at both physical bus ends.
    assert topo.terminations == (
        Termination(node=0, impedance_ohm=100.0),
        Termination(node=4, impedance_ohm=100.0))


# ---------------------------------------------------------------------------
# 2. Test: merging of termination and drop
# ---------------------------------------------------------------------------  

def test_first_drop_at_left_bus_end():
    """First drop sits at bus_start_m, merging with the left termination node."""
    topo = build_topology(
        drop_positions_m=[0.0, 3.0, 5.0],
        bus_start_m=0.0,
        bus_end_m=7.0,
        tx_drop_index=0,
        termination_ohm=100.0)

    # Four nodes: first drop merges with left termination.
    assert topo.n_nodes == 4

    # Three segments: no segment from termination to first drop (same node).
    assert topo.trunk_segments == (
        TrunkSegment(node_a=0, node_b=1, length_m=3.0),
        TrunkSegment(node_a=1, node_b=2, length_m=2.0),
        TrunkSegment(node_a=2, node_b=3, length_m=2.0))

    # First drop (TX) sits on node 0 — same node as the left termination.
    assert topo.drops == (
        DropAttachment(trunk_node=0, role="tx"),
        DropAttachment(trunk_node=1, role="rx"),
        DropAttachment(trunk_node=2, role="rx"))

    # Left termination shares node with first drop.
    assert topo.terminations == (
        Termination(node=0, impedance_ohm=100.0),
        Termination(node=3, impedance_ohm=100.0))


def test_last_drop_at_right_bus_end():
    """Last drop sits at bus_end_m, merging with the right termination node."""
    topo = build_topology(
        drop_positions_m=[1.0, 3.0, 7.0],
        bus_start_m=0.0,
        bus_end_m=7.0,
        tx_drop_index=0,
        termination_ohm=100.0)

    # Four nodes: last drop merges with right termination.
    assert topo.n_nodes == 4

    # Three segments: no segment from last drop to right termination.
    assert topo.trunk_segments == (
        TrunkSegment(node_a=0, node_b=1, length_m=1.0),
        TrunkSegment(node_a=1, node_b=2, length_m=2.0),
        TrunkSegment(node_a=2, node_b=3, length_m=4.0))

    # Last drop sits on node 3 — same node as the right termination.
    assert topo.drops == (
        DropAttachment(trunk_node=1, role="tx"),
        DropAttachment(trunk_node=2, role="rx"),
        DropAttachment(trunk_node=3, role="rx"))

    # Right termination shares node with last drop.
    assert topo.terminations == (
        Termination(node=0, impedance_ohm=100.0),
        Termination(node=3, impedance_ohm=100.0))
    

def test_drops_at_both_bus_ends():
    """First drop at bus_start and last drop at bus_end — both merge with terminations."""
    topo = build_topology(
        drop_positions_m=[0.0, 3.0, 7.0],
        bus_start_m=0.0,
        bus_end_m=7.0,
        tx_drop_index=0,
        termination_ohm=100.0)

    # Three nodes: first drop = left termination, last drop = right termination.
    assert topo.n_nodes == 3

    # Two segments connecting the three nodes.
    assert topo.trunk_segments == (
        TrunkSegment(node_a=0, node_b=1, length_m=3.0),
        TrunkSegment(node_a=1, node_b=2, length_m=4.0))

    # Drops share nodes with terminations at both ends.
    assert topo.drops == (
        DropAttachment(trunk_node=0, role="tx"),
        DropAttachment(trunk_node=1, role="rx"),
        DropAttachment(trunk_node=2, role="rx"))

    # Both terminations share nodes with drops.
    assert topo.terminations == (
        Termination(node=0, impedance_ohm=100.0),
        Termination(node=2, impedance_ohm=100.0))
    

# ---------------------------------------------------------------------------
# 3. Test: minimal bus
# ---------------------------------------------------------------------------  

def test_minimal_bus_with_single_drop():
    """Single drop between bus ends — smallest valid configuration."""
    topo = build_topology(
        drop_positions_m=[3.0],
        bus_start_m=0.0,
        bus_end_m=7.0,
        tx_drop_index=0,
        termination_ohm=100.0)

    # Three nodes: left term, the single drop, right term.
    assert topo.n_nodes == 3

    # Two segments connecting the three nodes.
    assert topo.trunk_segments == (
        TrunkSegment(node_a=0, node_b=1, length_m=3.0),
        TrunkSegment(node_a=1, node_b=2, length_m=4.0))

    # Single drop is TX (because tx_drop_index=0).
    assert topo.drops == (DropAttachment(trunk_node=1, role="tx"),)

    # Terminations at both ends.
    assert topo.terminations == (
        Termination(node=0, impedance_ohm=100.0),
        Termination(node=2, impedance_ohm=100.0))
    

def test_single_drop_at_bus_start():
    """Single drop at bus_start_m — drop merges with left termination."""
    topo = build_topology(
        drop_positions_m=[0.0],
        bus_start_m=0.0,
        bus_end_m=7.0,
        tx_drop_index=0,
        termination_ohm=100.0)

    # Two nodes: drop+left term share node 0, right term is node 1.
    assert topo.n_nodes == 2

    # One segment spanning the full bus length.
    assert topo.trunk_segments == (TrunkSegment(node_a=0, node_b=1, length_m=7.0),)

    # Drop sits on node 0 with the left termination.
    assert topo.drops == (DropAttachment(trunk_node=0, role="tx"),)

    # Left termination shares node with drop, right termination is alone.
    assert topo.terminations == (
        Termination(node=0, impedance_ohm=100.0),
        Termination(node=1, impedance_ohm=100.0))
    

# ---------------------------------------------------------------------------
# 4. Test: tx in the middle
# ---------------------------------------------------------------------------  

def test_tx_drop_in_middle():
    """TX is not the first drop — validates that tx_drop_index works correctly."""
    topo = build_topology(
        drop_positions_m=[1.0, 3.0, 5.0],
        bus_start_m=0.0,
        bus_end_m=7.0,
        tx_drop_index=1,
        termination_ohm=100.0)

    # Five nodes: left term, three drops, right term.
    assert topo.n_nodes == 5

    # Segments are independent of which drop is TX.
    assert topo.trunk_segments == (
        TrunkSegment(node_a=0, node_b=1, length_m=1.0),
        TrunkSegment(node_a=1, node_b=2, length_m=2.0),
        TrunkSegment(node_a=2, node_b=3, length_m=2.0),
        TrunkSegment(node_a=3, node_b=4, length_m=2.0))

    # Second drop is TX, others are RX.
    assert topo.drops == (
        DropAttachment(trunk_node=1, role="rx"),
        DropAttachment(trunk_node=2, role="tx"),
        DropAttachment(trunk_node=3, role="rx"))

    # Terminations unchanged by TX position.
    assert topo.terminations == (
        Termination(node=0, impedance_ohm=100.0),
        Termination(node=4, impedance_ohm=100.0))


# ---------------------------------------------------------------------------
# 5. Test: asymmetric bus
# --------------------------------------------------------------------------- 

def test_asymmetric_bus_layout():
    """Drops at irregular positions with varying segment lengths."""
    topo = build_topology(
        drop_positions_m=[0.5, 2.0, 6.5],
        bus_start_m=0.0,
        bus_end_m=7.0,
        tx_drop_index=0,
        termination_ohm=100.0)

    # Five nodes: left term, three drops, right term.
    assert topo.n_nodes == 5

    # Four segments with different lengths, derived from position differences.
    assert topo.trunk_segments == (
        TrunkSegment(node_a=0, node_b=1, length_m=0.5),
        TrunkSegment(node_a=1, node_b=2, length_m=1.5),
        TrunkSegment(node_a=2, node_b=3, length_m=4.5),
        TrunkSegment(node_a=3, node_b=4, length_m=0.5))

    # Drop roles follow tx_drop_index.
    assert topo.drops == (
        DropAttachment(trunk_node=1, role="tx"),
        DropAttachment(trunk_node=2, role="rx"),
        DropAttachment(trunk_node=3, role="rx"))

    # Standard terminations at both ends.
    assert topo.terminations == (
        Termination(node=0, impedance_ohm=100.0),
        Termination(node=4, impedance_ohm=100.0))
    

# ---------------------------------------------------------------------------
# 6. Test: no drop error raise
# --------------------------------------------------------------------------- 

def test_rejects_empty_drop_list():
    """build_topology requires at least one drop."""
    with pytest.raises(ValueError, match="at least one drop"):
        build_topology(
            drop_positions_m=[],
            bus_start_m=0.0,
            bus_end_m=7.0,
            tx_drop_index=0,
            termination_ohm=100.0)
        

# ---------------------------------------------------------------------------
# 7. Test: indices
# --------------------------------------------------------------------------- 

def test_rejects_bus_start_not_less_than_end():
    """bus_start_m must be strictly less than bus_end_m."""
    with pytest.raises(ValueError, match="bus_start_m.*less than.*bus_end_m"):
        build_topology(
            drop_positions_m=[3.0],
            bus_start_m=7.0,
            bus_end_m=7.0,
            tx_drop_index=0,
            termination_ohm=100.0)


def test_rejects_drop_before_bus_start():
    """Drop positions must not be smaller than bus_start_m."""
    with pytest.raises(ValueError, match="drop_positions_m.*outside.*bus"):
        build_topology(
            drop_positions_m=[-0.5, 3.0, 5.0],
            bus_start_m=0.0,
            bus_end_m=7.0,
            tx_drop_index=0,
            termination_ohm=100.0)


def test_rejects_drop_after_bus_end():
    """Drop positions must not exceed bus_end_m."""
    with pytest.raises(ValueError, match="drop_positions_m.*outside.*bus"):
        build_topology(
            drop_positions_m=[1.0, 3.0, 7.5],
            bus_start_m=0.0,
            bus_end_m=7.0,
            tx_drop_index=0,
            termination_ohm=100.0)
        

def test_rejects_duplicate_drop_positions():
    """Two drops at the same position are not allowed."""
    with pytest.raises(ValueError, match="drop_positions_m.*strictly.*increasing"):
        build_topology(
            drop_positions_m=[1.0, 3.0, 3.0],
            bus_start_m=0.0,
            bus_end_m=7.0,
            tx_drop_index=0,
            termination_ohm=100.0)
     

def test_rejects_unsorted_drop_positions():
    """Drop positions must be strictly increasing."""
    with pytest.raises(ValueError, match="drop_positions_m.*strictly.*increasing"):
        build_topology(
            drop_positions_m=[1.0, 5.0, 3.0],
            bus_start_m=0.0,
            bus_end_m=7.0,
            tx_drop_index=0,
            termination_ohm=100.0)
        

def test_rejects_negative_tx_drop_index():
    """tx_drop_index must not be negative."""
    with pytest.raises(ValueError, match="tx_drop_index.*out of range"):
        build_topology(
            drop_positions_m=[1.0, 3.0, 5.0],
            bus_start_m=0.0,
            bus_end_m=7.0,
            tx_drop_index=-1,
            termination_ohm=100.0)


def test_rejects_tx_drop_index_out_of_range():
    """tx_drop_index must be less than the number of drops."""
    with pytest.raises(ValueError, match="tx_drop_index.*out of range"):
        build_topology(
            drop_positions_m=[1.0, 3.0, 5.0],
            bus_start_m=0.0,
            bus_end_m=7.0,
            tx_drop_index=3,
            termination_ohm=100.0)
        
# ---------------------------------------------------------------------------
# 8. Test: termination
# --------------------------------------------------------------------------- 

def test_rejects_non_positive_termination():
    """Termination impedance must be positive."""
    with pytest.raises(ValueError, match="termination_ohm.*positive"):
        build_topology(
            drop_positions_m=[1.0, 3.0, 5.0],
            bus_start_m=0.0,
            bus_end_m=7.0,
            tx_drop_index=0,
            termination_ohm=0.0)
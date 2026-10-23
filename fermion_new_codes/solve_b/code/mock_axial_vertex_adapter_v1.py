"""Unrun dimensionless on-site axial-pair term adapter.

Implements MOCK_vertex_adapter_spec_v1.md only. No Hamiltonian/profile,
current, physical parameter, preparation or execution permission is selected.
The frozen core's axial_mock entry point remains disabled. Import performs
no scientific computation; the sole dependency is our existing Term API.
"""
from mock_unitary_lead_v1 import Term

GATE_PATH = '<internal>/VERTEX_gate_v1.md'
GATE_SHA256 = '3cbca4d133da6ebc02c23dcbfd5fc961233f95580abd421b53c6cfc1cbc8f8b3'
OPERATOR_STATUS = 'dimensionless W_x only; local factory; unrun'
PAIR_ORDER = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
PAIR_COEFFICIENTS = (
    (4, 0, 0, 0, 0, -4),
    (0, -8, 0, 0, 0, 0),
    (0, 0, -4, -4, 0, 0),
    (0, 0, -4, -4, 0, 0),
    (0, 0, 0, 0, -8, 0),
    (-4, 0, 0, 0, 0, 4),
)


def onsite_axial_pair_terms(site: int, total_sites: int) -> tuple[Term, ...]:
    """Return W_x=O_x+O_x^dagger, without U=lambda_X/a^3.

    Rows of K label light creation pairs, columns heavy input pairs.
    All four contracted pairing components are retained. The written
    annihilator order is H_d H_c; Term applies operators right to left.
    No site activation set, flow/clock coupling or physical run is chosen.
    The core assembler must still validate the selected basis and sector.
    """
    if type(total_sites) is not int or not 1 <= total_sites <= 6:
        raise ValueError('total_sites must be an integer from 1 through 6')
    if type(site) is not int or not 0 <= site < total_sites:
        raise ValueError('site must be an integer within total_sites')
    offset = 8 * site
    terms = []
    for row, (a, b) in enumerate(PAIR_ORDER):
        for column, (c, d) in enumerate(PAIR_ORDER):
            coefficient = PAIR_COEFFICIENTS[row][column]
            if coefficient:
                forward = Term(coefficient, (
                    (offset + 4 + a, True), (offset + 4 + b, True),
                    (offset + d, False), (offset + c, False),
                ))
                terms.extend((forward, forward.adjoint()))
    return tuple(terms)

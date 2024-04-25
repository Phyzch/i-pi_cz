import numpy as np

from ipi.engine.beads import Beads
from ipi.utils.messages import verbosity, info
from ipi.utils import units
import ipi.utils.mathtools as mt

def print_neb_instanton_geo(
    prefix, step, nbeads, natoms, names, q, pots, cell, shift, output_maker
):
    """
    adapted from instool.py: print_instanton_geo
    Alternative (but very useful) output of the instanton geometry and potential energy"""

    outfile = output_maker.get_output(prefix + "_" + str(step) + ".ener", "w")
    print("#Bead    Energy (eV)", file=outfile)
    for i in range(nbeads):
        print(
            (
                str(i)
                + "     "
                + str(units.unit_to_user("energy", "electronvolt", pots[i] - shift))
            ),
            file=outfile,
        )
    outfile.close_stream()


    # print out the coordinate in .xyz form
    unit = "angstrom"

    a, b, c, alpha, beta, gamma = mt.h2abc_deg(cell.h)

    outfile = output_maker.get_output(prefix + "_" + str(step) + ".xyz", "w")

    for i in range(nbeads):
        print(natoms, file=outfile)

        print(
            (
                "CELL(abcABC):  %f %f %f %f %f %f cell{atomic_unit}  Traj: positions{%s}   Bead:       %i"
                % (a, b, c, alpha, beta, gamma, unit, i)
            ),
            file=outfile,
        )

        for j in range(natoms):
            print(
                names[j],
                str(units.unit_to_user("length", unit, q[i, 3 * j])),
                str(units.unit_to_user("length", unit, q[i, 3 * j + 1])),
                str(units.unit_to_user("length", unit, q[i, 3 * j + 2])),
                file=outfile,
            )

    outfile.close_stream()


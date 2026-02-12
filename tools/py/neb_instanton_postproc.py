import numpy as np
import sys

import argparse
from ipi.utils.messages import verbosity, info

'''
Adapted from Instanton_postproc.py written by Y. Litman
'''

""" Reads all the information needed from a i-pi RESTART file and compute the partition functions of the reactant, transition state (TS) or
instanton according to J. Phys. Chem. Lett. 7, 437(2016) (Instanton Rate calculations) or J. Chem. Phys. 134, 054109 (2011) (Tunneling Splitting)


Syntax:    python  neb_instanton_postproc.py  <checkpoint_file> -c <case> -t  <temperature (K)> -e <ground_state_energy> 

Examples for rate calculation:
           python  Instanton_postproc.py   RESTART  -c  instanton    -t   300



Type python Instanton_postproc.py -h for more information


Relies on the infrastructure of i-pi, so the ipi package should
be installed in the Python module directory, or the i-pi
main directory must be added to the PYTHONPATH environment variable.
"""

# Chenghao Zhang. 2024.
from ipi.engine.simulation import Simulation
from ipi.utils.units import unit_to_internal, Constants
from ipi.utils.instools import red2comp
from ipi.utils.hesstools import clean_hessian
from ipi.utils.depend import dstrip
from ipi.engine.motion.instanton import SpringMapper


# UNITS
K2au = unit_to_internal("temperature", "kelvin", 1.0)
kb = Constants.kb
hbar = Constants.hbar
eV2au = unit_to_internal("energy", "electronvolt", 1.0)
cal2au = unit_to_internal("energy", "cal/mol", 1.0)
cm2au = unit_to_internal("frequency", "hertz", 1.0) * 3e10


# --------- parse input from command line ----------
def parse_input():
    # INPUT
    parser = argparse.ArgumentParser(
        description="""Post-processing routine in order to obtain different quantities from an instanton (or instanton related) calculation. 
        These quantities can be used for the calculation of rates or tunneling splittings in the instanton approximation."""
    )
    parser.add_argument("input", help="Restart file")  # positional argument.
    parser.add_argument(
        "-c",
        "--case",
        default=False,
        help="Type of the calculation to analyse. Options: 'instanton', 'reactant' or 'TS'.",
    )
    parser.add_argument(
        "-t", "--temperature", type=float, default=0.0, help="Temperature in K."
    )
    parser.add_argument(
        "-asr",
        "--asr",
        default="poly",
        help="Removes the zero frequency vibrational modes depending on the symmerty of the system",
    )
    parser.add_argument(
        "-e", "--energy_shift", type=float, default=0.0, help="energy of ground state in eV"
    )
    parser.add_argument(
        "-f",
        "--filter",
        default=[],
        help="List of atoms indexes to filter (i.e. eliminate its componentes in the position,mass and hessian arrays. It is 0 based.",
        type=int,
        action="append",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        default=False,
        action="store_true",
        help="Avoid the Qvib and Qrot calculation in the instanton case.",
    )

    parser.add_argument(
        "-freq",
        "--freq_reac",
        default= None,
        help="List of frequencies of the minimum. Required for splitting calculation."
    )

    args = parser.parse_args() # convert arguments to object and assign arguments as attributes of the namespace. return namespace. the name is specified by --.
    inputt = args.input
    case = args.case
    temp = args.temperature * K2au
    asr = args.asr
    V00 = args.energy_shift
    filt = args.filter
    quiet = args.quiet
    input_freq = args.freq_reac 
    Verbosity = verbosity
    Verbosity.level = "quiet"

    if case not in list(["instanton"]):
            raise ValueError(
            "We can not indentify the case. The valid cases is: 'instanton'"
        )

    if asr not in list(["poly", "linear", "crystal", "none"]):
        raise ValueError(
            "We can not indentify asr case. The valid cases are: 'poly', 'crystal' , 'linear' and 'none'"
        )

    if asr == "poly":
        nzeros = 6
    elif asr == "crystal":
        nzeros = 3
    else:
        nzeros = 0
    if asr == "linear":
        raise NotImplementedError("Sum rules for linear molecules is not implemented")

    if args.temperature == 0.0:
        raise ValueError("The temperature must be specified.'")
    
    return args, inputt, case, temp, asr, V00, filt, quiet, Verbosity, nzeros, input_freq 

# ----- read instanton data from check point file (RESTART) -------

def Read_instanton_data(inputt, V00, temp, quiet, asr, input_freq):
    '''
    Read data from RESTART file to carry out simulation.
    '''
    print("\nWe are ready to start. Reading {} ... (This can take a while)".format(inputt))

    simulation = Simulation.load_from_xml(
    inputt, custom_verbosity="quiet", request_banner=False, read_only=True
    )

    neb_beads = simulation.syslist[0].motion.beads.copy()
    m = simulation.syslist[0].motion.beads.m.copy()

    nbeads = simulation.syslist[0].motion.optarrays["instanton_bead_number"]  # number of half ring polymer beads / linear ring polymer beads.
    natoms = simulation.syslist[0].motion.beads.natoms

    hessian = simulation.syslist[0].motion.optarrays["instanton_hessian"]
    temp2 = simulation.syslist[0].motion.optarrays["instanton_temperature"]  # in atomic unit.
    pots = simulation.syslist[0].motion.optarrays["instanton_bead_pot"]   # pots for half ring polymer for rate calculation.
    pos = simulation.syslist[0].motion.optarrays["instanton_bead_q"]

    # type of calculation, 
    try:
        cal_type = simulation.syslist[0].motion.options["cal_type"]
    except:
        # if cal_type is unset, this by default is rate calculation. 
        cal_type = "rate"

    # V0 = simulation.syslist[0].motion.optarrays["energy_shift"]
 
    if V00 != 0.0:
        print("Use the energy shift (reactant energy) with the provided values from terminal (unit eV)")
        V0 = V00 * eV2au 
    else:
        raise("must provide the energy shift: the ground state energy of the reactant. use -e <energy(eV)> \
              (if value is 0, use small number, 1e-6 for example)")
        
    
    if np.absolute(temp - temp2) / K2au > 2:
        print(
            "\n Mismatch between provided temperature and temperature in the calculation"
        )
        sys.exit()
    
    # process hessians.
    # generate m3 for half ring polymer
    m3_one_bead = np.repeat(dstrip(m), 3)

    
    if cal_type == "rate":
        h0 = red2comp(hessian, nbeads, natoms)
        full_rp_beads_q, nbeads, hessian2 = get_double(pos, nbeads, natoms, h0)  # get position, nbeads and hessian for full ring polymer. (now nbeads is for full ring polymer)

        pos = full_rp_beads_q
        hessian = hessian2 

        m3_half_rp = np.tile(m3_one_bead, ( int(nbeads / 2) , 1))
        # now generate m3 for full ring polymer
        m3 = np.concatenate((m3_half_rp, m3_half_rp), axis = 0)

        omega2 = (temp * nbeads * kb / hbar) ** 2
        pots = np.concatenate((pots, np.flipud(pots)), axis= 0)

        h = 0
        spring = SpringMapper.spring_hessian(
            natoms, nbeads, m3_one_bead, omega2, mode = "full"
        )
        h = np.add(hessian, spring)

    elif cal_type == "splitting":
        if input_freq is None:
            print(
                'Please provide a name of the file containing the list of the frequencies for the minimum using "-freq" flag'
            )
            print(" You can generate that file using this script in the case reactant.")
            sys.exit()
        
        print(f"Linear ring polymer for splitting calculation. Has {nbeads} beads.")

        # effective temperature for linear ring polymer beads:
        effective_temp = temp * 2

        omega2 = (effective_temp * nbeads * kb / hbar) ** 2
        
        m3 = np.tile(m3_one_bead, ( int(nbeads / 2) , 1))

        h0 = red2comp(hessian, int(nbeads / 2), natoms)
        
        # TODO: Rewrite the code that output Hessian of the reflected beads. 
        permute_index_list = read_atom_permute_index(natoms)
        h0 = hessian_reflect(int(nbeads / 2), natoms, h0, permute_index_list)

        spring= SpringMapper.spring_hessian(
            natoms, nbeads, m3_one_bead, omega2, mode="splitting"
        )
        h = np.add(h0, spring)
        if asr != "none":
            print(
                "We are changing asr to none since we consider a fixed ended linear polymer for the post-processing"
            )
            asr = "none"

    return (neb_beads, m, nbeads, natoms, temp2, 
            pots, pos, 
            V0, h,  m3, 
            omega2, asr,
            cal_type)


# -----Some functions-----------------


def get_double(q0, nbeads0, natoms, h0):
    """Takes nbeads, positions and hessian (only the 'physcal part') of the half polymer and
    returns the equivalent for the full ringpolymer.
    :param: q0: 1d position array for half ring-polymer.
    :param: nbeads0: number of beads for half ring-polymer
    :param: natoms: number of atoms in the system for each bead (replica)
    :param: h0: hessian matrix for half ring-polymer
    :return: q, nbeads, h
    q: 1d position array for full ring-polymer
    nbeads: number of beads for full ring-polymer
    h: hessian for full ring-polymer
    """
    q = np.concatenate((q0, np.flipud(q0)), axis=0)  # flip the 1d array for half ring-polymer and then concatenate
    nbeads = 2 * nbeads0 # double beads number for full ring-polymer
    ii = 3 * natoms
    iii = 3 * natoms * nbeads0

    h = np.zeros((iii * 2, iii * 2))  # hessian matrix for full ring-polymer
    h[0:iii, 0:iii] = h0

    # diagonal block
    for i in range(nbeads0):
        x = i * ii + iii
        y = ((nbeads0 - 1) - i) * ii  # fold back, the bead in the original half ring-polymer that is the image of new ring polymer bead at position i.
        h[x : x + ii, x : x + ii] = h0[y : y + ii, y : y + ii]

    return q, nbeads, h

def read_atom_permute_index(natoms):
    pairs = []
    with open("atom_permute_index.txt", "r") as f:   # replace with your filename
        for line in f:
            # strip whitespace, skip empty lines
            parts = line.strip().split()
            if not parts:
                continue
            # convert to integers and store as tuple
            pair = tuple(map(int, parts))
            pairs.append(pair)
            first, second = pair 
            if (first >= natoms or second >= natoms):
                raise ValueError(f"the atom index exceeds natoms {pair}")
    
    return pairs 


def hessian_reflect(nbeads0, natoms, h0, permute_index_list):
    """
    For the tunneling splitting calculation. 
    Take zero temperature instanton path at one side of the barrier, 
    use the hessian at one side to generate hessian at another side. 
    :param: nbeads0: number of beads for instanton path at one side.
    :param: natoms: number of atoms in the system for each bead.
    :param: h0: hessian matrix for beads at one side.
    :param: permute_index_list: index for atoms to perform the permutation.
            example: [[0, 1], [2, 5], [3, 4], [8, 9]]
    """
    nbeads = nbeads0 * 2 
    ii = 3 * natoms 
    iii = 3 * natoms * nbeads0 

    h = np.zeros((iii * 2, iii * 2)) # hessian matrix for beads at both ends.
    h[0: iii, 0: iii] = h0 

    h_bead = np.zeros((3 * natoms, 3 * natoms))
    h_bead_permute = np.zeros((3 * natoms, 3 * natoms))
    for bead_index in range(nbeads0):
        h_bead = np.copy(h0[bead_index * ii : (bead_index + 1) * ii,
                            bead_index * ii : (bead_index + 1) * ii])
        # permute the atom 
        h_bead_permute = np.copy(h_bead)
        for pair in permute_index_list:
            first, second = pair 
            h_bead_permute[first * 3 : (first + 1) * 3, :] = h_bead[second * 3: (second + 1) * 3, :]
            h_bead_permute[second * 3 : (second + 1) * 3, :] = h_bead[first * 3 : (first + 1) * 3, :]
        h_bead_row_permuted = np.copy(h_bead_permute)
        for pair in permute_index_list:
            first, second = pair 
            h_bead_permute[:, first * 3 : (first + 1) * 3] = h_bead_row_permuted[:, second * 3 : (second + 1) * 3]
            h_bead_permute[:, second * 3 : (second + 1) * 3] = h_bead_row_permuted[:, first * 3: (first + 1) * 3]
        
        h_bead = np.copy(h_bead_permute)

        # reflect the x axis. Assume the the instanton is symmetric along x axis.
        for i in range(natoms):
            h_bead[i * 3, :] = - h_bead[i * 3, :]
            h_bead[:, i * 3] = - h_bead[:, i * 3]
        
        h[ii * nbeads - (bead_index + 1) * ii : ii * nbeads - bead_index * ii, 
          ii * nbeads - (bead_index + 1) * ii : ii * nbeads - bead_index * ii] = h_bead

    return h 

def spring_pot(nbeads, q, omega2, m3):
    '''
    omega2: square of angular velocity
    m3: mass
    '''
    e = 0.0
    for i in range(nbeads - 1):
        dq = q[i + 1, :] - q[i, :]
        e += omega2 * 0.5 * np.dot(m3[0] * dq, dq)
    return e


def Filter(pos, h, natoms, m, m3, filt):
    '''
    filter out atoms that not included in calculation.
    :param: filt: index of atoms to be filtered.
    :param: hessian in 1 bead.
    :param: m: mass
    :param: m3: size[3 * atom]. mass is the same along 3 dimen. m3 is the 1d matrix.
    :natom: number of atoms.
    '''
    filt3 = []
    for i in filt:
        filt3.append(3 * i)
        filt3.append(3 * i + 1)
        filt3.append(3 * i + 2)
    pos = np.delete(pos, filt3, axis=1) # [[pos]]

    aux = np.delete(h, filt3, axis=1)
    h = np.delete(aux, filt3, axis=0)  # delete row & column from hessian matrix in a single bead.

    m = np.delete(m, filt, axis=0)  # m: natoms, 1d array.
    m3 = np.delete(m3, filt3, axis=1)  # m3 size:  [3 * natoms]. it has a funny structure [[m3_data]], but it's actually 1d array
    natoms = natoms - len(filt)
    return pos, h, natoms, m, m3


# def get_rp_freq(w0,nbeads,temp,asr=None,mode='rate',nzero=0):


def get_rp_freq(w0, nbeads, temp, mode="rate"):
    """
    Compute the ring polymer frequencies for multidimensional harmonic potential
    defined by the frequencies w0.
    :param: w0: square of frequency of harmonic potential
    :param: nbeads: number of beads for half ring-polymer.
    :param: temp: temperature
    omega^2 = omega_0 ^2 + [2/(betaP * hbar) * sin(pi * |k|/N)]^2 for mode q_k. here mode q_k = 1/sqrt{N} sum_j e^{2ikj/N} q_j
    """
    hbar = 1.0
    kb = 1
    betaP = 1 / (kb * nbeads * temp)
    factor = betaP * hbar
    w = 0.0
    ww = []

    if np.amin(w0) < 0.0:
        print("@get_rp_freq: We have a negative frequency, something is going wrong.")
        sys.exit()

    if mode == "rate":
        # for i in range(nzero):
        #    for k in range(1, nbeads):
        #        w += np.log(factor*np.sqrt( 4./(betaP*hbar)**2 * np.sin(np.absolute(k)*np.pi/nbeads)**2 )
        #        # Yes, for each K w is nbeads

        for n in range(w0.size):
            for k in range(nbeads):
                if w0[n] == 0 and k == 0:
                    continue

                physical_freq = np.sqrt(
                        4.0
                        / (betaP * hbar) ** 2
                        * np.sin(np.absolute(k) * np.pi / nbeads) ** 2
                        + w0[n]
                    )

                w += np.log(
                    factor
                    * physical_freq
                )   # correct formula is log(2 * sinh(factor & physical_freq / 2))
                # note the w0 is the eigenvalue ( the square of the frequency )
        return w

    elif mode == "splitting":
        for n in range(w0.size):
            for k in range(nbeads):
                # note the w0 is the eigenvalue ( the square of the frequency )
                ww = np.append(
                    ww,
                    np.sqrt(
                        4.0
                        / (betaP * hbar) ** 2
                        * np.sin((k + 1) * np.pi / (2 * nbeads + 2)) ** 2
                        + w0[n]
                    ),
                )
        return np.array(ww)
    else:
        print("We can't indentify the mode")
        sys.exit()


def print_instanton_path(nbeads, natoms, names, bead_q ,pots, filename = "instanton_path.xyz"):
    '''
    output the instanton path in the format of:  natoms // energy // atom x, y, z.
    :param: beads: bead object in i-pi
    :param: pots: potential of each bead.
    '''
    q = np.copy(bead_q)  # coordinate

    q_au_to_angstrom = 0.529
    q = q * q_au_to_angstrom  # transform to unit of angstrom.

    print("instanton path is printed to file: " + str(filename))
    with open(filename, "w") as f:
        for bead_index in range(nbeads):
            f.write("                    " + str(natoms) + "\n")  # natoms
            energy = pots[bead_index]
            f.write("energy=   " + str(energy)+"\n")  # energy
            for atom_index in range(natoms):
                name = names[atom_index]
                f.write(name + "          ")  # name
                f.write( str(q[bead_index][atom_index * 3] ) + "  "
                        + str(q[bead_index][atom_index * 3 + 1]) + "  "
                        + str(q[bead_index][atom_index * 3 + 2]) + "\n"
                        )  # coordinate


def compute_instanton_rate_or_splitting():
    args, inputt, case, temp, asr, V00, filt, quiet, Verbosity, nzeros, input_freq = parse_input()

    (neb_beads, m, nbeads, natoms, temp2, 
     pots, pos, 
     V0, h, m3,  omega2, asr, cal_type) = Read_instanton_data(inputt, 
                                                         V00, 
                                                         temp, 
                                                         quiet,
                                                         asr,
                                                         input_freq)

    if cal_type == "rate":
        beta = 1.0 / (kb * temp)
        betaP = 1.0 / (kb * (nbeads) * temp)
    else:
        beta = 1.0 / (kb * temp)  
        betaP = beta / (2 * nbeads) # the linear ring polymer.

    print(("\nTemperature: {} K".format(temp / K2au)))
    print(("NBEADS: {}".format(nbeads)))
    print(("atoms:  {}".format(natoms)))
    print(("ASR:    {}".format(asr)))
    print(("1/(betaP*hbar) = {:8.5f}".format((1 / (betaP * hbar)))))

    if not quiet:
        print("Diagonalization ... \n\n")
        if cal_type == "rate":
            m3_for_hessian = m3
        else:
            m3_for_hessian = np.repeat(m3, repeats= 2, axis= 0) 
        hess_eigval, hess_eigvec, detI = clean_hessian(h, pos, natoms, nbeads, m, m3_for_hessian, asr, mofi=True)  # remove the  translational and rotational modes.
        print("Final lowest 10 frequencies (cm^-1)")
        d10 = np.array2string(
            np.sign(hess_eigval[0:10]) * np.absolute(hess_eigval[0:10]) ** 0.5 / cm2au,
            precision=2,
            max_line_width=100,
            formatter={"float_kind": lambda x: "%.2f" % x},
        )
        print(("{}".format(d10)))

        # print conditional number
    
    # print instanton path for half ring polymer
    nbeads_to_print = int(nbeads/2) 
    print_instanton_path(nbeads_to_print, natoms, neb_beads.names, pos, pots)

    if  cal_type == "rate":
        Qtras = ((np.sum(m)) / (2 * np.pi * beta * hbar**2)) ** 1.5  # see eq.(58) in review paper: https://doi.org/10.1080/0144235X.2018.1472353

        if asr == "poly" and not quiet:
            Qrot = (8 * np.pi * detI / ((hbar) ** 6 * (betaP) ** 3)) ** 0.5
            Qrot /= nbeads**3  # See eq. 60 in review paper : https://doi.org/10.1080/0144235X.2018.1472353
        else:
            Qrot = 1.0

        if not quiet:
            del_freq = np.sign(hess_eigval[1]) * np.absolute(hess_eigval[1]) ** 0.5 / cm2au
            print("Deleted frequency: {:8.3f} cm^-1".format(del_freq))   # zero mode frequency is deleted. hess_eigval[0] is imaginary freq. (unstable mode)

            if asr != "poly":
                print("WARNING asr != poly")
                print("First 10 eigenvalues")
                ten_eigv = np.sign(hess_eigval[0:10]) * np.absolute(hess_eigval[0:10]) ** 0.5 / cm2au
                print("{}".format(ten_eigv))
                print(
                    "Please check that this you don't have any unwanted zero frequency"
                )

            logQvib = (
                -np.sum(np.log(betaP * hbar * np.sqrt(np.absolute(np.delete(hess_eigval, 1)))))
                + nzeros * np.log(nbeads)
                + np.log(nbeads)     # See eq. 60 in review paper : https://doi.org/10.1080/0144235X.2018.1472353
            )

            logQvib1 = (
                -np.sum(np.log(betaP * hbar * np.sqrt(np.absolute(hess_eigval[3:]))))
                + nzeros * np.log(nbeads)
                + np.log(nbeads)     # See eq. 60 in review paper : https://doi.org/10.1080/0144235X.2018.1472353
            )
            print(f"For debug: logQvib with contribution excluding first 3 eigenvalues {logQvib1}")

        else:
            logQvib = 0.0

        pos_half_rp = pos[: int(len(pos) / 2), :]
        m3_half_rp = m3[:int(len(m3) / 2), :]
        BN = 2 * np.sum(m3_half_rp[1:, :] * (pos_half_rp[1:, :] - pos_half_rp[:-1, :]) ** 2)  # 2 * : account for full ring-polymer
        action1 = (pots.sum() - nbeads * V0) * 1.0 / (temp * nbeads * kb)   # \beta \hbar \sum(Vi - V0) potential contribution to the action
        action2 = spring_pot(nbeads, pos, omega2, m3) / (temp * nbeads * kb)  # free spring term contribution to the action.

        print(
            "\nWe are done. Instanton rate. Nbeads {} (diff only {})".format(
                nbeads, nbeads / 2
            )
        )

        print("V0  {} eV ( {} Kcal/mol) ".format(V0 / eV2au, V0 / cal2au / 1000))

        print(
            "   {:8s} {:8s}  | {:11s} | {:11s} | {:11s} | {:8s} ( {:8s},{:8s} ) |".format(
                "BN",
                "(BN*N)",
                "Qt(bohr^-3)",
                "Qrot",
                "log(Qvib*N)",
                "S/hbar",
                "S1/hbar",
                "S2/hbar",
            )
        )
        print(
            "{:8.3f} ( {:8.3f} ) | {:11.3f} | {:11.3f} | {:11.3f} | {:8.3f} ( {:8.3f} {:8.3f} ) |".format(
                BN,
                BN * nbeads,
                Qtras,
                Qrot,
                logQvib,
                (action1 + action2),
                action1,
                action2,
            )
        )
        print("\n\n")
    elif cal_type == "splitting":
        # read frequency of minimum. 
        out = open(input_freq, "r")
        d_min = np.zeros(natoms * 3)
        aux = out.readline().split()
        if len(aux) != (natoms * 3):
            print(("We are expecting {} frequencies.".format((natoms * 3 - 6))))
            print(("instead we have read  {}".format(len(aux))))
        for i in range((natoms * 3)):
            d_min[i] = float(aux[i])
        d_min = d_min.reshape((natoms * 3))
        out.close()

        # effective temperature for splitting calculation.
        effective_temp = temp * 2

        ww = get_rp_freq(d_min, nbeads, effective_temp, mode="splitting")
        react = np.sum(np.log(ww))

        assert len(pots) == int(nbeads / 2)
        
        action1 = (pots.sum() - int(nbeads / 2) * V0) * 1 / (effective_temp * nbeads * kb) 
        action1 = action1 * 2 

        action2 = spring_pot(int(nbeads/2), pos, omega2, m3) / (effective_temp * nbeads * kb)
        action2 = action2 * 2

        action = action1 + action2
        if action / hbar > 5.0:
            print(
                f"WARNING, S/h: {action / hbar} seems to big. Probably a proper energy shift is missing."
            )
        
        BN = np.sum(m3[1:, :] * (pos[1:, :] - pos[:-1, :]) ** 2)
        BN = BN * 2

        if not quiet:
            inst = np.sum(np.log(np.sqrt(np.absolute(np.delete(hess_eigval, [1])))))
            phi = np.exp(inst - react)
        else:
            phi = 1 
        
        tetaphi = (
            betaP * hbar * np.sqrt(action / (2 * hbar * np.pi)) * np.exp(-action / hbar)
        )
        teta = tetaphi / phi
        h = -teta / betaP  # h is half of the tunneling splitting.

        print("\n\nWe are done")
        print("Nbeads {}, betaP {} a.u.,hbar {} a.u".format(nbeads, betaP, hbar))
        print("")
        print("V0  {} eV ( {} Kcal/mol) ".format(V0 / eV2au, (V0 / cal2au) / 1000))
        print(
            "S1/hbar {} ,S2/hbar {} ,S/hbar {}".format(
                action1 / hbar, action2 / hbar, action / hbar
            )
        )
        print("BN {} a.u.".format(BN))
        print(
            "BN/(hbar^2 * betaN)  {}  (should be same as S/hbar) ".format(
                (BN / ((hbar**2) * betaP))
            )
        )
        print("")
        if quiet:
            print("phi is not computed because you specified the quiet option")
            print(
                ("We can provied only Tetaphi which value is {} a.u. ".format(tetaphi))
            )
        else:
            print(("phi {} a.u.   Teta {} a.u. ".format(phi, tetaphi / phi)))
            print(
                "Tunnelling splitting matrix element (h)  {} a.u ({} cm^-1)".format(
                    h, h / cm2au
                )
            )

            print(
                "Tunneling splitting for symmetric well (Delta = 2h) {} a.u. ({} cm^-1)".format(
                    h * 2, h * 2 / cm2au
                )
            )
    else:
        print("We can not recongnize the mode.")
        sys.exit()


compute_instanton_rate_or_splitting()
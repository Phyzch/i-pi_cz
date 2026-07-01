"""Functions that deal with hessian transformations and calculations"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np
import scipy 
from ipi.utils.messages import verbosity, info
import os
from ipi.engine.beads import Beads
from ipi.utils.depend import dstrip

def get_dynmat(h, m3, nbeads=1):
    """Computes the dynamical matrix.
    If nbeads > 1 the reduced form with shape = (3*natoms, 3*natoms*nbeads) is expected
    """

    # Check dimensions

    if h.shape != (m3.shape[1], m3.shape[1] * nbeads):
        print(h.shape, m3.shape)
        raise ValueError(
            "@get_dynmat: The provided hessian hasn't the proper dimension (3*natoms, 3*natoms*nbeads) "
        )

    ism = m3.reshape((1, -1)) ** (-0.5)
    ismT = m3[0].reshape((-1, 1)) ** (-0.5)

    dynmat = np.zeros(h.shape)
#    for i in range(nbeads):
    dynmat = np.multiply(ismT, np.multiply(h, ism))
    return dynmat


def clean_hessian(h, q, natoms, nbeads, m, m3, asr, mofi=False, neigs = None):
    """
    Removes the translations and rotations modes.
    IN  h      = hessian (3*natoms*nbeads, 3*natoms*nbeads)
        q      = positions
        natoms = number of atoms
        nbeads = number of beads
        m      = mass vector, one value for each atom
        m3     = mass vector, one value for each degree of freedom
        mofi   = An optional boolean which decides whether the det(M_of_I)
                 is returned or not
    OUT d      = non zero eigenvalues of the dynmatrix
        w      = eigenvectors without external modes

    #Adapted from ipi/engine/motion/phonons.py apply_asr"""

    info(" @clean_hessian: asr = %s " % asr, verbosity.medium)
    # Set some useful things
    ii = natoms * nbeads
    mm = np.zeros((nbeads, natoms))
    for i in range(nbeads):
        mm[i] = m       # m here is a 1d array with size [natoms]. Therefore, this repeat mass nbeads times.
    mm = mm.reshape(ii)
    ism = m3.reshape((ii * 3, 1)) ** (-0.5)  # ism: inverse square root of m3. 
    dynmat = np.multiply(ism.T, np.multiply(h, ism))  # dynmat is mass weighted hessian.
    # ismm = np.outer(ism, ism)
    # dynmat = np.multiply(h, ismm)

    if asr == "none" or asr is None:
        hm = dynmat
    else:
        # Computes the centre of mass.
        com = np.dot(np.transpose(q.reshape((ii, 3))), mm) / mm.sum()  # for 3d array, q should be [nbeads, natoms, 3]
        qminuscom = q.reshape((ii, 3)) - com  # recentered coordinate.
        ism = ism.flatten()

        if asr == "poly":
            # Computes the moment of inertia tensor.
            moi = np.zeros((3, 3), float)
            for k in range(ii):
                moi -= (
                    np.matmul(
                        np.cross(qminuscom[k], np.identity(3)),
                        np.cross(qminuscom[k], np.identity(3)),
                    )
                    * mm[k]
                )

            I, U = np.linalg.eig(moi)  # I: eigenvalue, U: eigenvector. eigenvector is the rotational axis. U[:,i] is eigenvector i.
            R = np.dot(qminuscom, U)  # coordinate for atoms with the principle axis as rotational axis (rotational frame)
            D = np.zeros((6, 3 * ii), float)

            # Computes the vectors along translations and rotations. 
            # See https://en.wikipedia.org/wiki/Eckart_conditions and https://web.archive.org/web/20210509233020/https://gaussian.com/wp-content/uploads/dl/vib.pdf
            # Translations
            D[0] = np.tile([1, 0, 0], ii) / ism
            D[1] = np.tile([0, 1, 0], ii) / ism
            D[2] = np.tile([0, 0, 1], ii) / ism
            # Rotations
            for i in range(3 * ii):
                iatom = i // 3
                idof = np.mod(i, 3)
                D[3, i] = (R[iatom, 1] * U[idof, 2] - R[iatom, 2] * U[idof, 1]) / ism[i]  # this is rotational motion for each atom in coordinate axis.
                D[4, i] = (R[iatom, 2] * U[idof, 0] - R[iatom, 0] * U[idof, 2]) / ism[i]
                D[5, i] = (R[iatom, 0] * U[idof, 1] - R[iatom, 1] * U[idof, 0]) / ism[i]

            for k in range(6):
                D[k] = D[k] / np.linalg.norm(D[k])
            # Computes the transformation matrix.
            transfmatrix = np.eye(3 * ii) - np.dot(D.T, D)  # this is the projection operator that projects out translation and rotation dof.
            hm = np.dot(transfmatrix.T, np.dot(dynmat, transfmatrix))

        elif asr == "crystal":
            # Computes the vectors along translations.
            # Translations
            D = np.zeros((3, 3 * ii), float)
            D[0] = np.tile([1, 0, 0], ii) / ism
            D[1] = np.tile([0, 1, 0], ii) / ism
            D[2] = np.tile([0, 0, 1], ii) / ism

            for k in range(3):
                D[k] = D[k] / np.linalg.norm(D[k])
            # Computes the transformation matrix.
            transfmatrix = np.eye(3 * ii) - np.dot(D.T, D)
            hm = np.dot(transfmatrix.T, np.dot(dynmat, transfmatrix))

    # Symmetrize to use linalg.eigh
    hmT = hm.T
    hm = (hmT + hm) / 2.0

    if neigs == None:
        d, w = np.linalg.eigh(hm)  # as we have projected out translation & rotational dof, there should be several eig close to 0 corresponds to such mode.
    else:
        if asr == 'poly':
            neigs_to_solve = neigs + 6
        elif asr == 'crystal':
            neigs_to_solve = neigs + 3
        else:
            neigs_to_solve = neigs 

        d, w = scipy.sparse.linalg.eigsh(hm, k = neigs_to_solve, which = 'SA')  # solve smallest eigenvalues up to neigs_to_solve

    # Count
    dd = (
        np.sign(d) * np.absolute(d) ** 0.5 / (2 * np.pi * 3e10 * 2.4188843e-17)
    )  # convert to cm^-1

    # Zeros
    cut0 = 0.01  # Note that dd[] units are cm^1
    condition = np.abs(dd) < cut0
    nzero = np.extract(condition, dd)

    if asr == "poly" and nzero.size != 6:
        info(
            " @GEOP: Warning, we have %d 'zero' frequencies" % nzero.size, verbosity.low
        )

    if asr == "crystal" and nzero.size != 3:
        info(
            " @GEOP: Warning, we have %d 'zero' frequencies" % nzero.size, verbosity.low
        )

    # Negatives
    cutNeg = -4  # Note that dd[] units are cm^1
    condition = dd < cutNeg
    nneg = np.extract(condition, dd)
    info(
        " @Clean hessian: We have %d 'neg' frequencies " % (nneg.size), verbosity.medium
    )

    # Now eliminate external degrees of freedom from the dynmatrix

    if nzero.size > 0:
        if np.linalg.norm(nzero) > cut0:
            info(
                " Warning @Clean hessian: We have deleted %d 'zero' frequencies "
                % (nzero.size),
                verbosity.high,
            )
            info(
                " but the norm is greater than 0.01 cm^-1.  This should not happen.",
                verbosity.high,
            )

        d = np.delete(d, list(range(nneg.size, nneg.size + nzero.size)))  # delete zero modes eigval. the eigenvalues and eigenvectors here are ordered
        w = np.delete(w, list(range(nneg.size, nneg.size + nzero.size)), axis=1) # delete zero modes eigvec.

    if mofi:
        if asr == "poly":
            return d, w, np.prod(I)
        else:
            return d, w, 1.0
    else:
        return d, w


# for debug purpose
def get_hessian_beadwise(dbead, dcell, dforces, x0, natoms, nbeads, fixatoms= [], d= 0.01):
    """Compute the physical hessian for each bead separately. This is only for debug purpose. 
    IN     dbead     = position of the bead (natoms, 3)
           dforces   = function to evaluate forces given positions
           natoms   = number of atoms
           nbeads   = number of beads
           fixatoms = indexes of fixed atoms
           d        = displacement

    OUT    h       = physical hessian for each bead (nbeads, natoms*3, natoms*3)
    """
    ndofs = natoms * 3
    h = np.zeros((nbeads, ndofs, ndofs), float)
    
    fixdofs = list()
    for i in fixatoms:
        fixdofs.extend([3 * i, 3 * i + 1, 3 * i + 2])  # add all fixdofs attached to fix atoms.

    def create_single_bead_and_forces(dbead, dforces):
        bead = Beads(dbead.natoms, 1)
        bead.m = dbead.m
        bead.names[:] = dbead.names
        cell = dcell.copy() 
        forces= dforces.copy(bead, cell)

        return bead, forces 
    
    def gm(bead, forces, q):
        assert bead.q.shape == q.shape
        bead.q[:] = q
        g = - np.copy(forces.f) 
        return g 

    bead, forces = create_single_bead_and_forces(dbead, dforces)
    for i in range(nbeads):
        info(" @get_hessian_beadwise: Computing hessian for bead %d of %d" % (i+1, nbeads), verbosity.low)
        bead_x0 = dstrip(x0[i].copy())
        bead_x0 = bead_x0[np.newaxis, :] 

        for j in range(ndofs):
           if j in fixdofs:
               continue
           else:
               x = bead_x0.copy()
               # PLUS
               x[:, j] = bead_x0[:, j] + d 
               g1 = gm(bead, forces, x)

               # Minus
               x[:, j] = bead_x0[:, j] - d 
               g2 = gm(bead, forces, x)

               # combine:
               g = (g1 - g2)/ (2 * d)

               h[i, j] = g.flatten()

    h = np.transpose(h, (1, 0, 2)).reshape((ndofs, nbeads * ndofs)) 
    
    return h

def get_hessian(
    gm, x0, natoms, nbeads=1, fixatoms=[], d=0.001, new_disc=False, friction=False
):
    """Compute the physical hessian given a function to evaluate energy and forces (gm).
    The intermediate steps are written as a temporary files so the full hessian calculations is only ONE step.

    IN     gm       = gradient mapper
           x0       = position vector
           natoms   = number of atoms
           nbeads   = number of beads
           fixatoms = indexes of fixed atoms
           d        = displacement

    OUT    h       = physical hessian ( (natoms-len(fixatoms) )*3 , nbeads*( natoms-len(fixatoms) )*3)
    """

    info(" @get_hessian: Computing hessian", verbosity.low)
    fixdofs = list()
    for i in fixatoms:
        fixdofs.extend([3 * i, 3 * i + 1, 3 * i + 2])  # add all fixdofs attached to fix atoms.
    ii = natoms * 3
    activedof = np.delete(np.arange(ii), fixdofs)
    ncalc = ii - len(fixdofs)  #for each bead, # of free dofs need calculation.
    if x0.size != natoms * 3 * nbeads:
        raise ValueError(
            "The position vector is not consistent with the number of atoms/beads."
        )

    h = np.zeros((ii, ii * nbeads), float)
    if friction:
        eta_h = np.zeros((nbeads, ii, ii, ii), float)

    # Check if there is a temporary file:
    i0 = -1

    for i in range(ii, -1, -1):
        try:
            b = np.loadtxt("hessian_" + str(i) + ".tmp")
        except IOError:
            pass
        else:
            h[:, :] = b[:, :]
            i0 = i
            print(("We have found a temporary file ( hessian_" + str(i) + ".tmp). "))
            if (
                b.shape == h.shape
            ):  # Check that the last temporary file was properly written
                break
            else:
                continue
    if friction:
        for i in range(ii, -1, -1):
            try:
                b = np.loadtxt("hessianEta_" + str(i) + ".tmp")
            except IOError:
                pass
            else:
                eta_h[:] = b.reshape((nbeads, ii, ii, ii))
                i0 = i
                print(
                    (
                        "We have found a temporary file ( hessianEta_"
                        + str(i)
                        + ".tmp). "
                    )
                )
                if (
                    b.shape == eta_h.shape
                ):  # Check that the last temporary file was properly written
                    break
                else:
                    continue

    # Start calculation:
    for j in range(i0 + 1, ii):
        if j in fixdofs:
            continue
        else:
            ndone = len(activedof[activedof < j])
            info(
                " @get_hessian: Computing hessian: %d of %d" % (ndone + 1, ncalc),
                verbosity.low,
            )
            x = x0.copy()

            # PLUS
            x[:, j] = x0[:, j] + d
            e, f1 = gm(x, new_disc)
            if friction:
                eta1 = gm.eta

            # Minus
            x[:, j] = x0[:, j] - d
            e, f2 = gm(x, new_disc)
            if friction:
                eta2 = gm.eta

            # COMBINE
            g = (f1 - f2) / (2 * d)

            # g = five_point_stencil(x0, gm, j, d)  # use five point stencil to compute the derivative.

            h[j, :] = g.flatten()
            f = open("hessian_" + str(j) + ".tmp", "w")
            np.savetxt(f, h)
            f.close()

            # if friction:
            #     eta_h[:, :, :, j] = (eta1 - eta2) / (2 * d)
            #     f = open("hessianEta_" + str(j) + ".tmp", "w")
            #     np.savetxt(f, eta_h.flatten())
            #     f.close()

    u, g = gm(x0)  # Keep the mapper updated

    for i in range(ii):
        try:
            os.remove("hessianEta_" + str(i) + ".tmp")
        except OSError:
            pass

        try:
            os.remove("hessian_" + str(i) + ".tmp")
        except OSError:
            pass

    if not friction:
        return h
    else:
        return h, eta_h

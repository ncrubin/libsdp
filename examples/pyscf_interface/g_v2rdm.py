"""
Driver for variational two-electron reduced-density matrix method using only the G-matrix
"""
from typing import List, Tuple
import itertools
import numpy as np
from numpy import einsum

import sys
sys.path.insert(0, '../../.')

import libsdp

import pyscf

def d1a_trace(nmo: int, nalpha: int, block_num: int) -> Tuple[List, List]:
    # Tr(D1a)
    block_number = [block_num] * nmo
    row = list(range(1, nmo + 1))
    column = list(range(1, nmo + 1))
    value = [1] * nmo
    bvals = [nalpha]
    F = libsdp.sdp_matrix()
    F.block_number = block_number
    F.row          = row
    F.column       = column
    F.value        = value
    return [F], bvals

def d1b_trace(nmo, nbeta, block_num) -> Tuple[List, List]:
    # Tr(D1b)
    block_number = [block_num] * nmo
    row = list(range(1, nmo + 1))
    column = list(range(1, nmo + 1))
    value = [1] * nmo
    bvals = [nbeta]
    F = libsdp.sdp_matrix()
    F.block_number = block_number
    F.row          = row
    F.column       = column
    F.value        = value
    return [F], bvals

def g2ab_trace(nmo, nalpha, nbeta, block_num) -> Tuple[List, List]:
    block_number=[block_num] * (nmo * nmo)
    value = [1.0] * (nmo * nmo)
    row = []
    column = []
    bvals = [nalpha * nbeta]
    for i, j in itertools.product(range(nmo), repeat=2):
        ij = i * nmo + j
        row.append(ij+1)
        column.append(ij+1)
    F = libsdp.sdp_matrix()
    F.block_number = block_number
    F.row          = row
    F.column       = column
    F.value        = value
    return [F], bvals


def g2ba_trace(nmo, nalpha, nbeta, block_num) -> Tuple[List, List]:
    return g2ab_trace(nmo, nalpha, nbeta, block_num)

def g2aabb_trace(nmo: int, nalpha: int, nbeta: int, block_num: int, alpha_beta: str) -> Tuple[List, List]:
    """
    Constrain the G^{aa}_{aa} and G^{bb}_{bb} sub-blocks separately
    
    The G2aabb is always arranged as

    | G^aa_aa | G^bb_aa |
    ----------------------
    | G^bb_aa | G^bb_bb |
    """
    block_number=[block_num] * (2 * nmo * nmo)
    value = [1.0] * (2 * nmo * nmo)
    row = []
    column = []
    bvals = []
    if alpha_beta == 'alpha':
        bvals = [nalpha * (nmo - nalpha + 1)]
        for i, j in itertools.product(range(nmo), repeat=2):
            ij = i * nmo + j
            row.append(ij+1)
            column.append(ij+1)
    elif alpha_beta == 'beta':
        bvals = [nbeta * (nmo - nbeta + 1)]
        shift = nmo * nmo
        for i, j in itertools.product(range(nmo), repeat=2):
            ij = i * nmo + j
            row.append(ij + 1 + shift)
            column.append(ij + 1 + shift)
    else:
        raise ValueError("alpha_beta must be alpha or beta. You provided {}".format(alpha_beta))

    F = libsdp.sdp_matrix()
    F.block_number = block_number
    F.row          = row
    F.column       = column
    F.value        = value
    return [F], bvals

def contract_g_to_d1a(nmo, nalpha, nbeta):
    # G -> D1a
    F = []
    bvals = []
    for i in range(nmo):
        for j in range(nmo):
            # set up row/col/data for constraint
            block_number=[]
            row=[]
            column=[]
            value=[]
            # loop over G-mat values
            gab_value_ab = 0.25 / (nmo - nbeta)
            gab_value_aa = 0.25 / (nmo - nalpha + 1)
            gab_value_aabb = 0.25 / nbeta
            for k in range(nmo):
                ik = i * nmo + k
                jk = j * nmo + k
                block_number.append(4)
                row.append(ik+1)
                column.append(jk+1)
                value.append(gab_value_ab)

                block_number.append(3)
                row.append(ik+1)
                column.append(jk+1)
                value.append(gab_value_aa)

                ij = i * nmo + j
                kk = k * nmo + k
                block_number.append(3)
                row.append(ij + 1)
                column.append(kk + 1)
                value.append(gab_value_aabb)

                ij = i * nmo + j
                kk = k * nmo + k
                block_number.append(3)
                row.append(kk + 1)
                column.append(ij + 1)
                value.append(gab_value_aabb)

            # subtract D1a value
            block_number.append(1)
            row.append(i+1)
            column.append(j+1)
            value.append(-1.0)

            # store row of A + bval 
            Fi = libsdp.sdp_matrix()
            Fi.block_number = block_number
            Fi.row          = row
            Fi.column       = column
            Fi.value        = value
            F.append(Fi)
            bvals.append(0.0)

    assert len(F) == len(bvals)
    return F, bvals

def contract_g_to_d1b(nmo, nalpha, nbeta):
    # G -> D1b
    F = []
    bvals = []
    shift = nmo * nmo
    for i in range(nmo):
        for j in range(nmo):
            # set up row/col/data for constraint
            block_number=[]
            row=[]
            column=[]
            value=[]

            # store G-matrix values
            g_value_ba = 0.25 / (nmo - nalpha)
            g_value_bb = 0.25 / (nmo - nbeta + 1)
            g_value_aabb = 0.25 / nalpha
            for k in range(nmo):
                ik = i * nmo + k
                jk = j * nmo + k
                block_number.append(5)
                row.append(ik+1)
                column.append(jk+1)
                value.append(g_value_ba)

                block_number.append(3)
                row.append(ik + 1 + shift)
                column.append(jk + 1 + shift)
                value.append(g_value_bb)

                ij = i * nmo + j
                kk = k * nmo + k
                block_number.append(3)
                row.append(ij + 1)
                column.append(kk + 1)
                value.append(g_value_aabb)

                ij = i * nmo + j
                kk = k * nmo + k
                block_number.append(3)
                row.append(kk + 1)
                column.append(ij + 1)
                value.append(g_value_aabb)

            # D1b component of the constraint
            block_number.append(2)
            row.append(i+1)
            column.append(j+1)
            value.append(-1.0)
    
            Fi = libsdp.sdp_matrix()
            Fi.block_number = block_number
            Fi.row          = row
            Fi.column       = column
            Fi.value        = value
            F.append(Fi)
            bvals.append(0.0)

    assert len(F) == len(bvals)
    return F, bvals


def build_hamiltonian(nmo, oei, tei):
    # F0 
    block_number=[]
    row=[]
    column=[]
    value=[]
    F = libsdp.sdp_matrix()
    # all 2 spin-blocks of the one-electron integrals
    for i in range(nmo):
        for j in range (0,nmo):
            block_number.append(1)
            row.append(i+1)
            column.append(j+1)
            value.append(oei[i][j])
    for i in range(nmo):
        for j in range (0,nmo):
            block_number.append(2)
            row.append(i+1)
            column.append(j+1)
            value.append(oei[i][j])

    # all 4 spin-blocks of the two-electron integrals
    shift = nmo * nmo
    tei_mat = tei.transpose(0, 1, 3, 2).reshape((nmo * nmo, nmo * nmo))
    for ij, kl in itertools.product(range(nmo * nmo), repeat=2):
        block_number.append(3)
        row.append(ij+1)
        column.append(kl+1)
        value.append(0.5 * tei_mat[ij, kl])
    for ij, kl in itertools.product(range(nmo * nmo), repeat=2):
        block_number.append(3)
        row.append(ij + 1 + shift)
        column.append(kl + 1 + shift)
        value.append(0.5 * tei_mat[ij, kl])
    for ij, kl in itertools.product(range(nmo * nmo), repeat=2):
        block_number.append(4)
        row.append(ij+1)
        column.append(kl+1)
        value.append(0.5 * tei_mat[ij, kl])
    for ij, kl in itertools.product(range(nmo * nmo), repeat=2):
        block_number.append(5)
        row.append(ij+1)
        column.append(kl+1)
        value.append(0.5 * tei_mat[ij, kl])
    F.block_number = block_number
    F.row          = row
    F.column       = column
    F.value        = value
    return F
 

def build_sdp(nalpha, nbeta, nmo, oei, tei):
    """ set up details of the SDP

    :param nalpha:       number of alpha electrons
    :param nbeta:        number of beta electrons
    :param nmo:          number of spatial molecular orbitals
    :param oei:          core Hamiltonian matrix
    :param tei:          two-electron repulsion integrals
    :return: b:          the constraint vector
    :return: F:          list of rows of constraint matrix in sparse format; 
                         note that F[0] is actually the vector defining the 
                         problem (contains the one- and two-electron integrals)
    :return: dimensions: list of dimensions of blocks of primal solution
    """

    # for a two-electron system, all we need are
    # D1a D1b G2aa,bb, G2ab, G2ba

    # block dimensions
    dimensions = []
    dimensions.append(nmo)     # D1a, block 1
    dimensions.append(nmo)     # D1a, block 2
    dimensions.append(2 * nmo * nmo) # G2aa,bb, block 3
    dimensions.append(nmo * nmo)  # G2ab, block 4
    dimensions.append(nmo * nmo)  # G2ba, block 5


    # number of blocks
    nblocks = len(dimensions)

    # Hamiltonian block
    F0 = build_hamiltonian(nmo, oei, tei)

    # constraints (F1, F2, ...)
    b = []
    F = [F0]

    # Tr(D1a)
    Fi, bvals = d1a_trace(nmo, nalpha, block_num=1)
    b.extend(bvals)
    F.extend(Fi)

    # Tr(D1b)
    Fi, bvals = d1b_trace(nmo, nalpha, block_num=2)
    b.extend(bvals)
    F.extend(Fi)

    # Tr(g2ab)
    Fi, bvals = g2ab_trace(nmo, nalpha, nbeta, block_num=4)
    b.extend(bvals)
    F.extend(Fi)

    # Tr(g2ba)
    Fi, bvals = g2ba_trace(nmo, nalpha, nbeta, block_num=5)
    b.extend(bvals)
    F.extend(Fi)

    # Tr(g2aa,bb)_alpha
    Fi, bvals = g2aabb_trace(nmo, nalpha, nbeta, block_num=3, alpha_beta='alpha')
    b.extend(bvals)
    F.extend(Fi)

    # Tr(g2aa,bb)_beta
    Fi, bvals = g2aabb_trace(nmo, nalpha, nbeta, block_num=3, alpha_beta='beta')
    b.extend(bvals)
    F.extend(Fi)

    # contraction condition D1a
    Fi, bvals = contract_g_to_d1a(nmo, nalpha, nbeta)
    b.extend(bvals)
    F.extend(Fi)

    # contraction condition D1b
    Fi, bvals = contract_g_to_d1b(nmo, nalpha, nbeta)
    b.extend(bvals)
    F.extend(Fi)

    assert len(F) - 1 == len(b)
    return b, F, dimensions

def main():

    # build molecule
    mol = pyscf.M(
        atom='H 0 0 0; H 0 0 1.0',
        basis='sto-3g',
        symmetry=False)

    # run RHF
    mf = mol.RHF().run()

    # get mo coefficient matrix
    C = mf.mo_coeff

    # get two-electron integrals
    tei = mol.intor('int2e')

    # transform two-electron integrals to mo basis
    tei = np.einsum('uj,vi,wl,xk,uvwx',C,C,C,C,tei)

    # get core hamiltonian
    kinetic   = mol.intor('int1e_kin')
    potential = mol.intor('int1e_nuc')
    oei       = kinetic + potential

    # transform core hamiltonian to mo basis
    oei = np.einsum('uj,vi,uv',C,C,oei)

    # number of occupied orbitals
    occ = mf.mo_occ
    nele = int(sum(occ))
    nalpha = nele // 2
    nbeta  = nalpha

    # number of spatial orbitals
    nmo = int(mf.mo_coeff.shape[1])

    # build inputs for the SDP
    # 
    # min   x.c
    # s.t.  Ax = b
    #       x >= 0
    # 
    # b is the right-hand side of Ax = b
    # F contains c followed by the rows of A, in SDPA sparse matrix format
    # 
    b, F, dimensions = build_sdp(nalpha,nbeta,nmo,oei,tei)

    # set options
    options = libsdp.sdp_options()

    maxiter = 5000000

    options.sdp_algorithm             = options.SDPAlgorithm.RRSDP
    options.maxiter                   = maxiter
    options.sdp_error_convergence     = 1e-8
    options.sdp_objective_convergence = 1e-8
    options.penalty_parameter_scaling = 0.1

    # solve sdp
    sdp = libsdp.sdp_solver(options)
    x = sdp.solve(b,F,dimensions,maxiter)

    # primal energy:
    objective = 0
    for i in range (0,len(F[0].block_number)):
        block  = F[0].block_number[i] - 1
        row    = F[0].row[i] - 1
        column = F[0].column[i] - 1
        value  = F[0].value[i]
        
        off = 0
        for j in range (0,block):
            off += dimensions[j] * dimensions[j]

        objective += x[off + row * dimensions[block] + column] * value

    print('')
    print('    * v2RDM electronic energy: %20.12f' % (objective))
    print('    * v2RDM total energy:      %20.12f' % (objective + mf.energy_nuc()))
    print('')

if __name__ == "__main__":
    main()



"""
Driver for variational two-electron reduced-density matrix method using only the G-matrix
but using the full metric
"""
from typing import List, Tuple
import itertools
import numpy as np
from numpy import einsum

import sys
sys.path.insert(0, '../../.')

import libsdp

import pyscf

from openfermion.chem.molecular_data import spinorb_from_spatial

def d1_trace(nso: int, nelec: int, block_num: int) -> Tuple[List, List]:
    # Tr(D1a)
    block_number = [block_num] * nso 
    row = list(range(1, nso + 1))
    column = list(range(1, nso + 1))
    value = [1] * nso
    bvals = [nelec]
    F = libsdp.sdp_matrix()
    F.block_number = block_number
    F.row          = row
    F.column       = column
    F.value        = value
    return [F], bvals

def g2_phph_trace(nso: int, nelec: int, block_num=2):
    block_number = [block_num] * (nso * nso)
    value = [1.0] * (nso * nso)
    bvals = [nelec * (nso - nelec + 1)]
    row = []
    column = []
    for i, j in itertools.product(range(nso), repeat=2):
        ij = i * nso + j
        row.append(ij+1)
        column.append(ij+1)
    F = libsdp.sdp_matrix()
    F.block_number = block_number
    F.row          = row
    F.column       = column
    F.value        = value
    return [F], bvals

def g2_hphp_trace(nso: int, nelec: int, block_num=2):
    block_number = [block_num] * (nso * nso)
    value = [1.0] * (nso * nso)
    bvals = [(nso - nelec) * (nelec + 1)]
    row = []
    column = []
    shift = nso * nso
    for i, j in itertools.product(range(nso), repeat=2):
        ij = i * nso + j
        row.append(ij + 1 + shift)
        column.append(ij + 1 + shift)
    F = libsdp.sdp_matrix()
    F.block_number = block_number
    F.row          = row
    F.column       = column
    F.value        = value
    return [F], bvals

def g2_phph_contract_to_d1(nso: int, nelec, block_num_d1, block_num_g):
    F = []
    bvals = []
    for p in range(nso):
        for s in range(nso):
            # set up row/col/data for constraint
            block_number=[]
            row=[]
            column=[]
            value=[]
            # loop over G-mat values
            for r in range(nso):
                pr = p * nso + r
                sr = s * nso + r
                block_number.append(block_num_g)
                row.append(pr+1)
                column.append(sr+1)
                value.append(1.)

            # subtract D1 value
            block_number.append(block_num_d1)
            row.append(p+1)
            column.append(s+1)
            value.append(-1.0 * (nso - nelec + 1))

            # store row of A + bval 
            Fi = libsdp.sdp_matrix()
            Fi.block_number = block_number
            Fi.row          = row
            Fi.column       = column
            Fi.value        = value
            F.append(Fi)
            bvals.append(0.0)

    return F, bvals

def g2_internal_constraints(nso: int, block_num_d1: int, block_num_g2: int):
    """
    p^ q r^ s = delta_{rs}p^ q - p^q s r^
    p^ q r^ s = delta_{pq}r^ s - q p^ r^ s
    p^ q r^ s = -delta_{pq}delta_{rs} + delta_{pq}r^ s + delta_{rs} p^ q + q p^ s r^
    """
    F = []
    bvals = []
    shift = nso * nso
    kdelta = np.eye(nso)
    for p, q, r, s in itertools.product(range(nso), repeat=4):
        block_number=[]
        row=[]
        column=[]
        value=[]
        pq  = p * nso + q
        sr = s * nso + r

        # G^{pq}_{sr}
        block_number.append(block_num_g2)
        row.append(pq + 1)
        column.append(sr + 1)
        value.append(-2.)

        # delta_rs p^ q
        block_number.append(block_num_d1)
        row.append(p + 1)
        column.append(q + 1)
        value.append(kdelta[r, s])
        # -p^ q s r^
        rs = r * nso + s
        block_number.append(block_num_g2)
        row.append(pq + 1)
        column.append(rs + 1 + shift)
        value.append(-1.)

        # delta_pq r^ s
        block_number.append(block_num_d1)
        row.append(r + 1)
        column.append(s + 1)
        value.append(kdelta[p, q])
        # -q p^ r^ s
        qp = q * nso + p
        block_number.append(block_num_g2)
        row.append(qp + 1 + shift)
        column.append(sr + 1)
        value.append(-1.)

        # store row of A + bval 
        Fi = libsdp.sdp_matrix()
        Fi.block_number = block_number
        Fi.row          = row
        Fi.column       = column
        Fi.value        = value
        F.append(Fi)
        bvals.append(0.)

    ######################################
    #
    # THIS CONSTRAINT IS CAUSING PROBLEMS
    #
    #######################################
    # for p, q, r, s in itertools.product(range(nso), repeat=4):
    #     block_number=[]
    #     row=[]
    #     column=[]
    #     value=[]
    #     pq = p * nso + q
    #     sr = s * nso + r

    #     # G^{pq}_{sr}
    #     block_number.append(block_num_g2)
    #     row.append(pq + 1)
    #     column.append(sr + 1)
    #     value.append(-1.)

    #     rs = r * nso + s
    #     qp = q * nso + p
    #     block_number.append(block_num_d1)
    #     row.append(p + 1)
    #     column.append(q + 1)
    #     value.append(kdelta[r, s])
    #     block_number.append(block_num_d1)
    #     row.append(r + 1)
    #     column.append(s + 1)
    #     value.append(kdelta[p, q])
    #     block_number.append(block_num_g2)
    #     row.append(qp + 1 + shift)
    #     column.append(rs + 1 + shift)
    #     value.append(1.)

    #     # store row of A + bval 
    #     Fi = libsdp.sdp_matrix()
    #     Fi.block_number = block_number
    #     Fi.row          = row
    #     Fi.column       = column
    #     Fi.value        = value
    #     F.append(Fi)
    #     bvals.append(-kdelta[p, q] * kdelta[r, s])

    return F, bvals


def antisymm(nso, block_num_d1, block_num_g2aabb):
    F = []
    bvals = []
    kdelta = np.eye(nso)
    nmo = nso // 2
    for i, l, j, k in itertools.product(range(nso), repeat=4):
        block_number=[]
        row=[]
        column=[]
        value=[]
        il = (2 * i) * nso + (2 * l)
        kj = (2 * k) * nso + (2 * j)
        jl = (2 * j) * nso + (2 * l)
        ki = (2 * k) * nso + (2 * i)

        block_number.append(block_num_g2aabb)
        row.append(jl + 1)
        column.append(ki + 1)
        value.append(1.)
        block_number.append(block_num_g2aabb)
        row.append(il + 1)
        column.append(kj + 1)
        value.append(1.)
        block_number.append(block_num_d1)
        row.append(i + 1)
        column.append(k + 1)
        value.append(-kdelta[j,l])
        block_number.append(block_num_d1)
        row.append(j + 1)
        column.append(k + 1)
        value.append(-kdelta[i,l])

        Fi = libsdp.sdp_matrix()
        Fi.block_number = block_number
        Fi.row          = row
        Fi.column       = column
        Fi.value        = value
        F.append(Fi)
        bvals.append(0.0)

        block_number=[]
        row=[]
        column=[]
        value=[]
        il = (2 * i + 1) * nso + (2 * l + 1)
        kj = (2 * k + 1) * nso + (2 * j + 1)
        jl = (2 * j + 1) * nso + (2 * l + 1)
        ki = (2 * k + 1) * nso + (2 * i + 1)

        block_number.append(block_num_g2aabb)
        row.append(jl + 1)
        column.append(ki + 1)
        value.append(1.)
        block_number.append(block_num_g2aabb)
        row.append(il + 1)
        column.append(kj + 1)
        value.append(1.)
        block_number.append(block_num_d1)
        row.append(i + 1)
        column.append(k + 1)
        value.append(-kdelta[j,l])
        block_number.append(block_num_d1)
        row.append(j + 1)
        column.append(k + 1)
        value.append(-kdelta[i,l])

        Fi = libsdp.sdp_matrix()
        Fi.block_number = block_number
        Fi.row          = row
        Fi.column       = column
        Fi.value        = value
        F.append(Fi)
        bvals.append(0.0)

    return F, bvals

def build_hamiltonian(nmo, oei, tei):
    of_eris = tei.transpose(0, 2, 3, 1)
    one_body_coefficients, two_body_coefficients = spinorb_from_spatial(oei, of_eris)
    nso = 2 * nmo
    # two_body_coefficients are <1'2'|21>
    # F0 
    block_number=[]
    row=[]
    column=[]
    value=[]
    F = libsdp.sdp_matrix()
    # all 2 spin-blocks of the one-electron integrals in spin-orbital representation
    for i in range(nso):
        for j in range(nso):
            block_number.append(1)
            row.append(i + 1)
            column.append(j + 1)
            value.append(one_body_coefficients[i, j])

    for i, j, k, l in itertools.product(range(nso), repeat=4):
        block_number.append(2)
        ij = i * (nso) + j
        kl = k * (nso) + l
        row.append(ij + 1)
        column.append(kl + 1)
        value.append(0.5 * two_body_coefficients[i, l, k, j])
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
    nso = 2 * nmo
    dimensions = []
    dimensions.append(nso)     # D1, block 1
    dimensions.append(2 * nso * nso)     # G, block 2


    # number of blocks
    nblocks = len(dimensions)

    # Hamiltonian block
    F0 = build_hamiltonian(nmo, oei, tei)

    # constraints (F1, F2, ...)
    b = []
    F = [F0]

    # Tr(D1a)
    Fi, bvals = d1_trace(nso, nalpha + nbeta, block_num=1)
    b.extend(bvals)
    F.extend(Fi)

    # Tr(g2-phph)
    Fi, bvals = g2_phph_trace(nso, nalpha + nbeta, block_num=2)
    b.extend(bvals)
    F.extend(Fi)

    # Tr(g2-hphp)
    Fi, bvals = g2_hphp_trace(nso, nalpha + nbeta, block_num=2)
    b.extend(bvals)
    F.extend(Fi)

    Fi, bvals = g2_phph_contract_to_d1(nso, nalpha + nbeta, 1, 2)
    b.extend(bvals)
    F.extend(Fi)

    Fi, bvals = g2_internal_constraints(nso, 1, 2)
    b.extend(bvals)
    F.extend(Fi)

    Fi, bvals = antisymm(nso, 1, 2)

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
    oei = np.einsum('uj,vi,uv',C,C,oei) - 0.5 * np.einsum('irrj->ij', tei)

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

    maxiter = 2000

    options.sdp_algorithm             = options.SDPAlgorithm.BPSDP
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



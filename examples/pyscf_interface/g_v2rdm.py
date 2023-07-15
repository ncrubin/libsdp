"""
Driver for variational two-electron reduced-density matrix method using only the G-matrix
"""
from typing import List, Tuple
import itertools
from itertools import product
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
    bvals = [nalpha * (nmo - nbeta)]
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
    block_number=[block_num] * (nmo * nmo)
    value = [1.0] * (nmo * nmo)
    row = []
    column = []
    bvals = [nbeta * (nmo - nalpha)]
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

def g2aabb_trace(nmo: int, block_num: int, row_offset: int, col_offset: int, trace_val: float) -> Tuple[List, List]:
    """
    Constrain the G^{aa}_{aa} and G^{bb}_{bb} sub-blocks separately
    
    The G2aabb is always arranged as

    | G^aa_aa | G^bb_aa |
    ----------------------
    | G^bb_aa | G^bb_bb |
    """
    block_number=[block_num] * (nmo * nmo)
    value = [1.0] * (nmo * nmo)
    row = []
    column = []
    bvals = []
    bvals = [trace_val]
    for i, j in itertools.product(range(nmo), repeat=2):
        ij = i * nmo + j
        row.append(ij + 1 + row_offset)
        column.append(ij + 1 + col_offset)
    F = libsdp.sdp_matrix()
    F.block_number = block_number
    F.row          = row
    F.column       = column
    F.value        = value
    return [F], bvals

def gaaaa_to_d1a(nmo, offset, nalpha, block_num_d1a, block_num_g2aabb):
    F = []
    bvals = []
    shift = nmo * nmo
    kdelta = np.eye(nmo)

    # g2aaaa(ij, kj) = (nmo - nalpha + 1) d1a(i, k)
    for i, k in product(range(nmo), repeat=2):
        block_number = []
        row = []
        column = []
        value = []
        for j in range(nmo):
            ij = i * nmo + j
            kj = k * nmo + j
            block_number.append(block_num_g2aabb)
            row.append(ij + 1 + offset)
            column.append(kj + 1 + offset)
            value.append(1.)

        block_number.append(block_num_d1a)
        row.append(i + 1)
        column.append(k + 1)
        value.append(-(nmo - nalpha + 1))
        Fi = libsdp.sdp_matrix()
        Fi.block_number = block_number
        Fi.row          = row
        Fi.column       = column
        Fi.value        = value
        F.append(Fi)
        bvals.append(0.0)

    # g2aaaa(ij, il) = nalpha djl - (nalpha - 1) d1a(l, j)
    for j, l in product(range(nmo), repeat=2):
        block_number = []
        row = []
        column = []
        value = []
        for i in range(nmo):
            ij = i * nmo + j
            il = i * nmo + l
            block_number.append(block_num_g2aabb)
            row.append(ij + 1 + offset)
            column.append(il + 1 + offset)
            value.append(1.)

        block_number.append(block_num_d1a)
        row.append(l + 1)
        column.append(j + 1)
        value.append(nalpha - 1)
        Fi = libsdp.sdp_matrix()
        Fi.block_number = block_number
        Fi.row          = row
        Fi.column       = column
        Fi.value        = value
        F.append(Fi)
        bvals.append(nalpha * kdelta[j, l])
    
    # g2aaaa(ij, kk) = nalpha d1a(i, j)
    for i, j in product(range(nmo), repeat=2):
        block_number = []
        row = []
        column = []
        value = []
        for k in range(nmo):
            ij = i * nmo + j
            kk = k * nmo + k
            block_number.append(block_num_g2aabb)
            row.append(ij + 1 + offset)
            column.append(kk + 1 + offset)
            value.append(1.)

        block_number.append(block_num_d1a)
        row.append(i + 1)
        column.append(j + 1)
        value.append(-nalpha)
        Fi = libsdp.sdp_matrix()
        Fi.block_number = block_number
        Fi.row          = row
        Fi.column       = column
        Fi.value        = value
        F.append(Fi)
        bvals.append(0)

    # g2aaaa(ii, kl) = nalpha d1a(l, k)
    for k, l in product(range(nmo), repeat=2):
        block_number = []
        row = []
        column = []
        value = []
        for i in range(nmo):
            ii = i * nmo + i
            kl = k * nmo + l
            block_number.append(block_num_g2aabb)
            row.append(ii + 1 + offset) 
            column.append(kl + 1 + offset)
            value.append(1.)

        block_number.append(block_num_d1a)
        row.append(l + 1)
        column.append(k + 1)
        value.append(-nalpha)
        Fi = libsdp.sdp_matrix()
        Fi.block_number = block_number
        Fi.row          = row
        Fi.column       = column
        Fi.value        = value
        F.append(Fi)
        bvals.append(0)

    return F, bvals


def gabab_to_d1(nmo, n1, block_num_d1_1, n2, block_num_d1_2, block_num_g2ab):
    F = []
    bvals = []
    shift = nmo * nmo
    kdelta = np.eye(nmo)

    # g2ab(ij, kj) = (nmo - nbeta) d1a(i, k)
    for i, k in product(range(nmo), repeat=2):
        block_number = []
        row = []
        column = []
        value = []
        for j in range(nmo):
            ij = i * nmo + j
            kj = k * nmo + j
            block_number.append(block_num_g2ab)
            row.append(ij + 1)
            column.append(kj + 1)
            value.append(1.)

        block_number.append(block_num_d1_1)
        row.append(i + 1)
        column.append(k + 1)
        value.append(-(nmo - n1))
        Fi = libsdp.sdp_matrix()
        Fi.block_number = block_number
        Fi.row          = row
        Fi.column       = column
        Fi.value        = value
        F.append(Fi)
        bvals.append(0)

    # g2ab(ij, il) = nalpha djl - nalpha d1b(l, j)
    for j, l in product(range(nmo), repeat=2):
        block_number = []
        row = []
        column = []
        value = []
        for i in range(nmo):
            ij = i * nmo + j
            il = i * nmo + l
            block_number.append(block_num_g2ab)
            row.append(ij + 1)
            column.append(il + 1)
            value.append(1.)

        block_number.append(block_num_d1_2)
        row.append(l + 1)
        column.append(j + 1)
        value.append(n2)
        Fi = libsdp.sdp_matrix()
        Fi.block_number = block_number
        Fi.row          = row
        Fi.column       = column
        Fi.value        = value
        F.append(Fi)
        bvals.append(n2 * kdelta[j, l])
    
    return F, bvals


def g2aabb_d1(nmo, row_offset, col_offset, n1, block_num_d1_1, n2, block_num_d1_2, block_num_g2aabb):
    F = []
    bvals = []
    shift = nmo * nmo
    kdelta = np.eye(nmo)

    # g2aabb(ij, kk) = nbeta d1a(i, j)
    for i, j in product(range(nmo), repeat=2):
        block_number = []
        row = []
        column = []
        value = []
        for k in range(nmo):
            ij = i * nmo + j
            kk = k * nmo + k
            block_number.append(block_num_g2aabb)
            row.append(ij + 1 + row_offset) 
            column.append(kk + 1 + col_offset)
            value.append(1.)

        block_number.append(block_num_d1_1)
        row.append(i + 1)
        column.append(j + 1)
        value.append(-n1)
        Fi = libsdp.sdp_matrix()
        Fi.block_number = block_number
        Fi.row          = row
        Fi.column       = column
        Fi.value        = value
        F.append(Fi)
        bvals.append(0)

    # g2aabb(ii, kl) = nalpha d1b(l, k)
    for k, l in product(range(nmo), repeat=2):
        block_number = []
        row = []
        column = []
        value = []
        for i in range(nmo):
            ii = i * nmo + i
            kl = k * nmo + l
            block_number.append(block_num_g2aabb)
            row.append(ii + 1 + row_offset) 
            column.append(kl + 1 + col_offset)
            value.append(1.)

        block_number.append(block_num_d1_2)
        row.append(l + 1)
        column.append(k + 1)
        value.append(-n2)
        Fi = libsdp.sdp_matrix()
        Fi.block_number = block_number
        Fi.row          = row
        Fi.column       = column
        Fi.value        = value
        F.append(Fi)
        bvals.append(0)
    return F, bvals


# def antisymm_bbbb(nmo, block_num_d1, block_num_g2aabb):
#     F = []
#     bvals = []
#     kdelta = np.eye(nmo)
#     shift = nmo * nmo
#     for p, q, s, r in product(range(nmo), repeat=4):
#         block_number=[]
#         row=[]
#         column=[]
#         value=[]
#         ps = p * nmo + s
#         rq = r * nmo + q
# 
#         qs = q * nmo + s
#         rp = r * nmo + p
# 
#         pr = p * nmo + r
#         sq = s * nmo + q
# 
#         qs = q * nmo + s
#         rp = r * nmo + p
# 
#         # p^ s q^ r
#         block_number.append(block_num_g2aabb)
#         row.append(ps + 1 + shift)
#         column.append(rq + 1 + shift)
#         value.append(-1.)
#         # q^ s p^ r
#         block_number.append(block_num_g2aabb)
#         row.append(qs + 1 + shift)
#         column.append(rp + 1 + shift)
#         value.append(-1.)
#         # p^ r q^ s
#         block_number.append(block_num_g2aabb)
#         row.append(pr + 1 + shift)
#         column.append(sq + 1 + shift)
#         value.append(-1.)
#         # q^ s p^ r
#         block_number.append(block_num_g2aabb)
#         row.append(qs + 1 + shift)
#         column.append(rp + 1 + shift)
#         value.append(-1.)
# 
#         # p^ r delta_{qs}
#         block_number.append(block_num_d1)
#         row.append(p + 1)
#         column.append(r + 1)
#         value.append(kdelta[q, s])
#         # q^ r delta_ps
#         block_number.append(block_num_d1)
#         row.append(q + 1)
#         column.append(r + 1)
#         value.append(kdelta[p, s])
#         # p^ s delta_rq
#         block_number.append(block_num_d1)
#         row.append(p + 1)
#         column.append(s + 1)
#         value.append(kdelta[r, q])
#         # q^ r delta_ps
#         block_number.append(block_num_d1)
#         row.append(q + 1)
#         column.append(r + 1)
#         value.append(kdelta[p, s])
#         Fi = libsdp.sdp_matrix()
#         Fi.block_number = block_number
#         Fi.row          = row
#         Fi.column       = column
#         Fi.value        = value
#         F.append(Fi)
#         bvals.append(0.0)
# 
#     return F, bvals

def g2aaaa_antisymmetry(nmo, offset, d1_block_id, g2aabb_block_num):
    """
    g2aaaa subblock should satisfy antisymmetry of d2aa

    :param offset:      the offset of the row and column geminal labels
    :param d1_block_id: the relevant block of d1 (a or b)
    """
    F = []
    b = []
    delta = np.eye(nmo)

    # g2aaaa(ik, jl) + g2aaaa(ij, kl) - d1a(i, k) djl - d1a(i, j) dlk
    for i, j in product(range(nmo), repeat=2):
        ij = i * nmo + j
        for k, l in product(range(nmo), repeat=2):
            kl = k * nmo + l
            block_number=[]
            row=[]
            column=[]
            value=[]

            ik = i * nmo + k
            jl = j * nmo + l

            block_number.append(g2aabb_block_num)
            row.append(ij + offset + 1)
            column.append(kl + offset + 1)
            value.append(1.0)

            block_number.append(g2aabb_block_num)
            row.append(ik + offset + 1)
            column.append(jl + offset + 1)
            value.append(1.0)

            block_number.append(d1_block_id)
            row.append(i + 1)
            column.append(k + 1)
            value.append(-delta[j, l])

            block_number.append(d1_block_id)
            row.append(i + 1)
            column.append(j + 1)
            value.append(-delta[l, k])

            Fi = libsdp.sdp_matrix()
            Fi.block_number = block_number
            Fi.row          = row
            Fi.column       = column
            Fi.value        = value
            F.append(Fi)
            b.append(0.0)

    # g2aaaa(lj, ki) + g2aaaa(lk, ji) - d1a(l, k) dji - d1a(l, j) dki
    for l, k in product(range(nmo), repeat=2):
        lk = l * nmo + k
        for j, i in product(range(nmo), repeat=2):
            ji = j * nmo + i
            lj = l * nmo + j
            ki = k * nmo + i

            block_number=[]
            row=[]
            column=[]
            value=[]

            block_number.append(g2aabb_block_num)
            row.append(lj + offset + 1)
            column.append(ki + offset + 1)
            value.append(1.0)

            block_number.append(g2aabb_block_num)
            row.append(lk + offset + 1)
            column.append(ji + offset + 1)
            value.append(1.0)

            block_number.append(d1_block_id)
            row.append(l + 1)
            column.append(k + 1)
            value.append(-delta[j, i])

            block_number.append(d1_block_id)
            row.append(l + 1)
            column.append(j + 1)
            value.append(-delta[k, i])

            Fi = libsdp.sdp_matrix()
            Fi.block_number = block_number
            Fi.row          = row
            Fi.column       = column
            Fi.value        = value

            F.append(Fi)
            b.append(0.0)

    # g2aaaa(ij, kl) - g2aaaa(lk, ji) - d1a(i, k) djl + d1a(l, j) dki = 0
    for i, j in product(range(nmo), repeat=2):
        ij = i * nmo + j
        for k, l in product(range(nmo), repeat=2):
            kl = k * nmo + l

            block_number=[]
            row=[]
            column=[]
            value=[]

            lk = l * nmo + k
            ji = j * nmo + i

            block_number.append(g2aabb_block_num)
            row.append(ij + offset + 1)
            column.append(kl + offset + 1)
            value.append(1.0)

            block_number.append(g2aabb_block_num)
            row.append(lk + offset + 1)
            column.append(ji + offset + 1)
            value.append(-1.0)

            block_number.append(d1_block_id)
            row.append(i + 1)
            column.append(k + 1)
            value.append(-delta[j, l])

            block_number.append(d1_block_id)
            row.append(l + 1)
            column.append(j + 1)
            value.append(delta[k, i])

            Fi = libsdp.sdp_matrix()
            Fi.block_number = block_number
            Fi.row          = row
            Fi.column       = column
            Fi.value        = value

            F.append(Fi)
            b.append(0.0)

    # g2aaaa(ik, jl) - g2aaaa(lj, ki) - d1a(i, j) dkl + d1a(l, k) dji
    for i, k in product(range(nmo), repeat=2):
        ik = i * nmo + k
        for j, l in product(range(nmo), repeat=2):
            jl = j * nmo + l

            block_number=[]
            row=[]
            column=[]
            value=[]

            lj = l * nmo + j
            ki = k * nmo + i

            block_number.append(g2aabb_block_num)
            row.append(lj + offset + 1)
            column.append(ki + offset + 1)
            value.append(-1.0)

            block_number.append(g2aabb_block_num)
            row.append(ik + offset + 1)
            column.append(jl + offset + 1)
            value.append(1.0)

            block_number.append(d1_block_id)
            row.append(i + 1)
            column.append(j + 1)
            value.append(-delta[k, l])

            block_number.append(d1_block_id)
            row.append(l + 1)
            column.append(k + 1)
            value.append(delta[j, i])

            Fi = libsdp.sdp_matrix()
            Fi.block_number = block_number
            Fi.row          = row
            Fi.column       = column
            Fi.value        = value

            F.append(Fi)
            b.append(0.0)
    return F, b

# def antisymm_aaaa(nmo, offset, block_num_d1, block_num_g2aabb):
#     F = []
#     bvals = []
#     kdelta = np.eye(nmo)
#     for i, l, j, k in itertools.product(range(nmo), repeat=4):
#         block_number=[]
#         row=[]
#         column=[]
#         value=[]
#         il = i * nmo + l
#         kj = k * nmo + j
#         jl = j * nmo + l
#         ki = k * nmo + i
# 
#         block_number.append(block_num_g2aabb)
#         row.append(jl + 1 + offset)
#         column.append(ki + 1 + offset)
#         value.append(1.)
#         block_number.append(block_num_g2aabb)
#         row.append(il + 1 + offset)
#         column.append(kj + 1 + offset)
#         value.append(1.)
#         block_number.append(block_num_d1)
#         row.append(i + 1)
#         column.append(k + 1)
#         value.append(-kdelta[j,l])
#         block_number.append(block_num_d1)
#         row.append(j + 1)
#         column.append(k + 1)
#         value.append(-kdelta[i,l])
# 
#         Fi = libsdp.sdp_matrix()
#         Fi.block_number = block_number
#         Fi.row          = row
#         Fi.column       = column
#         Fi.value        = value
#         F.append(Fi)
#         bvals.append(0.0)
# 
#     return F, bvals

# def antisymm_aaaa(nmo, offset, block_num_d1, block_num_g2aabb):
#     F = []
#     bvals = []
#     kdelta = np.eye(nmo)
#     for i, j in product(range(nmo), repeat=2):
#         ij = i * nmo + j
#         for k, l in product(range(nmo), repeat=2):
#             kl = k * nmo + l
#             block_number = []
#             row = []
#             column = []
#             value = []
#             ik = i * nmo + k
#             jl = j * nmo + l
#             block_number.append(block_num_g2aabb)
#             row.append(ij + offset + 1)
#             column.append(kl + offset + 1)
#             value.append(1.0)
# 
#             block_number.append(block_num_g2aabb)
#             row.append(ik + offset + 1)
#             column.append(jl + offset + 1)
#             value.append(1.0)
# 
#             block_number.append(block_num_d1)
#             row.append(i + 1)
#             column.append(k + 1)
#             value.append(-kdelta[j, l])
#    
#             block_number.append(block_num_d1)
#             row.append(i + 1)
#             column.append(j + 1)
#             value.append(-kdelta[l, k])
#    
#             Fi = libsdp.sdp_matrix()
#             Fi.block_number = block_number
#             Fi.row          = row
#             Fi.column       = column
#             Fi.value        = value
# 
#             F.append(Fi)
#             bvals.append(0.0)
# 
#     return F, bvals

def g2ab_2_ba(nmo, ab_block, ba_block):
    F = []
    bvals = []
    for i, j in product(range(nmo), repeat=2):
        ij = i * nmo + j
        ji = j * nmo + i
        for k, l in product(range(nmo), repeat=2):
            kl = k * nmo + l
            lk = l * nmo + k
            block_number = []
            row = []
            column = []
            value = []

            block_number.append(ab_block)
            row.append(ij + 1)
            column.append(kl + 1)
            value.append(1.)
            block_number.append(ba_block)
            row.append(lk + 1)
            column.append(ji + 1)
            value.append(-1.)

            Fi = libsdp.sdp_matrix()
            Fi.block_number = block_number
            Fi.row          = row
            Fi.column       = column
            Fi.value        = value

            F.append(Fi)
            bvals.append(0.0)
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
    # aaaa-block
    for ij, kl in itertools.product(range(nmo * nmo), repeat=2):
        block_number.append(3)
        row.append(ij + 1)
        column.append(kl + 1)
        value.append(0.5 * tei_mat[ij, kl])
    # bbbb-block
    for ij, kl in itertools.product(range(nmo * nmo), repeat=2):
        block_number.append(3)
        row.append(ij + 1 + shift)
        column.append(kl + 1 + shift)
        value.append(0.5 * tei_mat[ij, kl])
    # bbaa-block
    for ij, kl in itertools.product(range(nmo * nmo), repeat=2):
        block_number.append(3)
        row.append(ij + 1 + shift)
        column.append(kl + 1)
        value.append(0.5 * tei_mat[ij, kl])
    # aabb-block
    for ij, kl in itertools.product(range(nmo * nmo), repeat=2):
        block_number.append(3)
        row.append(ij + 1)
        column.append(kl + 1 + shift)
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

    ###################
    # 
    # TRACE CONDITIONS
    #
    ####################
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
    Fi, bvals = g2aabb_trace(nmo, block_num=3, row_offset=0, col_offset=0, trace_val=nalpha * (nmo - nalpha + 1))
    b.extend(bvals)
    F.extend(Fi)

    # Tr(g2aa,bb)_beta
    Fi, bvals = g2aabb_trace(nmo, block_num=3, row_offset=nmo*nmo, col_offset=nmo*nmo, trace_val=nbeta * (nmo - nbeta + 1))
    b.extend(bvals)
    F.extend(Fi)

    Fi, bvals = g2aabb_trace(nmo, block_num=3, row_offset=0, col_offset=nmo*nmo, trace_val=nbeta * nalpha)
    b.extend(bvals)
    F.extend(Fi)

    Fi, bvals = g2aabb_trace(nmo, block_num=3, row_offset=nmo*nmo, col_offset=0, trace_val=nbeta * nalpha)
    b.extend(bvals)
    F.extend(Fi)

    # ##########################
    # # 
    # # Contraction Conditions
    # #
    # ##########################
    # Fi, bvals = gaaaa_to_d1a(nmo, 0, nalpha, 1, 3)
    # b.extend(bvals)
    # F.extend(Fi)

    # Fi, bvals = gaaaa_to_d1a(nmo, nmo * nmo, nbeta, 2, 3)
    # b.extend(bvals)
    # F.extend(Fi)

    # Fi, bvals = g2aabb_d1(nmo, 0, nmo * nmo, nbeta, 1, nalpha, 2, 3)
    # b.extend(bvals)
    # F.extend(Fi)

    # Fi, bvals = g2aabb_d1(nmo, nmo * nmo, 0,  nalpha, 2, nbeta, 1, 3)
    # b.extend(bvals)
    # F.extend(Fi)

    # Fi, bvals = gabab_to_d1(nmo, nbeta, 1, nalpha, 2, 4)
    # b.extend(bvals)
    # F.extend(Fi)

    # Fi, bvals = gabab_to_d1(nmo, nalpha, 2, nbeta, 1, 5)
    # b.extend(bvals)
    # F.extend(Fi)

    ######################
    #
    # Antisymm conditions
    #
    #######################
    Fi, bvals = g2aaaa_antisymmetry(nmo, 0, 1, 3)
    b.extend(bvals)
    F.extend(Fi)
    Fi, bvals = g2aaaa_antisymmetry(nmo, nmo * nmo, 1, 3)
    b.extend(bvals)
    F.extend(Fi)



    # Fi, bvals = antisymm_aaaa(nmo, nmo * nmo, 2, 3)
    # b.extend(bvals)
    # F.extend(Fi)

    ##########################
    #
    # G2ab <-> G2ba
    #
    ####################
    # Fi, bvals = g2ab_2_ba(nmo, 4, 5)

    assert len(F) - 1 == len(b)
    return b, F, dimensions

def main():

    # build molecule
    mol = pyscf.M(
        atom='B 0 0 0; H 0 0 1.0',
        basis='sto-3g',
        symmetry=False)

    # run RHF
    mf = mol.RHF().run()
    myci = pyscf.fci.FCI(mf) 
    roots, wfs = myci.kernel(nroots=2)
    print("FCI energy :", roots[0])

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

    maxiter = 6000

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



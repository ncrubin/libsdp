"""
Driver for variational two-electron reduced-density matrix method. Integrals come from PySCF
"""
from itertools import product

import numpy as np
from numpy import einsum

import sys
sys.path.insert(0, '../../.')

import libsdp
from v2rdm_sdp import v2rdm_sdp
from g2_v2rdm_sdp import g2_v2rdm_sdp

import pyscf

import openfermion as of

def get_random_state_rdms(nmo, nalpha, nbeta):
    import fqe
    fqe_wf = fqe.Wavefunction([[nalpha + nbeta, nalpha - nbeta, nmo]])
    fqe_wf.set_wfn(strategy='random')
    fqe_data = fqe_wf.sector((nalpha + nbeta, nalpha - nbeta))
    opdm, tpdm = fqe_data.get_openfermion_rdms()
    phdm = np.einsum('il,jk', opdm, np.eye(2 * nmo)) - np.einsum('ikjl->ijkl', tpdm)
    rdms = {}
    rdms['aaaa'] = phdm[::2, ::2, ::2, ::2]
    rdms['bbbb'] = phdm[1::2, 1::2, 1::2, 1::2]
    rdms['aabb'] = phdm[::2, ::2, 1::2, 1::2]
    rdms['bbaa'] = phdm[1::2, 1::2, ::2, ::2]
    rdms['ab'] = phdm[::2, 1::2, 1::2, ::2]
    rdms['ba'] = phdm[1::2, ::2, ::2, 1::2]
    rdms['opdm_a'] = opdm[::2, ::2]
    rdms['opdm_b'] = opdm[1::2, 1::2]
    return rdms

def main():

    # build molecule
    # mol = pyscf.M(
    #     atom='H 0 0 0; He 0 0 1.0',
    #     basis='sto-3g',
    #     charge=1,
    #     symmetry=False)
    mol = pyscf.M(
        atom='H 0 0 0; Li 0 0 1.0',
        basis='sto-3g',
        charge=2,
        symmetry=False)
    # mol = pyscf.M(
    #     atom='H 0 0 0; He 0 0 1.0; H 0 1. 0; He 0 1. 1.',
    #     basis='sto-3g',
    #     charge=2,
    #     symmetry=False)



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

    # use this one for d-only
    #my_sdp = v2rdm_sdp(nalpha, nbeta, nmo, oei, tei, q2 = False, g2 = False)

    # use this one for g-only 
    my_sdp = g2_v2rdm_sdp(nalpha, nbeta, nmo, oei, tei, d2 = False, q2 = False, constrain_spin = True)

    b = my_sdp.b
    F = my_sdp.F
    dimensions = my_sdp.dimensions

    # set options
    options = libsdp.sdp_options()

    maxiter = 100000

    options.sdp_algorithm             = options.SDPAlgorithm.BPSDP
    options.maxiter                   = maxiter
    options.sdp_error_convergence     = 1e-5
    options.sdp_objective_convergence = 1e-5
    options.penalty_parameter_scaling = 0.1

    # solve sdp
    sdp_solver = libsdp.sdp_solver(options)
    x = sdp_solver.solve(b, F, dimensions, maxiter)

    # now that the sdp is solved, we can play around with the primal and dual solutions
    z = np.array(sdp_solver.get_z())
    c = np.array(sdp_solver.get_c())
    y = np.array(sdp_solver.get_y())

    dual_energy = np.dot(b, y)
    primal_energy = np.dot(c, x)

    #dum = 0.0
    #for i in range (0, len(b)):
    #    if np.abs(b[i]) > 1e-6:
    #        dum += b[i] * y[i]
    #        print(b[i], y[i])
    #print()
    #print(dum)

    # action of A^T on y
    ATy = np.array(sdp_solver.get_ATu(y))

    # action of A on x 
    Ax = np.array(sdp_solver.get_Au(x))

    dual_error = c - z - ATy
    primal_error = Ax - b

    # extract blocks of rdms
    #x = my_sdp.get_rdm_blocks(x)
    #z = my_sdp.get_rdm_blocks(z)
    #c = my_sdp.get_rdm_blocks(c)
    #ATy = my_sdp.get_rdm_blocks(ATy)

    print('')
    print('    * v2RDM electronic energy: %20.12f' % (primal_energy))
    print('    * v2RDM total energy:      %20.12f' % (primal_energy + mf.energy_nuc()))
    print('')
    print('    ||Ax - b||:                %20.12f' % (np.linalg.norm(primal_error)))
    print('    ||c - ATy - z||:           %20.12f' % (np.linalg.norm(dual_error)))
    print('    |c.x - b.y|:               %20.12f' % (np.linalg.norm(dual_energy - primal_energy)))
    print('')

    # get individual constraint matrices and build up SOS hamiltonian

    # zeroth constraint
    a0_y = my_sdp.get_constraint_matrix(0) * y[0]

    # all other constraints
    ai_y = np.zeros(len(x), dtype = 'float64')
    for i in range (1, len(y)):
        a = my_sdp.get_constraint_matrix(i)
        ai_y = ai_y + a * y[i]

    # check that individual constraint matrices sum up correctly
    assert(np.isclose(0.0, np.linalg.norm(ATy - a0_y - ai_y)))

    # check that individual constraint matrices sum up correctly, again
    assert(np.isclose(np.linalg.norm(dual_error), np.linalg.norm(c - z - a0_y - ai_y)))

    # sum of squares hamiltonian
    c_sos = z + ai_y
    full_dual =  z + ai_y

    # check that c_sos . x = 0 ... this should approach zero with sufficiently tight convergence
    sos_energy = np.dot(c_sos, x)
    #print(sos_energy)

    # sum of squares hamiltonian, blocked
    c_sos = my_sdp.get_rdm_blocks(c_sos)
    c_mat = my_sdp.get_rdm_blocks(np.array(c))
    z = my_sdp.get_rdm_blocks(z)

    import scipy
    for i in range (0, len(c_sos)):
        block = my_sdp.blocks[i]
        idx = my_sdp.get_block_id(block)
        w = scipy.linalg.eigh(c_sos[idx], eigvals_only=True)
        print('    most negative eigenvalue of the %5s block of the SOS hamiltonian: %20.12f' % (block, w[0]) )
    print()

    c2aaaa_mat = c_mat[3][:nmo*nmo, :nmo*nmo]
    c2bbbb_mat = c_mat[3][nmo*nmo:, nmo*nmo:]
    c2aabb_mat = c_mat[3][:nmo*nmo, nmo*nmo:]
    c2bbaa_mat = c_mat[3][nmo*nmo:, :nmo*nmo]
    assert np.allclose(c2aaaa_mat, c2bbbb_mat)
    assert np.allclose(c2aaaa_mat, c2aabb_mat)
    assert np.allclose(c2aaaa_mat, c2bbaa_mat)

    c2ab_mat = c_mat[4]
    c2ba_mat = c_mat[5]
    assert np.allclose(c2ab_mat, 0)
    assert np.allclose(c2ba_mat, 0)

    c2aaaa = c2aaaa_mat.reshape((nmo, nmo, nmo, nmo)).transpose(0, 1, 3, 2)
    c2bbbb = c2bbbb_mat.reshape((nmo, nmo, nmo, nmo)).transpose(0, 1, 3, 2)
    c2aabb = c2aabb_mat.reshape((nmo, nmo, nmo, nmo)).transpose(0, 1, 3, 2)
    c2bbaa = c2bbaa_mat.reshape((nmo, nmo, nmo, nmo)).transpose(0, 1, 3, 2)
    c2ab = c2ab_mat.reshape((nmo, nmo, nmo, nmo)).transpose((0, 1, 3, 2))
    c2ba = c2ba_mat.reshape((nmo, nmo, nmo, nmo)).transpose((0, 1, 3, 2))
    c1a_mat = c_mat[1]
    c1b_mat = c_mat[2]

    hamiltonian = of.FermionOperator()
    for p, q, s, r in product(range(nmo), repeat=4):
        for sigma, tau in product(range(2), repeat=2):
            # hamiltonian += of.FermionOperator(((2 * p + sigma, 1), (2 * q + sigma, 0), (2 * s + tau, 1), (2 * r + tau, 0)), coefficient=0.5 * tei[p, q, s, r])
            hamiltonian += of.FermionOperator(((2 * p + sigma, 1), (2 * q + sigma, 0), (2 * s + tau, 1), (2 * r + tau, 0)), coefficient=c2aaaa[p, q, s, r])
    for p, q in product(range(nmo), repeat=2):
        hamiltonian += of.FermionOperator(((2 * p, 1), (2 * q, 0)), coefficient=c1a_mat[p, q])
        hamiltonian += of.FermionOperator(((2 * p + 1, 1), (2 * q + 1, 0)), coefficient=c1b_mat[p, q])

    hamiltonian_matrix = of.get_sparse_operator(hamiltonian).todense()
    n_sz_index = []
    nso = 2 * nmo
    for ii in range(4**nmo):
        ket = np.binary_repr(ii, width=nso)
        ket_a = ket[::2]
        ket_a_int = int(ket_a, 2)
        ket_b = ket[1::2]
        ket_b_int = int(ket_b, 2)
        if ket_a.count('1') == nalpha and ket_b.count('1') == nbeta:
            n_sz_index.append(ii)

    hamiltonian_n_sz_matrix = hamiltonian_matrix[:, n_sz_index]
    hamiltonian_n_sz_matrix = hamiltonian_n_sz_matrix[n_sz_index, :]
    # hamiltonian_n_sz_matrix = hamiltonian_matrix
    w, v = np.linalg.eigh(hamiltonian_n_sz_matrix)
    print("Nicks CI roots")
    print(w)

    my_ci = pyscf.fci.FCI(mf)
    my_ci.nroots = len(w)
    my_ci.kernel()
    print("PYSCF ROOTS")
    pyscf_roots = my_ci.eci - mf.energy_nuc()
    print(pyscf_roots)
    assert np.allclose(pyscf_roots, w)
    # print("pyscf FCI electroni-energy ", my_ci.e_tot - mf.energy_nuc())


    assert np.allclose(z[1], 0) # this may be true in general...not sure
    assert np.allclose(z[2], 0)
    # z += my_sdp.get_rdm_blocks(ATy)
    z = my_sdp.get_rdm_blocks(full_dual)
    z_aaaa_mat = z[3][:nmo*nmo, :nmo*nmo]
    z_bbbb_mat = z[3][nmo*nmo:, nmo*nmo:]
    z_aabb_mat = z[3][:nmo*nmo, nmo*nmo:]
    z_bbaa_mat = z[3][nmo*nmo:, :nmo*nmo]
    z_ab_mat = z[4]
    z_ba_mat = z[5]

    z_aaaa = z_aaaa_mat.reshape((nmo, nmo, nmo, nmo)).transpose(0, 1, 3, 2)
    z_bbbb = z_bbbb_mat.reshape((nmo, nmo, nmo, nmo)).transpose(0, 1, 3, 2)
    z_aabb = z_aabb_mat.reshape((nmo, nmo, nmo, nmo)).transpose(0, 1, 3, 2)
    z_bbaa = z_bbaa_mat.reshape((nmo, nmo, nmo, nmo)).transpose(0, 1, 3, 2)
    z_ab = z_ab_mat.reshape((nmo, nmo, nmo, nmo)).transpose(0, 1, 3, 2)
    z_ba = z_ba_mat.reshape((nmo, nmo, nmo, nmo)).transpose(0, 1, 3, 2)

    # test if I stored the coefficients correctly
    for p, q, s, r in product(range(nmo), repeat=4):
        pq = p * nmo + q
        rs = r * nmo + s
        assert np.isclose(z_aaaa_mat[pq, rs], z_aaaa[p, q, s, r])
        assert np.isclose(z_bbbb_mat[pq, rs], z_bbbb[p, q, s, r])
        assert np.isclose(z_aabb_mat[pq, rs], z_aabb[p, q, s, r])
        assert np.isclose(z_bbaa_mat[pq, rs], z_bbaa[p, q, s, r])
        assert np.isclose(z_aabb_mat[pq, rs], z_bbaa_mat[rs, pq])
        assert np.isclose(z_ab_mat[pq, rs], z_ab[p, q, s, r])
        assert np.isclose(z_ba_mat[pq, rs], z_ba[p, q, s, r])

    for p, q, s, r in product(range(nmo), repeat=4):
        # this is aabb(pq, rs) == bbaa(rs, pq), aabb[p, q, s, r] == bbaa[r, s, q, p]
        assert np.isclose(z_aabb[p, q, s, r], z_bbaa[r, s, q, p])

    big_z_dict = {}
    big_z_dict[(0, 0)] = z_aaaa
    big_z_dict[(1, 1)] = z_bbbb
    big_z_dict[(0, 1)] = z_aabb
    big_z_dict[(1, 0)] = z_bbaa
    dual_hamiltonian = of.FermionOperator()
    for p, q, r, s in product(range(nmo), repeat=4):
        for sigma, tau in product(range(2), repeat=2):
            dual_hamiltonian += of.FermionOperator(((2 * p + sigma, 1), (2 * q + sigma, 0),
                                                    (2 * r + tau, 1), (2 * s + tau, 0)), 
                                                    coefficient=big_z_dict[(sigma, tau)][p, q, r, s])
        dual_hamiltonian += of.FermionOperator(((2 * p, 1), (2 * q + 1, 0), 
                                                (2 * r + 1, 1), (2 * s, 0)),
                                                coefficient=z_ab[p, q, r, s])
        dual_hamiltonian += of.FermionOperator(((2 * p + 1, 1), (2 * q, 0), 
                                                (2 * r, 1), (2 * s + 1, 0)),
                                                coefficient=z_ba[p, q, r, s])
    for p, q in product(range(nmo), repeat=2):
        dual_hamiltonian += of.FermionOperator(((2 * p, 1), (2 * q, 0)), coefficient=z[1][p, q])
        dual_hamiltonian += of.FermionOperator(((2 * p + 1, 1), (2 * q + 1, 0)), coefficient=z[2][p, q])


    dual_hamiltonian_matrix = of.get_sparse_operator(dual_hamiltonian).todense()
    dual_hamiltonian_matrix_n = dual_hamiltonian_matrix[:, n_sz_index]
    dual_hamiltonian_matrix_n = dual_hamiltonian_matrix_n[n_sz_index, :]

    w2, v2 = np.linalg.eigh(dual_hamiltonian_matrix_n)
    print("These two sets of numbers should be the same")
    print(w2)#  + dual_energy)
    print(w)
    # print("L2 - norm diff ", np.linalg.norm(w2 + dual_energy - w))


# L2 norm diff of w2 + dual_energy and w
# HeH+
# 1.0E-3, 0.09426509678581267
# 1.0E-4, 0.09426530431254303
# 1.0E-5, 0.09426538000770822
# 1.0E-6, 0.09426533107788634

# He2H2
# 1.0E-3, 0.15728569394051634
# 1.0E-4, 0.15868251542530862
# 1.0E-5, 0.2572867330719202
if __name__ == "__main__":
    main()


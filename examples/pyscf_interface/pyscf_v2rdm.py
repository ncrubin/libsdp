"""
Driver for variational two-electron reduced-density matrix method. Integrals come from PySCF
"""
import numpy as np
from numpy import einsum

import sys
sys.path.insert(0, '../../.')

import libsdp
from v2rdm_sdp import v2rdm_sdp
from g2_v2rdm_sdp import g2_v2rdm_sdp

import pyscf

def main():

    # build molecule
    mol = pyscf.M(
        atom='B 0 0 0; H 0 0 1.0',
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
    options.sdp_error_convergence     = 1e-4
    options.sdp_objective_convergence = 1e-4
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

    # check that c_sos . x = 0 ... this should approach zero with sufficiently tight convergence
    sos_energy = np.dot(c_sos, x)
    #print(sos_energy)

    # sum of squares hamiltonian, blocked
    c_sos = my_sdp.get_rdm_blocks(c_sos)
    z = my_sdp.get_rdm_blocks(z)

    import scipy
    for i in range (0, len(c_sos)):
        block = my_sdp.blocks[i]
        idx = my_sdp.get_block_id(block)
        w = scipy.linalg.eigh(c_sos[idx], eigvals_only=True)
        print('    most negative eigenvalue of the %5s block of the SOS hamiltonian: %20.12f' % (block, w[0]) )
    print()

if __name__ == "__main__":
    main()


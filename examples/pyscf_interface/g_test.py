import itertools
from pyscf import gto, scf, ao2mo, fci
from pyscf.fci.cistring import make_strings

import numpy as np

import fqe

import openfermion as of
from openfermion.utils.rdm_mapping_functions import map_two_pdm_to_particle_hole_dm


def pyscf_to_fqe_wf(
    pyscf_cimat: np.ndarray, pyscf_mf=None, norbs=None, nelec=None
) -> fqe.Wavefunction:
    if pyscf_mf is None:
        assert norbs is not None
        assert nelec is not None
    else:
        mol = pyscf_mf.mol
        nelec = mol.nelec
        norbs = pyscf_mf.mo_coeff.shape[1]

    # Get alpha and beta strings
    norb_list = tuple(list(range(norbs)))
    n_alpha_strings = [x for x in make_strings(norb_list, nelec[0])]
    n_beta_strings = [x for x in make_strings(norb_list, nelec[1])]

    # get fqe Wavefunction object to populate
    fqe_wf_ci = fqe.Wavefunction([[sum(nelec), nelec[0] - nelec[1], norbs]])
    fqe_data_ci = fqe_wf_ci.sector((sum(nelec), nelec[0] - nelec[1]))
    fqe_graph_ci = fqe_data_ci.get_fcigraph()

    # get coeff mat to populate Wavefunction object
    fqe_orderd_coeff = np.zeros(
        (fqe_graph_ci.lena(), fqe_graph_ci.lenb()), dtype=np.complex128
    )  # only works for complex128 right now
    for paidx, pyscf_alpha_idx in enumerate(n_alpha_strings):
        for pbidx, pyscf_beta_idx in enumerate(n_beta_strings):
            fqe_orderd_coeff[
                fqe_graph_ci.index_alpha(pyscf_alpha_idx), fqe_graph_ci.index_beta(pyscf_beta_idx)
            ] = pyscf_cimat[paidx, pbidx]

    # populate Wavefunction object
    fqe_data_ci.coeff = fqe_orderd_coeff
    return fqe_wf_ci

def main():
    mol = gto.M()
    mol.atom = [['Li', 0, 0, 0], ['H', 0, 0, 1.4]]
    mol.basis = 'sto-3g'
    mol.build()
    nelec = mol.nelectron
    sz = 0
    nalpha = nelec // 2
    nbeta = nelec // 2

    mf = scf.RHF(mol)
    mf.kernel()

    # check if integrals in chem ordering
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
    norb = oei.shape[0]
    nso = 2 * norb

    myci = fci.FCI(mf) 
    roots, wfs = myci.kernel(nroots=2)

    h1 = oei - 0.5 * np.einsum('irrj->ij', tei)

    gs_e, gs_wfn = roots[0], pyscf_to_fqe_wf(wfs[0], pyscf_mf=mf)
    opdm, tpdm = gs_wfn.sector((nelec, sz)).get_openfermion_rdms()
    opdma = opdm[::2, ::2]
    opdmb = opdm[1::2, 1::2]
    phdm = map_two_pdm_to_particle_hole_dm(tpdm, opdm) # spin-orbital phdm

    kdelta = np.eye(nso)
    hpdm = np.zeros_like(phdm)
    for p, q, r, s in itertools.product(range(nso), repeat=4):
        hpdm[p, q, r, s] = kdelta[p, q] * kdelta[r, s]
        hpdm[p, q, r, s] -= kdelta[p, q] * opdm[s, r]
        hpdm[p, q, r, s] -= kdelta[r, s] * opdm[q, p]
        hpdm[p, q, r, s] += phdm[q, p, s, r]
    hpdm_mat = hpdm.transpose(0, 1, 3, 2).reshape((nso**2, nso**2))
    w, v = np.linalg.eigh(hpdm_mat)
    assert np.isclose(hpdm_mat.trace(), (nso - nelec) * (nelec + 1))

    phdm_aaaa = phdm[::2, ::2, ::2, ::2]
    phdm_bbbb = phdm[1::2, 1::2, 1::2, 1::2]
    phdm_aabb = phdm[::2, ::2, 1::2, 1::2]
    phdm_bbaa = phdm[1::2, 1::2, ::2, ::2]
    phdm_abba = phdm[::2, 1::2, 1::2, ::2]
    phdm_baab = phdm[1::2, ::2, ::2, 1::2]

    gs_e_test_one_body = np.einsum('ij,ij', h1, opdma).real + np.einsum('ij,ij', h1, opdmb).real
    gs_e_test_two_body = np.einsum('ijkl,ijkl', tei, phdm_aaaa).real + np.einsum('ijkl,ijkl', tei, phdm_bbbb).real + \
                         np.einsum('ijkl,ijkl', tei, phdm_aabb).real + np.einsum('ijkl,ijkl', tei, phdm_bbaa).real
    gs_e_test_two_body *= 0.5
    assert np.isclose(gs_e_test_one_body + gs_e_test_two_body + mf.energy_nuc(),
                      gs_e)

    # test contraction to D1a
    d1a  = 0.25 / (norb - nbeta) * np.einsum('irrj', phdm_abba)
    d1a += 0.25 / (norb - nalpha + 1) * np.einsum('irrj', phdm_aaaa)
    d1a += 0.25 / nbeta * np.einsum('ijrr', phdm_aabb)
    d1a += 0.25 / nbeta * np.einsum('rrji', phdm_bbaa)
    assert np.allclose(d1a, opdma)
    
    # test contraction D1b
    d1b  = 0.25 / (norb - nalpha) * np.einsum('irrj', phdm_baab)
    d1b += 0.25 / (norb - nbeta + 1) * np.einsum('irrj', phdm_bbbb)
    d1b += 0.25 / nalpha * np.einsum('ijrr', phdm_bbaa)
    d1b += 0.25 / nalpha * np.einsum('rrji', phdm_aabb)
    assert np.allclose(d1b, opdmb)



if __name__ == "__main__":
    main()
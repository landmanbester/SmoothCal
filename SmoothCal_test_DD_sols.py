"""
Created 17/04/2018

@author Landman Bester

Testing time + frequency + direction SmoothCal

"""

import numpy as np
import Simulator
import utils
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.sparse.linalg import cg

def interpolate_gains(K, Kfull, g_ML, Sigmayinv, Na, Nnu, Nt, Nsource, Ndir):
    """
    
    :param xp: points at which we want to reconstruct gains
    :param g_ML: maximum likelihood soln
    :param K: prior covaraince matrix
    :param Kfull: the prior covariance on the grid we want to interpolate to
    :param Sigmayinv: inverse of noise covariance
    :param Na: number of antennae
    :param Nnu: number of channels
    :param Nt: number of times
    :param Nsource: number of directions (in soln)
    :return: 
    """
    gmean = np.zeros([Na, Nnu * Ndir * Nt])
    for p  in xrange(Na):
        # get pinv
        Sigmayinvtmp = np.ma.zeros([Nnu * Nsource * Nt, Nnu * Nsource * Nt], dtype=np.complex128)
        for v in xrange(Nnu):
            Sigmayinvtmp[v * Nt * Nsource:(v + 1) * Nt * Nsource, v * Nt * Nsource:(v + 1) * Nt * Nsource] = Sigmayinv[p, v]
        Ky = Kfull + np.linalg.pinv(Sigmayinvtmp)
        # get mean
        gmean[p] = Kfull.dot(np.linalg.solve(Ky, g_ML[p].reshape(Nnu * Nsource * Nt)))
    return gmean

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def get_index_set(Sigmayinv, Nnu, Nt, Nsource):
    """
    Extracts non-zero indices of Sigmayinv
    :param Nnu: 
    :param Nt: 
    :param Nsource: 
    :return: 
    """
    # get non-zero indices of first block (the not so smart way)
    #I = np.argwhere(np.abs(Sigmayinv[0, 0]) >= 1e-13).squeeze()
    I = np.argwhere(np.ones_like(Sigmayinv[0, 0])).squeeze()
    Irow = I[:, 0]
    Icol = I[:, 1]
    row_indices = []
    col_indices = []
    Nblock = Nt * Nsource
    for v in xrange(Nnu):
        row_indices.append(list(v*Nblock + Irow))
        col_indices.append(list(v*Nblock + Icol))
    return flatten_list(row_indices), flatten_list(col_indices)


def give_j_and_Sigmayinv(A, W, V, Na, Nnu, j, Sigmayinv):
    for p in xrange(Na):
        for v in xrange(Nnu):
            j[p, v] = A[p, v].conj().T.dot(V[p, v]*W[p, v])
            Sigmayinv[p, v] = A[p, v].conj().T.dot(W[p, v, :, None]*A[p, v])  # should do diagdot more efficiently
    return j, Sigmayinv


def give_V_and_W(Vpq, Wpq, Na, Nnu, Nt):
    """
    Swaps axes and reshapes array from [Na, Na, Nnu, Nt] into [Na, Nnu, Nt*Na] required 
    for per antennae and frequency response
    """
    return Vpq.swapaxes(1, 2).swapaxes(2, 3).reshape(Na, Nnu, Na*Nt), \
           Wpq.swapaxes(1, 2).swapaxes(2, 3).reshape(Na, Nnu, Na*Nt)


def give_response(Xpq_DD, gnow, Na, Nnu, Nt, Nsource, A):
    for p in xrange(Na):
        for v in xrange(Nnu):
            for s in xrange(Nsource):
                for t in xrange(Nt):
                    Rpnuts = Xpq_DD[p, :, v, t, s] * gnow[:, v, s, t].conj()
                    A[p, v, t * Na:(t + 1) * Na, s*Nt + t] = Rpnuts
    return A


def give_SmoothCal_direct_update(j, Sigmayinv, gains, K, Na, Nnu, Nt, Nsource, Ix, Iy):
    jtmp = j.reshape(Na, Nnu * Nsource * Nt)
    gainstmp = gains.reshape(Na, Nnu * Nsource * Nt)
    Kfull = utils.kron_kron(K)
    for p in xrange(Na):
        Sigmayinvtmp = np.ma.zeros([Nnu * Nsource * Nt, Nnu * Nsource * Nt], dtype=np.complex128)
        for v in xrange(Nnu):
            Sigmayinvtmp[v * Nt * Nsource:(v + 1) * Nt * Nsource, v * Nt * Nsource:(v + 1) * Nt * Nsource] = Sigmayinv[p, v]
        Ky = Kfull + np.linalg.pinv(Sigmayinvtmp)
        # plt.figure()
        # plt.imshow(np.abs(Ky))
        # plt.colorbar()
        # plt.show()
        # plt.close()
        rhs_vec = Kfull.dot(jtmp[p]) + gainstmp[p]
        tmp = np.linalg.solve(Ky, rhs_vec)
        rhs_vec = rhs_vec - Kfull.dot(tmp)
        gainstmp[p] = (gainstmp[p] + rhs_vec)/2.0
    return gainstmp.reshape(Na, Nnu, Nsource, Nt)

def give_SmoothCal_update(j, Sigmayinv, gains, K, Na, Nnu, Nt, Nsource, Ix, Iy):
    jtmp = j.reshape(Na, Nnu * Nsource * Nt)
    gainstmp = gains.reshape(Na, Nnu * Nsource * Nt)
    for p in xrange(Na):
        data = []
        for v in xrange(Nnu):
            data.append(np.linalg.pinv(Sigmayinv[p, v])[Ix[0:Nt**2*Nsource**2], Iy[0:Nt**2*Nsource**2]])
        data = flatten_list(data)
        Sigmay = sps.csr_matrix((data, (Ix, Iy)), shape=(Nnu * Nsource * Nt, Nnu * Nsource * Nt))
        mvec = lambda x: utils.kron_matvec(K, x) + Sigmay.dot(x)
        Ky = sps.linalg.LinearOperator(dtype=np.float64, shape=(Nnu * Nsource * Nt, Nnu * Nsource * Nt), matvec=mvec)
        plt.figure()
        plt.imshow(np.abs(Ky))
        plt.colorbar()
        plt.show()
        plt.close()
        rhs_vec = utils.kron_matvec(K, jtmp[p]) + gainstmp[p]
        # plt.figure()
        # plt.plot(rhs_vec.real, 'b')
        # plt.plot(rhs_vec.imag, 'r')
        # plt.show()
        # plt.close()
        tmp = sps.linalg.cg(Ky, rhs_vec, tol=1e-5)
        if tmp[1] > 0:
            print "Tolerance not achieved"
        rhs_vec = rhs_vec - utils.kron_matvec(K, tmp[0])
        gainstmp[p] = (gainstmp[p] + rhs_vec)/2.0
    return gainstmp.reshape(Na, Nnu, Nsource, Nt)


def give_StefCal_update(j, Sigmayinv, gains, Na, Nnu, Nt, Nsource):
    gainstmp = np.zeros([Na, Nnu, Nt*Nsource], dtype=np.complex128)
    for p in xrange(Na):
        for v in xrange(Nnu):
            gainstmp[p, v] = (np.linalg.pinv(Sigmayinv[p, v]).dot(j[p, v]) + gains[p, v].reshape(Nt*Nsource))/2.0
    return gainstmp.reshape(Na, Nnu, Nsource, Nt)


def DD_StefCal(Vpq, Wpq, Xpq_DD, gains, Na, Nnu, Nt, Nsource, A, Sigmayinv, j, maxiter=100, tol=1e-3):
    diff = 1.0
    i = 0
    while i < maxiter and diff >= tol:
        gold = gains.copy()
        A = give_response(Xpq_DD, gold, Na, Nnu, Nt, Nsource, A)
        j, Sigmayinv = give_j_and_Sigmayinv(A, Wpq, Vpq, Na, Nnu, j, Sigmayinv)
        gains = give_StefCal_update(j, Sigmayinv, gains, Na, Nnu, Nt, Nsource)
        # roll back the phase
        phase = np.angle(gains[0])
        gains *= np.exp(-1.0j*phase[None, :, :, :])
        diff = (np.abs(gains - gold)).max()
        i += 1
        print "At iter %i max difference is %f" % (i, diff)

    if i >= maxiter:
        print "Maximum iterations reached"

    return gains


def DD_SmoothCal(Vpq, Wpq, Xpq_DD, gains, K, Na, Nnu, Nt, Nsource, A, Sigmayinv, j, Ix, Iy, maxiter=100, tol=1e-3):
    diff = 1.0
    i = 0
    while i < maxiter and diff >= tol:
        gold = gains.copy()
        A = give_response(Xpq_DD, gold, Na, Nnu, Nt, Nsource, A)
        j, Sigmayinv = give_j_and_Sigmayinv(A, Wpq, Vpq, Na, Nnu, j, Sigmayinv)
        gains = give_SmoothCal_direct_update(j, Sigmayinv, gains, K, Na, Nnu, Nt, Nsource, Ix, Iy)
        # roll back the phase
        phase = np.angle(gains[0])
        gains *= np.exp(-1.0j*phase[None, :, :, :])
        diff = (np.abs(gains - gold)).max()
        i += 1
        print "At iter %i max difference is %f" % (i, diff)

    if i >= maxiter:
        print "Maximum iterations reached"

    return gains


if __name__ == "__main__":
    #np.random.seed(123456)
    # Cubical ordering [s, model, time, freq, ant, ant, corr, corr]
    # set time and freq domain
    Nt = 10
    tmin = 0.0
    tmax = 100.0
    if Nt == 1:
        t = np.array([0.0])
    else:
        t = np.linspace(tmin, tmax, Nt)

    Nnu = 10
    numin = 1.0
    numax = 10.0
    if Nnu == 1:
        nu = np.array([1.0])
    else:
        nu = np.linspace(numin, numax, Nnu)

    Npix = 35
    lmax = 0.25
    mmax = 0.25
    l = np.linspace(-lmax, lmax, Npix)
    dell = l[1] - l[0]
    m = np.linspace(-mmax, mmax, Npix)
    Ns = [Nnu, Nt, Npix, Npix]

    # covariance params
    theta_nu = np.array([0.5, 0.5])
    theta_t = np.array([0.25, 1.5])
    theta_l = np.array([0.25, 1.0])
    theta_m = np.array([0.25, 1.0])

    thetas = np.array([theta_nu, theta_t, theta_l, theta_m])

    # model image
    Nsource = 4
    max_I = 50.0
    min_I = 10.0
    IM, lm, locs, alphas = Simulator.sim_sky(Npix, Nsource, max_I, min_I, lmax, mmax, nu, nu[Nnu//2])

    # gains
    Na = 9
    gains_full, K_full = Simulator.sim_DD_gains(Na, Ns, thetas,
                                      bounds=[(tmin, tmax), (numin, numax), (-lmax, lmax), (-mmax, mmax)])

    # for p in xrange(Na):
    #     for i in xrange(Nnu):
    #         for j in xrange(Nt):
    #             plt.figure()
    #             plt.imshow(gains_full[p, i, j].real)
    #             plt.colorbar()
    #             plt.show()
    #             plt.close()

    # get gains at source locations
    gains = np.zeros([Na, Nnu, Nt, Nsource], dtype=np.complex128)
    Ks = np.zeros([Nsource, Nsource], dtype=np.float64)
    #Km = np.zeros([Nsource, Nsource], dtype=np.float64)
    ll, mm = np.meshgrid(l, m)
    for i in xrange(Nsource):
        gains[:, :, :, i] = gains_full[:, :, :, locs[i][0], locs[i][1]]
        for j in xrange(Nsource):
            li, mi = lm[i]
            lj, mj = lm[j]
            klij = utils.sqexp(li-lj, theta_l)
            kmij = utils.sqexp(mi-mj, theta_m)
            Ks[i, j] = klij*kmij

    K = np.array([K_full[0], Ks, K_full[1]])

    # set zeroth antennas phase to zero
    phi = np.angle(gains[0])
    gains *= np.exp(-1.0j*phi[None, :, :, :])

    # uv-coverage
    umax = 10.0
    vmax = 10.0
    upq, vpq, pqlist, N = Simulator.sim_uv(Na, Nt, umax, vmax, rot_params=(1.0, 1.0))
    umax = np.abs(upq).max()

    assert(1.0/(2*umax) > dell)

    # plt.figure()
    # plt.plot(upq.flatten(), vpq.flatten(), 'xr')
    # plt.show()

    # data
    Xpq_DD = np.zeros([Na, Na, Nnu, Nt, Nsource], dtype=np.complex128)
    Xpq = np.zeros([Na, Na, Nnu, Nt], dtype=np.complex128)
    Xpq, Xpq_DD = utils.R_DD_model(IM, upq, vpq, lm, pqlist, nu, nu[Nnu//2], np.ones_like(gains), Xpq, Nnu, Nt, Nsource, Xpq_DD)
    sigma = 0.1
    Vpq = np.zeros([Na, Na, Nnu, Nt], dtype=np.complex128)
    Vpq = utils.R_DD(IM, upq, vpq, lm, pqlist, nu, nu[Nnu//2], gains, Vpq, Nnu, Nt, Nsource) + sigma**2*(
            np.random.randn(Na, Na, Nnu, Nt) + 1.0j*np.random.randn(Na, Na, Nnu, Nt))

    # Create mask (autocorrelation)
    I = np.tile(np.diag(np.ones(Na, dtype=np.int8))[:, :, None, None], (1, 1, Nnu, Nt))

    Wpq = np.ones_like(Vpq, dtype=np.float64)/(2*sigma**2)  # should the 2 be there?

    Vpq = np.ma.MaskedArray(Vpq, I, dtype=np.complex128)
    Wpq = np.ma.MaskedArray(Wpq, I, dtype=np.complex128)
    Xpq = np.ma.MaskedArray(Xpq, I, dtype=np.complex128)

    # direction dependent mask
    I_DD = np.tile(np.diag(np.ones(Na, dtype=np.int8))[:, :, None, None, None], (1, 1, Nnu, Nt, Nsource))
    Xpq_DD = np.ma.MaskedArray(Xpq_DD, I_DD, dtype=np.complex128)

    # reshape for response
    Vpq, Wpq = give_V_and_W(Vpq, Wpq, Na, Nnu, Nt)

    # response
    A = np.ma.zeros([Na, Nnu, Nt*Na, Nt*Nsource], dtype=np.complex128)

    # test response
    gains = gains.swapaxes(2, 3)  # we need time on the last axis so reshape does the correct thing
    A = give_response(Xpq_DD, gains, Na, Nnu, Nt, Nsource, A)

    # # should all be true if sigma is zero
    # for p in xrange(Na):
    #     for v in xrange(Nnu):
    #         tmp = A[p, v].dot(gains[p, v].flatten())
    #         print np.allclose(tmp, Vpq[p, v])

    # j amd Sigmayinv
    j = np.ma.zeros([Na, Nnu, Nt*Nsource], dtype=np.complex128)
    Sigmayinv = np.ma.zeros([Na, Nnu, Nt*Nsource, Nt*Nsource], dtype=np.complex128)

    # for testing
    #j, Sigmayinv = give_j_and_Sigmayinv(A, Wpq, Vpq, Na, Nnu, j, Sigmayinv)

    # for p in xrange(Na):
    #     print "p = ", p
    #     #tmp = np.linalg.pinv(Sigmayinv[p, 0])
    #     tmp = Sigmayinv[p, 0]
    #     for s in xrange(Nsource*Nt):
    #         for sp in xrange(Nsource*Nt):
    #             print '{:3.2e}'.format(tmp[s, sp].real),
    #         print
    #
    #     print
    #
    #     for s in xrange(Nsource*Nt):
    #         for sp in xrange(Nsource*Nt):
    #             print '{:3.2e}'.format(tmp[s, sp].imag),
    #         print
    #
    #     print

    # test indices
    Sigmayinv2 = np.ma.zeros([Nnu * Nsource * Nt, Nnu * Nsource * Nt], dtype=np.complex128)

    for v in xrange(Nnu):
        Sigmayinv2[v*Nt*Nsource:(v+1)*Nt*Nsource, v*Nt*Nsource:(v+1)*Nt*Nsource] = Sigmayinv[0, v]

    Ix, Iy = get_index_set(Sigmayinv, Nnu, Nt, Nsource)

    # test ML solution
    gbar = np.ones_like(gains, dtype=np.complex128)
    gbar = DD_StefCal(Vpq, Wpq, Xpq_DD, gbar, Na, Nnu, Nt, Nsource, A, Sigmayinv, j, tol=1e-3, maxiter=20)

    # for p in xrange(Na):
    #     for v in xrange(Nnu):
    #         print (np.abs(gbar[p, v] - gains[p, v])).max(), (np.abs(gbar[p, v] - np.ones_like(gains, dtype=np.complex128))).max()

    # reinitialise just in case
    A = np.ma.zeros([Na, Nnu, Nt*Na, Nt*Nsource], dtype=np.complex128)
    j = np.ma.zeros([Na, Nnu, Nt*Nsource], dtype=np.complex128)
    Sigmayinv = np.ma.zeros([Na, Nnu, Nt*Nsource, Nt*Nsource], dtype=np.complex128)

    # SmoothCal solution
    gbar2 = np.ones_like(gains, dtype=np.complex128)
    gbar2 = DD_SmoothCal(Vpq, Wpq, Xpq_DD, gbar2, K, Na, Nnu, Nt, Nsource, A, Sigmayinv, j, Ix, Iy, tol=1e-3, maxiter=20)

    # interpolate spacial axes
    # first we need an ML gain soln
    # j, Sigmayinv, gains, Na, Nnu, Nt, Nsource
    # get response
    A = give_response(Xpq_DD, gbar2, Na, Nnu, Nt, Nsource, A)

    # get j and Sigmayinv
    j, Sigmayinv = give_j_and_Sigmayinv(A, Wpq, Vpq, Na, Nnu, j, Sigmayinv)

    # get ML soln
    g_ML = give_StefCal_update(j, Sigmayinv, gbar2, Na, Nnu, Nt, Nsource)

    # now interpolate


    # for p in xrange(Na):
    #     for v in xrange(Nnu):
    #         print (np.abs(gbar2[p, v] - gains[p, v])).max(), (np.abs(gbar2[p, v] - np.ones_like(gains, dtype=np.complex128))).max()

    # check result
    for s in xrange(Nsource):
        fig, ax = plt.subplots(nrows=3, ncols=Na, figsize=(15,8))
        for p in xrange(Na):
            ax[0, p].imshow(np.abs(gains[p, :, s, :]))
            ax[1, p].imshow(np.abs(gbar[p, :, s, :]))
            ax[2, p].imshow(np.abs(gbar[p, :, s, :] - gains[p, :, s, :]))

        fig.tight_layout()
        plt.show()
        plt.close()

    # check result
    for s in xrange(Nsource):
        fig, ax = plt.subplots(nrows=3, ncols=Na, figsize=(15,8))
        for p in xrange(Na):
            ax[0, p].imshow(np.abs(gains[p, :, s, :]))
            ax[1, p].imshow(np.abs(gbar2[p, :, s, :]))
            ax[2, p].imshow(np.abs(gbar2[p, :, s, :]) - np.abs(gains[p, :, s, :]))

        fig.tight_layout()
        plt.show()
        plt.close()

    # fig, ax = plt.subplots(nrows=1, ncols=Na, figsize=(15, 8))
    # tmp = np.abs(gbar2[0] - gains[0])
    # for p in xrange(1, Na):
    #     tmp2 = np.abs(gbar2[p] - gains[p])
    #     ax[p].imshow(tmp-tmp2)
    #     print np.max(tmp-tmp2), (tmp - tmp2).min()
    #
    # fig.tight_layout()
    # plt.show()











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


def give_SmoothCal_update(j, Sigmayinv, gains, K, Na, Nnu, Nt):
    jtmp = j.reshape(Na, Nnu * Nt)
    Sigmayinvtmp = Sigmayinv.reshape(Na, Nnu * Nt)
    gainstmp = gains.reshape(Na, Nnu * Nt)
    for p in xrange(Na):
        mvec = lambda x: utils.kron_matvec(K, x) + x/Sigmayinvtmp[p]
        Ky = sps.linalg.LinearOperator(dtype=np.float64, shape=(Nnu * Nt, Nnu * Nt), matvec=mvec)
        rhs_vec = utils.kron_matvec(K, jtmp[p]) + gainstmp[p]
        tmp = sps.linalg.cg(Ky, rhs_vec, tol=1e-8)
        if tmp[1]>0:
            print "Tolerance not achieved"
        rhs_vec = rhs_vec - utils.kron_matvec(K, tmp[0])
        gainstmp[p] = (gainstmp[p] + rhs_vec)/2.0
    return gainstmp.reshape(Na, Nnu, Nt)


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
        A[...] = give_response(Xpq_DD, gold, Na, Nnu, Nt, Nsource, A)
        j[...], Sigmayinv[...] = give_j_and_Sigmayinv(A, Wpq, Vpq, Na, Nnu, j, Sigmayinv)
        gains[...] = give_StefCal_update(j, Sigmayinv, gains, Na, Nnu, Nt, Nsource)
        diff = np.abs(gains - gold).max()
        i += 1
        print "At iter %i max difference is %f" % (i, diff)

    if i >= maxiter:
        print "Maximum iterations reached"

    return gains


def DD_SmoothCal(Vpq, Wpq, Xpq, gains, K, Na, Nnu, Nt, A, Sigmayinv, j, maxiter=100, tol=1e-3):
    diff = 1.0
    i = 0
    while i<maxiter and diff >= tol:
        gold = gains.copy()
        A[...] = give_response(Xpq, gold, Na, Nnu, Nt, A)
        j[...], Sigmayinv[...] = give_j_and_Sigmayinv(A, Wpq, Vpq, Na, Nnu, j, Sigmayinv)
        gains[...] = give_SmoothCal_update(j, Sigmayinv, gains, K, Na, Nnu, Nt)
        diff = np.abs(gains - gold).max()
        i += 1
        print "At iter %i max difference is %f" % (i, diff)

    if i >= maxiter:
        print "Maximum iterations reached"

    return gains


if __name__ == "__main__":
    # Cubical ordering [s, model, time, freq, ant, ant, corr, corr]
    # set time and freq domain
    Nt = 5
    tmin = 0.0
    tmax = 10.0
    if Nt == 1:
        t = np.array([0.0])
    else:
        t = np.linspace(tmin, tmax, Nt)

    Nnu = 5
    numin = 1.0
    numax = 10.0
    if Nnu == 1:
        nu = np.array([1.0])
    else:
        nu = np.linspace(numin, numax, Nnu)

    Npix = 35
    lmax = 0.1
    mmax = 0.1
    l = np.linspace(-lmax, lmax, Npix)
    dell = l[1] - l[0]
    m = np.linspace(-mmax, mmax, Npix)
    Ns = [Nnu, Nt, Npix, Npix]

    # covariance params
    theta_nu = np.array([0.15, 5.0])
    theta_t = np.array([0.25, 1.0])
    theta_l = np.array([0.1, 0.025])
    theta_m = np.array([0.1, 0.025])

    thetas = np.array([theta_nu, theta_t, theta_l, theta_m])

    # model image
    Nsource = 3
    max_I = 1.0
    min_I = 0.0
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
    Kl = np.zeros([Nsource, Nsource], dtype=np.float64)
    Km = np.zeros([Nsource, Nsource], dtype=np.float64)
    ll, mm = np.meshgrid(l, m)
    for i in xrange(Nsource):
        gains[:, :, :, i] = gains_full[:, :, :, locs[i][0], locs[i][1]]
        for j in xrange(Nsource):
            li, mi = lm[i]
            lj, mj = lm[j]
            klij = utils.sqexp(li-lj, theta_l)
            kmij = utils.sqexp(mi-mj, theta_m)
            Kl[i, j] = klij
            Km[i, j] = kmij

    K = np.array([K_full[0], K_full[1], Kl, Km])

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
    Vpq = utils.R_DD(IM, upq, vpq, lm, pqlist, nu, nu[Nnu//2], gains, Vpq, Nnu, Nt, Nsource) #+ sigma**2*(
            #np.random.randn(Na, Na, Nnu, Nt) + 1.0j*np.random.randn(Na, Na, Nnu, Nt))

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

    # j amd Sigmayinv
    j = np.ma.zeros([Na, Nnu, Nt*Nsource], dtype=np.complex128)
    Sigmayinv = np.ma.zeros([Na, Nnu, Nt*Nsource, Nt*Nsource], dtype=np.complex128)

    #j[...], Sigmayinv[...] = give_j_and_Sigmayinv(A, Wpq, Vpq, Na, Nnu, j, Sigmayinv)

    # Sigmaylist = list(Sigmayinv[0,0])
    #
    # for row in Sigmaylist:
    #     for val in row:
    #         print '{:3.2e}'.format(val.real),
    #     print
    #
    # print
    #
    # for row in Sigmaylist:
    #     for val in row:
    #         print '{:3.2e}'.format(val.imag),
    #     print


    # test response
    gains = gains.swapaxes(2, 3)  # we need time on the last axis so reshape does the correct thing
    # A[...] = give_response(Xpq_DD, gains, Na, Nnu, Nt, Nsource, A)
    #
    # for p in xrange(Na):
    #     for v in xrange(Nnu):
    #         tmp = A[p, v].dot(gains[p, v].flatten())
    #         print np.allclose(tmp, Vpq[p, v])

    # test ML solution
    gbar = np.ones_like(gains, dtype=np.complex128)
    gbar = DD_StefCal(Vpq, Wpq, Xpq_DD, gbar, Na, Nnu, Nt, Nsource, A, Sigmayinv, j, tol=1e-3, maxiter=20)

    for p in xrange(Na):
        for v in xrange(Nnu):
            print (np.abs(gbar[p, v]) - np.abs(gains[p, v])).max()

    # # SmoothCal solution
    # gbar2 = np.ones_like(gains, dtype=np.complex128)
    # gbar2[...] = tf_SmoothCal(Vpq, Wpq, Xpq, gbar2, K, Na, Nnu, Nt, A, Sigmayinv, j, tol=5e-3, maxiter=20)

    # # check result
    # fig, ax = plt.subplots(nrows=3, ncols=Na, figsize=(15,8))
    # for p in xrange(Na):
    #     ax[0, p].imshow(np.abs(gains[p]))
    #     ax[1, p].imshow(np.abs(gbar[p]))
    #     ax[2, p].imshow(np.abs(gbar[p] - gains[p]))
    #
    # fig.tight_layout()
    # plt.show()

    # # check result
    # fig, ax = plt.subplots(nrows=3, ncols=Na, figsize=(15,8))
    # for p in xrange(Na):
    #     ax[0, p].imshow(np.abs(gains[p, 0]))
    #     ax[1, p].imshow(np.abs(gbar[p, 0]))
    #     ax[2, p].imshow(np.abs(gbar[p, 0] - gains[p, 0]))
    #
    # fig.tight_layout()
    # plt.show()
    #
    # fig, ax = plt.subplots(nrows=1, ncols=Na, figsize=(15, 8))
    # tmp = np.abs(gbar2[0] - gains[0])
    # for p in xrange(1, Na):
    #     tmp2 = np.abs(gbar2[p] - gains[p])
    #     ax[p].imshow(tmp-tmp2)
    #     print np.max(tmp-tmp2), (tmp - tmp2).min()
    #
    # fig.tight_layout()
    # plt.show()











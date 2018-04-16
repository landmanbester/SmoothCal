"""
Created 11/04/2018

@author Landman Bester

Testing time + frequency SmoothCal

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
            Sigmayinv[p, v] = np.diag((A[p, v].conj().T.dot(W[p, v, :, None]*A[p, v])).real)  # should do diagdot more efficiently
    return j, Sigmayinv

def give_V_and_W(Vpq, Wpq, Na, Nnu, Nt):
    """
    Swaps axes and reshapes array from [Na, Na, Nnu, Nt] into [Na, Nnu, Nt*Na] required 
    for per antennae and frequency response
    """
    return Vpq.swapaxes(1, 2).swapaxes(2, 3).reshape(Na, Nnu, Na*Nt), \
           Wpq.swapaxes(1, 2).swapaxes(2, 3).reshape(Na, Nnu, Na*Nt)

def give_response(Xpq, gnow, Na, Nnu, Nt, A):
    for p in xrange(Na):
        for v in xrange(Nnu):
            for t in xrange(Nt):
                Rpnut = Xpq[p, :, v, t] * (gnow[:, v, t].conj())
                A[p, v, t * Na:(t + 1) * Na, t] = Rpnut
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

def give_StefCal_update(j, Sigmayinv, gains, Na, Nnu):
    for p in xrange(Na):
        for v in xrange(Nnu):
            gains[p, v] = (j[p, v]/Sigmayinv[p, v] + gains[p, v])/2.0
    return gains

def tf_StefCal(Vpq, Wpq, Xpq, gains, Na, Nnu, Nt, A, Sigmayinv, j, maxiter=100, tol=1e-3):
    diff = 1.0
    i = 0
    while i<maxiter and diff >= tol:
        gold = gains.copy()
        A[...] = give_response(Xpq, gold, Na, Nnu, Nt, A)
        j[...], Sigmayinv[...] = give_j_and_Sigmayinv(A, Wpq, Vpq, Na, Nnu, j, Sigmayinv)
        gains[...] = give_StefCal_update(j, Sigmayinv, gains, Na, Nnu)
        diff = np.abs(gains - gold).max()
        i += 1
        print "At iter %i max difference is %f" % (i, diff)

    if i >= maxiter:
        print "Maximum iterations reached"

    return gains

def tf_SmoothCal(Vpq, Wpq, Xpq, gains, K, Na, Nnu, Nt, A, Sigmayinv, j, maxiter=100, tol=1e-3):
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
    # set time and freq domain
    Nt = 100
    tmin = 0.0
    tmax = 100.0
    t = np.linspace(tmin, tmax, Nt)

    Nnu = 100
    numin = 1.0
    numax = 10.0
    nu = np.linspace(numin, numax, Nnu)

    Ns = [Nnu, Nt]

    # covariance params
    theta_nu = np.array([0.5, 1.5])
    theta_t = np.array([0.25, 0.5])

    thetas = np.array([theta_nu, theta_t])

    # model image
    Npix = 35
    Nsource = 5
    max_I = 2.0
    lmax = 0.1
    mmax = 0.1
    IM, lm, locs = Simulator.sim_sky(Npix, Nsource, max_I, lmax, mmax, nu, nu[Nnu//2])
    #IM[:, 0] = 0.0

    # gains
    Na = 4
    gains, K = Simulator.sim_DI_gains(Na, Ns, thetas, bounds=[(tmin, tmax), (numin, numax)])

    # uv-coverage
    umax = 10.0
    vmax = 10.0
    upq, vpq, pqlist, N = Simulator.sim_uv(Na, Nt, umax, vmax, rot_params=(1.5, 0.75))

    # print upq
    # print vpq

    plt.figure()
    plt.plot(upq.flatten(), vpq.flatten(), 'xr')
    plt.show()

    # import sys
    # sys.exit()

    # data
    Xpq = np.zeros([Na, Na, Nnu, Nt], dtype=np.complex128)
    Xpq = utils.R_DI(IM, upq, vpq, lm, pqlist, nu, nu[Nnu//2], np.ones_like(gains), Xpq, Nnu, Nt, Nsource)
    sigma = 0.1
    Vpq = np.zeros([Na, Na, Nnu, Nt], dtype=np.complex128)
    Vpq = utils.R_DI(IM, upq, vpq, lm, pqlist, nu, nu[Nnu//2], gains, Vpq, Nnu, Nt, Nsource) + sigma**2*(
            np.random.randn(Na, Na, Nnu, Nt) + 1.0j*np.random.randn(Na, Na, Nnu, Nt))

    # Create mask (autocorrelation)
    I = np.tile(np.diag(np.ones(Na, dtype=np.int8))[:, :, None, None], (1, 1, Nnu, Nt))

    Wpq = np.ones_like(Vpq, dtype=np.float64)/(2*sigma**2)

    Vpq = np.ma.MaskedArray(Vpq, I)
    Wpq = np.ma.MaskedArray(Wpq, I)
    Xpq = np.ma.MaskedArray(Xpq, I)

    # reshape for response
    Vpq, Wpq = give_V_and_W(Vpq, Wpq, Na, Nnu, Nt)

    # response
    A = np.ma.zeros([Na, Nnu, Nt*Na, Nt], dtype=np.complex128)

    # get j amd Sigmayinv
    j = np.ma.zeros([Na, Nnu, Nt], dtype=np.complex128)
    Sigmayinv = np.ma.zeros([Na, Nnu, Nt], dtype=np.float64)

    # # test ML solution
    # gbar = np.ones_like(gains, dtype=np.complex128)
    # gbar[...] = tf_StefCal(Vpq, Wpq, Xpq, gbar, Na, Nnu, Nt, A, Sigmayinv, j, tol=5e-3, maxiter=20)

    # SmoothCal solution
    gbar2 = np.ones_like(gains, dtype=np.complex128)
    gbar2[...] = tf_SmoothCal(Vpq, Wpq, Xpq, gbar2, K, Na, Nnu, Nt, A, Sigmayinv, j, tol=5e-3, maxiter=20)

    # # check result
    # fig, ax = plt.subplots(nrows=3, ncols=Na, figsize=(15,8))
    # for p in xrange(Na):
    #     ax[0, p].imshow(np.abs(gains[p]))
    #     ax[1, p].imshow(np.abs(gbar[p]))
    #     ax[2, p].imshow(np.abs(gbar[p] - gains[p]))
    #
    # fig.tight_layout()
    # plt.show()

    # check result
    fig, ax = plt.subplots(nrows=3, ncols=Na, figsize=(15,8))
    for p in xrange(Na):
        ax[0, p].imshow(np.abs(gains[p]))
        ax[1, p].imshow(np.abs(gbar2[p]))
        ax[2, p].imshow(np.abs(gbar2[p] - gains[p]))

    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=Na, figsize=(15, 8))
    tmp = np.abs(gbar2[0] - gains[0])
    for p in xrange(1, Na):
        tmp2 = np.abs(gbar2[p] - gains[p])
        ax[p].imshow(tmp-tmp2)
        print np.max(tmp-tmp2), (tmp - tmp2).min()

    fig.tight_layout()
    plt.show()











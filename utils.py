
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from astropy.io import fits

speed_of_light = 2.99792458e8

#@jit(nopython=True, nogil=True, cache=True)
def R_DI_T(IM, upq, vpq, lm, pqlist, gains, Xpq, Nt, Nsource):
    """
    Full response operator including DDE's coded as a DFT.
    Note empty Xpq passed in for jitting purposes (don't want to be creating arrays inside a jitted function)
    :param IM: Nnu x Nsource array containing model image at Nnu freqs
    :param upq: N x Nt array of baseline coordinates in units of lambda at the reference frequency
    :param vpq: N x Nt array of baseline coordinates in units of lambda at the reference frequency
    :param lm: Nsource x 2 array of sky coordinates for sources
    :param pqlist: a list of antennae pairs (used for the iterator)
    :param gains: Na x Nnu x Nt x Nsource array containg direction dependent gains for antennaes
    :param DD: whether to applu DD gains or not
    :return: Xpq: Na x Na x Nnu x Nt array to hold model visibilities
    """
    for k, pq in enumerate(iter(pqlist)):
        p = pq[0]
        q = pq[1]
        for j in xrange(Nt):
            u = upq[k, j]
            v = vpq[k, j] # convert units
            for s in xrange(Nsource):
                l, m = lm[s]
                complex_phase = 2.0*np.pi*(u*l + v*m)
                K = np.cos(complex_phase) - 1.0j*np.sin(complex_phase)
                Xpq[p, q, j] += K * gains[p, j] * IM[s] * np.conj(gains[q, j])
            Xpq[q, p, j] = np.conj(Xpq[p, q, j])
    return Xpq

@jit(nopython=True, nogil=True, cache=True)
def R_DI(IM, upq, vpq, lm, pqlist, freqs, ref_freq, gains, Xpq, Nnu, Nt, Nsource):
    """
    Full response operator including DDE's coded as a DFT.
    Note empty Xpq passed in for jitting purposes (don't want to be creating arrays inside a jitted function)
    :param IM: Nnu x Nsource array containing model image at Nnu freqs
    :param upq: N x Nt array of baseline coordinates in units of lambda at the reference frequency
    :param vpq: N x Nt array of baseline coordinates in units of lambda at the reference frequency
    :param lm: Nsource x 2 array of sky coordinates for sources
    :param pqlist: a list of antennae pairs (used for the iterator)
    :param freqs: array of frequencies
    :param ref_freq: reference frequency
    :param gains: Na x Nnu x Nt x Nsource array containg direction dependent gains for antennaes
    :param DD: whether to applu DD gains or not
    :return: Xpq: Na x Na x Nnu x Nt array to hold model visibilities
    """
    ref_wavelength = speed_of_light/ref_freq
    for k, pq in enumerate(iter(pqlist)):
        p = pq[0]
        q = pq[1]
        for i in xrange(Nnu):
            wavelength = speed_of_light/freqs[i]
            for j in xrange(Nt):
                u = upq[k, j]*ref_wavelength/wavelength
                v = vpq[k, j]*ref_wavelength/wavelength  # convert units
                for s in xrange(Nsource):
                    l, m = lm[s]
                    complex_phase = -2.0*np.pi*(u*l + v*m)
                    K = np.cos(complex_phase) + 1.0j*np.sin(complex_phase)
                    Xpq[p, q, i, j] += K * gains[p, i, j] * IM[i, s] * np.conj(gains[q, i, j])
                Xpq[q, p, i, j] = np.conj(Xpq[p, q, i, j])
    return Xpq

#@jit(nopython=True, nogil=True, cache=True)
def R_DD_model(IM, upq, vpq, lm, pqlist, freqs, ref_freq, gains, Xpq, Nnu, Nt, Nsource, Xpq_DD):
    """
    Full response operator including DDE's coded as a DFT.
    Note empty Xpq passed in for jitting purposes (don't want to be creating arrays inside a jitted function)
    :param IM: Nnu x Nsource array containing model image at Nnu freqs
    :param upq: N x Nt array of baseline coordinates in units of lambda at the reference frequency
    :param vpq: N x Nt array of baseline coordinates in units of lambda at the reference frequency
    :param lm: Nsource x 2 array of sky coordinates for sources
    :param pqlist: a list of antennae pairs (used for the iterator)
    :param freqs: array of frequencies
    :param ref_freq: reference frequency
    :param gains: Na x Nnu x Nt x Nsource array containg direction dependent gains for antennaes
    :param DD: whether to applu DD gains or not
    :return: Xpq: Na x Na x Nnu x Nt array to hold model visibilities
    """
    ref_wavelength = speed_of_light/ref_freq
    for k, pq in enumerate(iter(pqlist)):
        p = pq[0]
        q = pq[1]
        for i in xrange(Nnu):
            wavelength = speed_of_light/freqs[i]
            for j in xrange(Nt):
                u = upq[k, j]*ref_wavelength/wavelength
                v = vpq[k, j]*ref_wavelength/wavelength  # convert units
                for s in xrange(Nsource):
                    l, m = lm[s]
                    complex_phase = -2.0*np.pi*(u*l + v*m)
                    K = np.cos(complex_phase) + 1.0j*np.sin(complex_phase)
                    Xpq_DD[p, q, i, j, s] = K * gains[p, i, j, s] * IM[i, s] * np.conj(gains[q, i, j, s])
                    Xpq_DD[q, p, i, j, s] = np.conj(Xpq_DD[p, q, i, j, s])
                    Xpq[p, q, i, j] += Xpq_DD[p, q, i, j, s]
                Xpq[q, p, i, j] = np.conj(Xpq[p, q, i, j])


    return Xpq, Xpq_DD

#@jit(nopython=True, nogil=True, cache=True)
def R_DD(IM, upq, vpq, lm, pqlist, freqs, ref_freq, gains, Xpq, Nnu, Nt, Nsource):
    """
    Full response operator including DDE's coded as a DFT.
    Note empty Xpq passed in for jitting purposes (don't want to be creating arrays inside a jitted function)
    :param IM: Nnu x Nsource array containing model image at Nnu freqs
    :param upq: N x Nt array of baseline coordinates in units of lambda at the reference frequency
    :param vpq: N x Nt array of baseline coordinates in units of lambda at the reference frequency
    :param lm: Nsource x 2 array of sky coordinates for sources
    :param pqlist: a list of antennae pairs (used for the iterator)
    :param freqs: array of frequencies
    :param ref_freq: reference frequency
    :param gains: Na x Nnu x Nt x Nsource array containg direction dependent gains for antennaes
    :param DD: whether to applu DD gains or not
    :return: Xpq: Na x Na x Nnu x Nt array to hold model visibilities
    """
    ref_wavelength = speed_of_light/ref_freq
    for k, pq in enumerate(iter(pqlist)):
        p = pq[0]
        q = pq[1]
        for i in xrange(Nnu):
            wavelength = speed_of_light/freqs[i]
            for j in xrange(Nt):
                u = upq[k, j]*ref_wavelength/wavelength
                v = vpq[k, j]*ref_wavelength/wavelength  # convert units
                for s in xrange(Nsource):
                    l, m = lm[s]
                    complex_phase = -2.0*np.pi*(u*l + v*m)
                    K = np.cos(complex_phase) + 1.0j*np.sin(complex_phase)
                    Xpq[p, q, i, j] += K * gains[p, i, j, s] * IM[i, s] * np.conj(gains[q, i, j, s])
                Xpq[q, p, i, j] = np.conj(Xpq[p, q, i, j])
    return Xpq

@jit(nopython=True, nogil=True, cache=True)
def apply_DD_gains(Xpq_DD, gains, Nnu, Nt, Nsource, pqlist, Xpq):
    """
    Full response operator including DDE's coded as a DFT.
    Note empty Xpq passed in for jitting purposes (don't want to be creating arrays inside a jitted function)
    :param IM: Nnu x Nsource array containing model image at Nnu freqs
    :param upq: N x Nt array of baseline coordinates in units of lambda at the reference frequency
    :param vpq: N x Nt array of baseline coordinates in units of lambda at the reference frequency
    :param lm: Nsource x 2 array of sky coordinates for sources
    :param pqlist: a list of antennae pairs (used for the iterator)
    :param freqs: array of frequencies
    :param ref_freq: reference frequency
    :param gains: Na x Nnu x Nt x Nsource array containg direction dependent gains for antennaes
    :param DD: whether to applu DD gains or not
    :return: Xpq: Na x Na x Nnu x Nt array to hold model visibilities
    """
    for k, pq in enumerate(iter(pqlist)):
        p = pq[0]
        q = pq[1]
        for i in xrange(Nnu):
            for j in xrange(Nt):
                for s in xrange(Nsource):
                    Xpq[p, q, i, j] += gains[p, i, j, s] * Xpq_DD[p, q, i, j, s] * np.conj(gains[q, i, j, s])
    return Xpq

# still work in progress
def RH(Xpq, Wpq, upq, vpq, lm, ID, pqlist, PSFmax=None):
    """
    The adjoint of the DFT response operator
    :param Xpq: Na x Na x Nt array containing model visibilities
    :param upq: Na x Nt array of baseline coordinates
    :param vpq: Na x Nt array of baseline coordinates
    :param lm: 2 x Npix**2 array of sky coordinates
    :param ID: Npix x Npix array to hold resulting image
    :return: 
    """
    ID_flat = ID.flatten()
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0]) - 1
        q = int(pq[1]) - 1
        uv = np.vstack((upq[i, :], vpq[i, :]))
        X = Xpq[p, q, :]*Wpq[p, q, :]
        K = np.exp(-2.0j * np.pi * np.dot(lm.T, uv))
        ID_flat += np.dot(K, X).real
        # if q != p:
        #     uv = np.vstack((-upq[i, :], -vpq[i, :]))
        #     X = Xpq[q, p, :]*Wpq[q, p, :]
        #     K = np.exp(-2.0j * np.pi * np.dot(lm.T, uv))
        #     ID_flat += np.dot(K, X).real
    ID = ID_flat.reshape(ID.shape)
    if PSFmax is not None:
        return ID/PSFmax
    else:
        return ID

def kron_tensorvec(A, b):
    """
    Tensor product over non-square Knonecker matrices
    :param A: an array of arrays holding matrices [..., K3, K2, K1] where Ki is Mi x Gi
    :param b: the RHS vector of length prod(G1, G2, ..., GD)
    :return: the solution vector alpha = Ab of length prod(M1, M2, ..., MD)
    """
    D = A.shape[0]
    # get shape of sub-matrices
    G = np.zeros(D, dtype=np.int8)
    M = np.zeros(D, dtype=np.int8)
    for d in xrange(D):
        M[d], G[d] = A[d].shape
    x = b
    for d in xrange(D):
        Gd = G[d]
        rem = np.prod(np.delete(G, d))
        X = np.reshape(x, (Gd, rem))
        Z = np.einsum("ab,bc->ac", A[d], X)
        Z = np.einsum("ab -> ba", Z)
        x = Z.flatten()
        # replace with new dimension
        G[d] = M[d]
    return x

def kron_matvec(A, b):
    """
    Computes matrix vector product of kronecker matrix in linear time. 
    :param A: an array of arrays holding matrices [..., K3, K2, K1] (note ordering)
    :param b: the RHS vector
    :return: A.dot(b)
    """
    D = A.shape[0]
    N = b.size
    x = b
    for d in xrange(D):
        Gd = A[d].shape[0]
        X = np.reshape(x,(Gd, N//Gd))
        Z = np.einsum("ab,bc->ac", A[d], X)
        Z = np.einsum("ab -> ba", Z)
        x = Z.flatten()
    return x

def kron_cholesky(A):
    """
    Computes the cholesky decomposition of a kronecker matrix
    :param A: an array of arrays holding matrices [K1, K2, K3, ...]
    :return: 
    """
    D = A.shape[0]
    L = np.zeros_like(A)
    for i in xrange(D):
        try:
            L[i] = np.linalg.cholesky(A[i])
        except: # add jitter
            L[i] = np.linalg.cholesky(A[i])
    return L

def kron_kron(A):
    """
    Computes the Kronecker product of a series of matrices or vectors 
    :param A: an array of arrays holding matrices/vectors [A1, A2, ..., AD]
    :return: the Kronecker product of all elements of A
    """
    D = A.shape[0]
    # expand the kronecker product for A
    K = np.kron(A[0], A[1])
    if D > 2:
        for i in xrange(2, D):
            K = np.kron(K, A[i])
    return K

def abs_diff(x, xp):
    """
    Vectorised absolute differences for covariance matrix computation
    :param x: 
    :param xp: 
    :return: 
    """
    return np.tile(x, (xp.size,1)).T - np.tile(xp, (x.size,1))

def sqexp(x, theta):
    return theta[0]**2*np.exp(-x**2/(2*theta[1])**2)

def draw_samples_ND_grid(x, theta, Nsamps, meanf=None):
    """
    Draw N dimensional samples on a Euclidean grid
    :param meanf: mean function
    :param x: array of arrays containing targets [x_1, x_2, ..., x_D]
    :param theta: array of arrays containing [theta_1, theta_2, ..., theta_D]
    :param Nsamps: number of samples to draw
    :return: array containing samples [Nsamps, N_1, N_2, ..., N_D]
    """

    D = x.shape[0]
    Ns = []
    K = np.empty(D, dtype=object)
    Ntot=1
    for i in xrange(D):
        Ns.append(x[i].size)
        XX = abs_diff(x[i], x[i])
        K[i] = sqexp(XX, theta[i]) + 1e-13*np.eye(Ns[i])
        Ntot *= Ns[i]

    L = kron_cholesky(K)
    samps = np.zeros([Nsamps]+Ns, dtype=np.complex128)
    for i in xrange(Nsamps):
        xi = np.random.randn(Ntot) + 1.0j*np.random.randn(Ntot)
        if meanf is not None:
            samps[i] = meanf(x) + kron_matvec(L, xi).reshape(Ns)
        else:
            samps[i] = kron_matvec(L, xi).reshape(Ns)
    return samps, K

# def plot_vis(Xpq, Xpq_corrected, Xpq_corrected2, upq, vpq, p, q):
#     # plot absolute value of visibilities as function of baseline length
#     plt.figure('visabs')
#     plt.plot(np.abs(upq[p,:] - vpq[q,:]), np.abs(Xpq[p,q,:]), 'k+', label='True vis')
#     plt.plot(np.abs(upq[p, :] - vpq[q, :]), np.abs(Xpq_corrected[p, q, :]), 'b+', label='Corrected vis 1')
#     plt.plot(np.abs(upq[p, :] - vpq[q, :]), np.abs(Xpq_corrected2[p, q, :]), 'g+', label='Corrected vis 2')
#     plt.savefig('/home/landman/Projects/SmoothCal/figures/abs_vis_compare.png', dpi=250)
#     # plot phase of visibilities as function of baseline length
#     plt.figure('visphase')
#     plt.plot(np.abs(upq[p,:] - vpq[q,:]), np.arctan(Xpq[p,q,:].imag/Xpq[p,q,:].real), 'k+', label='True vis')
#     plt.plot(np.abs(upq[p, :] - vpq[q, :]), np.arctan(Xpq_corrected[p,q,:].imag/Xpq_corrected[p,q,:].real), 'b+', label='Corrected vis 1')
#     plt.plot(np.abs(upq[p, :] - vpq[q, :]), np.arctan(Xpq_corrected2[p,q,:].imag/Xpq_corrected[p,q,:].real), 'g+', label='Corrected vis 2')
#     plt.savefig('/home/landman/Projects/SmoothCal/figures/phase_vis_compare.png', dpi=250)
#     return
#
# def plot_fits(IM, IR, ID, name):
#     # save images to fits
#     hdu = fits.PrimaryHDU(ID)
#     hdul = fits.HDUList([hdu])
#     hdul.writeto('/home/landman/Projects/SmoothCal/figures/ID_' + name + '.fits', overwrite=True)
#     hdul.close()
#
#     hdu = fits.PrimaryHDU(IM)
#     hdul = fits.HDUList([hdu])
#     hdul.writeto('/home/landman/Projects/SmoothCal/figures/IM_' + name + '.fits', overwrite=True)
#     hdul.close()
#
#     hdu = fits.PrimaryHDU(IR)
#     hdul = fits.HDUList([hdu])
#     hdul.writeto('/home/landman/Projects/SmoothCal/figures/IR_' + name + '.fits', overwrite=True)
#     hdul.close()
#     return
#
# def plot_gains(tfull, gfull_true, Sigmay_full, gbar_full, gbar_stef_full, pqlist):
#     for i, pq in  enumerate(iter(pqlist)):
#         p = int(pq[0])-1
#         q = int(pq[1])-1
#
#         fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))
#         ax[0].fill_between(tfull, (gfull_true[p]*gfull_true[q].conj()).real + np.sqrt(1.0/Sigmay_full[p] + 1.0/Sigmay_full[q])/np.sqrt(2),
#                            (gfull_true[p] * gfull_true[q].conj()).real - np.sqrt(1.0 / Sigmay_full[p] + 1.0 / Sigmay_full[q])/np.sqrt(2),
#                            facecolor='b', alpha=0.25)
#         ax[0].plot(tfull, (gfull_true[p]*gfull_true[q].conj()).real, 'k', label='True')
#         ax[0].plot(tfull, (gbar_full[p]*gbar_full[q].conj()).real, 'b--', alpha=0.5, label='SmoothCal')
#         ax[0].plot(tfull, (gbar_stef_full[p,:]*gbar_stef_full[q, :].conj()).real, 'g--', alpha=0.5, label='StefCal')
#         #ax[0].errorbar(tfull, (gfull_true[0]*gfull_true[1].conj()).real, np.sqrt(1.0/Sigmay_full[0] + 1.0/Sigmay_full[1]), fmt='xr', alpha=0.25)
#         ax[0].set_xlabel(r'$t$', fontsize=18)
#         ax[0].set_ylabel(r'$Real(g_p g_q^\dagger)$', fontsize=18)
#         #ax[0].legend()
#
#         ax[1].fill_between(tfull, (gfull_true[p] * gfull_true[q].conj()).imag + np.sqrt(1.0 / Sigmay_full[p] + 1.0 / Sigmay_full[q])/np.sqrt(2),
#                            (gfull_true[p] * gfull_true[q].conj()).imag - np.sqrt(1.0 / Sigmay_full[p] + 1.0 / Sigmay_full[q])/np.sqrt(2),
#                            facecolor='b', alpha=0.25)
#         ax[1].plot(tfull, (gfull_true[p] * gfull_true[q].conj()).imag, 'k', label='True')
#         ax[1].plot(tfull, (gbar_full[p] * gbar_full[q].conj()).imag, 'b--', alpha=0.5, label='SmoothCal')
#         ax[1].plot(tfull, (gbar_stef_full[p, :] * gbar_stef_full[q, :].conj()).imag, 'g--', alpha=0.5, label='StefCal')
#         #ax[1].errorbar(tfull, (gfull_true[0] * gfull_true[1].conj()).imag, np.sqrt(1.0/Sigmay_full[0] + 1.0/Sigmay_full[1]), fmt='xr', alpha=0.25)
#         ax[1].set_xlabel(r'$t$', fontsize=18)
#         ax[1].set_ylabel(r'$Imag(g_p g_q^\dagger)$', fontsize=18)
#         ax[1].legend(loc=2)
#
#         fig.savefig('/home/landman/Projects/SmoothCal/figures/Full_sim_combined'+str(p)+str(q) +'.png', dpi = 250)
#
#         # plot errors
#         plt.figure('error2')
#         plt.plot(tfull, np.abs(gfull_true[p] * gfull_true[q].conj() - gbar_full[p] * gbar_full[q].conj()), 'k.', label='SmoothCal')
#         plt.plot(tfull, np.abs(gfull_true[p, :] * gfull_true[q, :].conj() - gbar_stef_full[p, :] * gbar_stef_full[q, :].conj()), 'g--', label='StefCal')
#         plt.fill_between(tfull, np.sqrt(np.diag(Dlist_full[p].val).real + np.diag(Dlist_full[q].val).real), np.zeros(Nfull), facecolor='b', alpha=0.5)
#         plt.xlabel(r'$t$', fontsize=18)
#         plt.ylabel(r'$|\epsilon|$', fontsize=18)
#         plt.legend()
#         plt.savefig('/home/landman/Projects/SmoothCal/figures/Sim_error_combined'+str(p)+str(q) +'.png', dpi = 250)
#
#         #plt.show()
#         plt.close('all')
#     return
#
# def apply_gains(Vpq, g, pqlist, Nt, Xpq):
#     for i, pq in enumerate(iter(pqlist)):
#         p = int(pq[0])-1
#         q = int(pq[1])-1
#         gptemp = g[p]
#         gqtempH = g[q].conj()
#         for j in xrange(Nt):
#             Xpq[p, q, j] = Vpq[p, q, j]/(gptemp[j]*gqtempH[j])
#             Xpq[q, p, j] = Xpq[p, q, j].conj()
#     return Xpq
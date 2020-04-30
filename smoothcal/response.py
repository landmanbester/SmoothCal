import numpy as np
from numba import jit, prange

@jit(fastmath=True)
def param2vis(time_bin_indices, time_bin_counts, antenna1, antenna2,
              g0a, g0p, g1a, g1p, 
              k0, k1, l0, l1, 
              b00a, b00p, b01a, b01p, b10a, b10p, b11a, b11p,
              c, d, phi, R00, R01, R10, R11,
              I, Q, U, V, kappa, freq):

    time_bin_indices -= time_bin_indices.min()  # for later dask chunking capability
    ntime, nant = g0a.shape
    nchan = freq.size
    nbl = nant*(nant-1)//2
    nrow = ntime*nbl
    Vpq = np.zeros((nrow, nchan, 2, 2), dtype=np.complex128)
    for t in range(ntime):
        for row in range(time_bin_indices[t],
                         time_bin_indices[t] + time_bin_counts[t]):
            p = int(antenna1[row])
            q = int(antenna2[row])
            
            phip = phi[t, p]
            phiq = phi[t, q]
            
            gp0a = g0a[t, p]
            gp0p = g0p[t, p]
            gq0a = g0a[t, q]
            gq0p = g0p[t, q]

            gp1a = g1a[t, p]
            gp1p = g1p[t, p]
            gq1a = g1a[t, q]
            gq1p = g1p[t, q]

            kp0 = k0[t, p]
            kq0 = k0[t, q]

            kp1 = k1[t, p]
            kq1 = k1[t, q]

            lp0 = l0[t, p]
            lq0 = l0[t, q]

            lp1 = l1[t, p]
            lq1 = l1[t, q]

            cp = c[t, p]
            cq = c[t, q]

            dp = d[t, p]
            dq = d[t, q]
            
            for nu in range(nchan):
                bp00a = b00a[p, nu]
                bp00p = b00p[p, nu]
                bq00a = b00a[q, nu]
                bq00p = b00p[q, nu]

                bp01a = b01a[p, nu]
                bp01p = b01p[p, nu]
                bq01a = b01a[q, nu]
                bq01p = b01p[q, nu]

                bp10a = b10a[p, nu]
                bp10p = b10p[p, nu]
                bq10a = b10a[q, nu]
                bq10p = b10p[q, nu]

                bp11a = b11a[p, nu]
                bp11p = b11p[p, nu]
                bq11a = b11a[q, nu]
                bq11p = b11p[q, nu]

                Vpq[row, nu, 0, 0] = R00(gp0a, gp0p, gp1a, gp1p, gq0a, gq0p, gq1a, gq1p,
                                         kp0, kp1, lp0, lp1, kq0, kq1, lq0, lq1,
                                         bp00a, bp00p, bp01a, bp01p, bp10a, bp10p, bp11a, bp11p,
                                         bq00a, bq00p, bq01a, bq01p, bq10a, bq10p, bq11a, bq11p,
                                         cp, dp, cq, dq, phip, phiq,
                                         I, Q, U, V, kappa, freq[nu])

                Vpq[row, nu, 0, 1] = R01(gp0a, gp0p, gp1a, gp1p, gq0a, gq0p, gq1a, gq1p,
                                         kp0, kp1, lp0, lp1, kq0, kq1, lq0, lq1,
                                         bp00a, bp00p, bp01a, bp01p, bp10a, bp10p, bp11a, bp11p,
                                         bq00a, bq00p, bq01a, bq01p, bq10a, bq10p, bq11a, bq11p,
                                         cp, dp, cq, dq, phip, phiq,
                                         I, Q, U, V, kappa, freq[nu])

                Vpq[row, nu, 1, 0] = R10(gp0a, gp0p, gp1a, gp1p, gq0a, gq0p, gq1a, gq1p,
                                         kp0, kp1, lp0, lp1, kq0, kq1, lq0, lq1,
                                         bp00a, bp00p, bp01a, bp01p, bp10a, bp10p, bp11a, bp11p,
                                         bq00a, bq00p, bq01a, bq01p, bq10a, bq10p, bq11a, bq11p,
                                         cp, dp, cq, dq, phip, phiq,
                                         I, Q, U, V, kappa, freq[nu])

                Vpq[row, nu, 1, 1] = R11(gp0a, gp0p, gp1a, gp1p, gq0a, gq0p, gq1a, gq1p,
                                         kp0, kp1, lp0, lp1, kq0, kq1, lq0, lq1,
                                         bp00a, bp00p, bp01a, bp01p, bp10a, bp10p, bp11a, bp11p,
                                         bq00a, bq00p, bq01a, bq01p, bq10a, bq10p, bq11a, bq11p,
                                         cp, dp, cq, dq, phip, phiq,
                                         I, Q, U, V, kappa, freq[nu])

    return Vpq


# @jit(fastmath=True, cache=True)
def jones2vis(time_bin_indices, time_bin_counts, antenna1, antenna2,
              G, K, B, D, P, Bs):
    """
    Applies gains to model evlautaed at (l,m) = (0,0).
    Note all gains have the same number of axes which are in the order
        
        (time, ant, chan, corr, corr)
    
    We assume that the model visibilities correspond to a point source at the phase
    center so source coherency is always unity. 
    
    Bs is the 2x2 source brightness matrix.

    P is the feed rotation matrix (paralactic angle rotation) which has shape

        (time, ant, corr, corr)

    This function is only for testing and verification.
    """
    time_bin_indices -= time_bin_indices.min()  # for later dask chunking capability
    ntime, nant, nchan, _, _ = B.shape
    nbl = nant*(nant-1)//2
    nrow = ntime*nbl
    Vpq = np.zeros((nrow, nchan, 2, 2), dtype=np.complex128)
    for t in range(ntime):
        for row in range(time_bin_indices[t],
                         time_bin_indices[t] + time_bin_counts[t]):
            p = int(antenna1[row])
            q = int(antenna2[row])
            Pp = P[t, p]
            Pq = P[t, q]
            for nu in range(nchan):
                Ip = (t, p, nu)
                Iq = (t, q, nu)
                Vpq[row, nu] = G[Ip] @ K[Ip] @ B[Ip] @ D[Ip] @ Pp @ Bs[nu] @ Pq.conj().T @ D[Iq].conj().T @ B[Iq].conj().T @ K[Iq].conj().T @ G[Iq].conj().T
    return Vpq

def jacobian(time_bin_indices, time_bin_counts, antenna1, antenna2, freq,
             R00, R01, R10, R11, dR00, dR01, dR10, dR11,
             params, solparams, chain, mode='FULL'):
    time_bin_indices -= time_bin_indices.min()  # for later dask chunking capability
    ntime  = time_bin_indices.size
    nant = np.maximum(antenna1.max(), antenna2.max())
    nrow = antenna1.size
    nchan = freq.size 
    if mode == "DIAG":
        corrs = ['00', '11']
    elif mode == 'FULL':
        corr = ['00', '01', '10', '11']
    else:
        raise ValueError('Unrecognised mode %s'%mode)
    
    J = np.zeros((nrow, nchan, len(corrs), ntime, nchan, len(solparams)), dtype=np.complex128)
    for t in range(ntime):
        for row in range(time_bin_indices[t],
                         time_bin_indices[t] + time_bin_counts[t]):
            p = int(antenna1[row])
            q = int(antenna2[row])
            Pp = P[t, p]
            Pq = P[t, q]
            for chan in range(nchan):
                for icorr, corr in enumerate(corrs):
                    for iparam, param in enumerate(solparams):
                        J[row, chan, icorr, t, chan, iparam] = RIME[corr]['deriv'][iparam](params)



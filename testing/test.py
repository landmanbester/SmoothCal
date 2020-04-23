import numpy as np
from numpy.testing import assert_array_almost_equal
from pyrap.tables import table
from daskms import xds_from_ms, xds_from_table
from africanus.calibration.utils import chunkify_rows
from africanus.rime import parallactic_angles, feed_rotation
from smoothcal.utils import define_fields, params2jones, field2param, symbolic_jones_chain

def param_vs_jones(time_bin_indices, time_bin_counts, antenna1, antenna2,
                   g0a, g0p, g1a, g1p, 
                   k0, k1, l0, l1, 
                   b00a, b00p, b01a, b01p, b10a, b10p, b11a, b11p,
                   c, d, phi, R00, R01, R10, R11,
                   I, Q, U, V, kappa, freq,
                   G, K, B, D, P, Bs):

    time_bin_indices -= time_bin_indices.min()  # for later dask chunking capability
    ntime, nant = g0a.shape
    nchan = freq.size
    nbl = nant*(nant-1)//2
    nrow = ntime*nbl
    for t in range(ntime):
        for row in range(time_bin_indices[t],
                         time_bin_indices[t] + time_bin_counts[t]):
            q = int(antenna1[row])
            p = int(antenna2[row])
            phip = phi[t, p]
            phiq = phi[t, q]

            # P
            Pp = P[t, p]
            assert P[t, p, 0, 0] == np.exp(-1j*phip)
            assert P[t, p, 0, 1] == 0.0
            assert P[t, p, 1, 0] == 0.0
            assert P[t, p, 1, 1] == np.exp(1j*phip)

            Pq = P[t, q]
            assert P[t, q, 0, 0] == np.exp(-1j*phiq)
            assert P[t, q, 0, 1] == 0.0
            assert P[t, q, 1, 0] == 0.0
            assert P[t, q, 1, 1] == np.exp(1j*phiq)
            
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

                # G
                assert G[t, p, nu, 0, 0] == np.exp(gp0a + 1j*gp0p)
                assert G[t, p, nu, 0, 1] == 0.0
                assert G[t, p, nu, 1, 0] == 0.0
                assert G[t, p, nu, 1, 1] == np.exp(gp1a + 1j*gp1p)

                assert G[t, q, nu, 0, 0] == np.exp(gq0a + 1j*gq0p)
                assert G[t, q, nu, 0, 1] == 0.0
                assert G[t, q, nu, 1, 0] == 0.0
                assert G[t, q, nu, 1, 1] == np.exp(gq1a + 1j*gq1p)

                # K
                assert K[t, p, nu, 0, 0] == np.exp(1j*(kp0*freq[nu] + lp0))
                assert K[t, p, nu, 0, 1] == 0.0
                assert K[t, p, nu, 1, 0] == 0.0
                assert K[t, p, nu, 1, 1] == np.exp(1j*(kp1*freq[nu] + lp1))

                assert K[t, q, nu, 0, 0] == np.exp(1j*(kq0*freq[nu] + lq0))
                assert K[t, q, nu, 0, 1] == 0.0
                assert K[t, q, nu, 1, 0] == 0.0
                assert K[t, q, nu, 1, 1] == np.exp(1j*(kq1*freq[nu] + lq1))

                # B
                assert B[t, p, nu, 0, 0] == np.exp(bp00a + 1j*bp00p)
                assert B[t, p, nu, 0, 1] == np.exp(bp01a + 1j*bp01p)
                assert B[t, p, nu, 1, 0] == np.exp(bp10a + 1j*bp10p)
                assert B[t, p, nu, 1, 1] == np.exp(bp11a + 1j*bp11p)

                assert B[t, q, nu, 0, 0] == np.exp(bq00a + 1j*bq00p)
                assert B[t, q, nu, 0, 1] == np.exp(bq01a + 1j*bq01p)
                assert B[t, q, nu, 1, 0] == np.exp(bq10a + 1j*bq10p)
                assert B[t, q, nu, 1, 1] == np.exp(bq11a + 1j*bq11p)

                # D
                assert D[t, p, nu, 0, 0] == np.exp(1j*(cp*freq[nu] + dp))
                assert D[t, p, nu, 0, 1] == 0.0
                assert D[t, p, nu, 1, 0] == 0.0
                assert D[t, p, nu, 1, 1] == np.exp(1j*(cp*freq[nu] + dp))

                assert D[t, q, nu, 0, 0] == np.exp(1j*(cq*freq[nu] + dq))
                assert D[t, q, nu, 0, 1] == 0.0
                assert D[t, q, nu, 1, 0] == 0.0
                assert D[t, q, nu, 1, 1] == np.exp(1j*(cq*freq[nu] + dq))

                # Bs
                assert Bs[nu, 0, 0] == I+V
                assert Bs[nu, 0, 1] == np.exp(2j*freq[nu]**2*kappa)*(Q + 1j*U)
                assert Bs[nu, 1, 0] == np.exp(2j*freq[nu]**2*kappa)*(Q - 1j*U)
                assert Bs[nu, 1, 1] == I-V

                Ip = (t, p, nu)
                Iq = (t, q, nu)
                VJones = G[Ip] @ K[Ip] @ B[Ip] @ D[Ip] @ Pp @ Bs[nu] @ Pq.conj().T @ D[Iq].conj().T @ B[Iq].conj().T @ K[Iq].conj().T @ G[Iq].conj().T

                Vpars00 = R00(gp0a, gp0p, gp1a, gp1p, gq0a, gq0p, gq1a, gq1p,
                                kp0, kp1, lp0, lp1, kq0, kq1, lq0, lq1,
                                bp00a, bp00p, bp01a, bp01p, bp10a, bp10p, bp11a, bp11p,
                                bq00a, bq00p, bq01a, bq01p, bq10a, bq10p, bq11a, bq11p,
                                cp, dp, cq, dq, phip, phiq,
                                I, Q, U, V, kappa, freq[nu])

                assert_array_almost_equal(VJones[0, 0], Vpars00, decimal=10)

                Vpars01 = R01(gp0a, gp0p, gp1a, gp1p, gq0a, gq0p, gq1a, gq1p,
                                         kp0, kp1, lp0, lp1, kq0, kq1, lq0, lq1,
                                         bp00a, bp00p, bp01a, bp01p, bp10a, bp10p, bp11a, bp11p,
                                         bq00a, bq00p, bq01a, bq01p, bq10a, bq10p, bq11a, bq11p,
                                         cp, dp, cq, dq, phip, phiq,
                                         I, Q, U, V, kappa, freq[nu])

                assert_array_almost_equal(VJones[0, 1], Vpars01, decimal=10)

                Vpars10 = R10(gp0a, gp0p, gp1a, gp1p, gq0a, gq0p, gq1a, gq1p,
                                         kp0, kp1, lp0, lp1, kq0, kq1, lq0, lq1,
                                         bp00a, bp00p, bp01a, bp01p, bp10a, bp10p, bp11a, bp11p,
                                         bq00a, bq00p, bq01a, bq01p, bq10a, bq10p, bq11a, bq11p,
                                         cp, dp, cq, dq, phip, phiq,
                                         I, Q, U, V, kappa, freq[nu])

                assert_array_almost_equal(VJones[1, 0], Vpars10, decimal=10)

                Vpars11 = R11(gp0a, gp0p, gp1a, gp1p, gq0a, gq0p, gq1a, gq1p,
                                         kp0, kp1, lp0, lp1, kq0, kq1, lq0, lq1,
                                         bp00a, bp00p, bp01a, bp01p, bp10a, bp10p, bp11a, bp11p,
                                         bq00a, bq00p, bq01a, bq01p, bq10a, bq10p, bq11a, bq11p,
                                         cp, dp, cq, dq, phip, phiq,
                                         I, Q, U, V, kappa, freq[nu])

                assert_array_almost_equal(VJones[1, 1], Vpars11, decimal=10)
    return

if __name__=="__main__":
    # get time mapping
    time = table('/home/landman/Data/SmoothCalTests/VLA_30stamps_60s_32chan_20Mhz.MS_p0').getcol('TIME')
    utimes = np.unique(time)
    utimes_per_chunk = utimes.size  # no chunking at this stage
    row_chunks, tbin_idx, tbin_counts = chunkify_rows(time, utimes_per_chunk)
    
    # chunkify time mapping
    xds = xds_from_ms('/home/landman/Data/SmoothCalTests/VLA_30stamps_60s_32chan_20Mhz.MS_p0', 
                      columns=('TIME', 'ANTENNA1', 'ANTENNA2'), 
                      chunks={"row": row_chunks})[0]
    
    ant1 = xds.ANTENNA1.data.compute()
    ant2 = xds.ANTENNA2.data.compute()

    freq = xds_from_table('/home/landman/Data/SmoothCalTests/VLA_30stamps_60s_32chan_20Mhz.MS_p0' + '::SPECTRAL_WINDOW')[0].CHAN_FREQ.data.compute().squeeze()

    freq /= np.mean(freq)

    ant_pos = xds_from_table('/home/landman/Data/SmoothCalTests/VLA_30stamps_60s_32chan_20Mhz.MS_p0' + '::ANTENNA')[0].POSITION.data.compute().squeeze()
    phase_dir = xds_from_table('/home/landman/Data/SmoothCalTests/VLA_30stamps_60s_32chan_20Mhz.MS_p0' + '::FIELD')[0].PHASE_DIR.data.compute().squeeze()

    # compute paralactic angles and feed rotation
    parangles = parallactic_angles(utimes, ant_pos, phase_dir)
    nant = parangles.shape[-1]
    P = feed_rotation(parangles, feed_type='circular')

    np.random.seed(420)
    xi = define_fields(utimes, freq, nant)
    I = 1.0
    Q = 0.1
    U = 0.1
    V = 0.01
    kappa = 0.1

    # get Jones matrices
    G, K, B, D, Bs = params2jones(xi, freq, I, Q, U, V, kappa)

    g0a = field2param(xi['g0a'])
    g0p = field2param(xi['g0p'])
    g1a = field2param(xi['g1a'])
    g1p = field2param(xi['g1p'])

    b00a = np.ascontiguousarray(field2param(xi['b00a']).T)
    b00p = np.ascontiguousarray(field2param(xi['b00p']).T)
    b01a = np.ascontiguousarray(field2param(xi['b01a']).T)
    b01p = np.ascontiguousarray(field2param(xi['b01p']).T)
    b10a = np.ascontiguousarray(field2param(xi['b10a']).T)
    b10p = np.ascontiguousarray(field2param(xi['b10p']).T)
    b11a = np.ascontiguousarray(field2param(xi['b11a']).T)
    b11p = np.ascontiguousarray(field2param(xi['b11p']).T)

    k0 = field2param(xi['k0'])
    k1 = field2param(xi['k1'])
    l0 = field2param(xi['l0'])
    l1 = field2param(xi['l1'])

    c = field2param(xi['c'])
    d = field2param(xi['d'])

    R00, R01, R10, R11 = symbolic_jones_chain(Print=False)

    param_vs_jones(tbin_idx, tbin_counts, ant1, ant2,
                    g0a, g0p, g1a, g1p, 
                    k0, k1, l0, l1, 
                    b00a, b00p, b01a, b01p, b10a, b10p, b11a, b11p,
                    c, d, parangles, R00, R01, R10, R11,
                    I, Q, U, V, kappa, freq,
                    G, K, B, D, P, Bs)
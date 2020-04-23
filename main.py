import numpy as np
import argparse
from pyrap.tables import table
from daskms import xds_from_ms, xds_from_table
from africanus.calibration.utils import chunkify_rows
from africanus.rime import parallactic_angles, feed_rotation
from smoothcal.utils import define_fields, params2jones, field2param, symbolic_jones_chain
from smoothcal.response import jones2vis, param2vis

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=str, nargs='+')
    p.add_argument("--data_column", default="CORRECTED_DATA", type=str,
                   help="The column to image.")
    p.add_argument("--model_column", default="MODEL_DATA", type=str,
                   help="Column to write model visibilities to.")
    p.add_argument("--weight_column", default='WEIGHT_SPECTRUM', type=str,
                   help="Weight column to use. Will use WEIGHT if no WEIGHT_SPECTRUM is found")
    p.add_argument("--fid", type=int, default=0,
                   help="Field ID")
    p.add_argument("--ddid", type=int, default=0,
                   help="Spectral window")
    return p

def main(args):
    # get time mapping
    time = table(args.ms).getcol('TIME')
    utimes = np.unique(time)
    utimes_per_chunk = utimes.size  # no chunking at this stage
    row_chunks, tbin_idx, tbin_counts = chunkify_rows(time, utimes_per_chunk)
    
    # chunkify time mapping
    xds = xds_from_ms(args.ms, 
                      columns=('TIME', 'ANTENNA1', 'ANTENNA2'), 
                      chunks={"row": row_chunks})[args.fid]
    
    ant1 = xds.ANTENNA1.data.compute()
    ant2 = xds.ANTENNA2.data.compute()
    
    # data = getattr(xds, args.data_column).data
    # model = getattr(xds, args.model_column).data
    # weight = getattr(xds, args.weight_column).data

    freq = xds_from_table(args.ms + '::SPECTRAL_WINDOW')[args.ddid].CHAN_FREQ.data.compute().squeeze()

    freq /= np.mean(freq)

    # get feed rotation matrix
    ant_pos = xds_from_table(args.ms + '::ANTENNA')[0].POSITION.data.compute().squeeze()
    phase_dir = xds_from_table(args.ms + '::FIELD')[args.fid].PHASE_DIR.data.compute().squeeze()

    # compute paralactic angles and feed rotation
    parangles = parallactic_angles(utimes, ant_pos, phase_dir)
    nant = parangles.shape[-1]
    P = feed_rotation(parangles, feed_type='circular')

    # create white fields
    np.random.seed(420)
    xi = define_fields(utimes, freq, nant)
    I = 1.0
    Q = 0.0
    U = 0.0
    V = 0.0
    kappa = 0.0

    # get Jones matrices
    G, K, B, D, Bs = params2jones(xi, freq, I, Q, U, V, kappa)

    from numpy.testing import assert_array_almost_equal

    from time import time

    # get vis
    print("jones2vis")
    ti = time()
    Vpq1 = jones2vis(tbin_idx, tbin_counts, ant1, ant2,
                     G, K, B, D, P, Bs)
    print(time() - ti)

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

    print("param2vis")
    ti = time()
    Vpq2 = param2vis(tbin_idx, tbin_counts, ant1, ant2,
                     g0a, g0p, g1a, g1p, 
                     k0, k1, l0, l1, 
                     b00a, b00p, b01a, b01p, b10a, b10p, b11a, b11p,
                     c, d, parangles, R00, R01, R10, R11,
                     I, Q, U, V, kappa, freq)
    print(time()-ti)

    from numpy.testing import assert_array_almost_equal
    assert_array_almost_equal(Vpq1, Vpq2, decimal=5)


if __name__=="__main__":
    args = create_parser().parse_args()

    args.ms = args.ms[0]

    main(args)
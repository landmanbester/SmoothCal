import numpy as np
import argparse
from pyrap.tables import table
from daskms import xds_from_ms, xds_from_table, xds_to_table, Dataset
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
    # set mock model
    I = 1.0
    Q = 0.1
    U = 0.1
    V = 0.01
    kappa = 0.25

    # determine polarisation type
    poltbl = table(args.ms + '::POLARIZATION')
    corr_type_set = set(poltbl.getcol('CORR_TYPE').squeeze())
    if corr_type_set.issubset(set([9, 10, 11, 12])):
        pol_type = 'linear'
    elif corr_type_set.issubset(set([5, 6, 7, 8])):
        pol_type = 'circular'
    else:
        raise ValueError("Cannot determine polarisation type "
                        "from correlations %s. Constructing "
                        "a feed rotation matrix will not be "
                        "possible." % (corr_type_set,))
    poltbl.close()

    # get symbolic jones chain
    jonesterms = 'GKBPD'
    RIME, params, solparams = symbolic_jones_chain(jonesterms, pol_type)

    # get time mapping
    time = table(args.ms).getcol('TIME')
    utimes = np.unique(time)
    utimes_per_chunk = utimes.size  # no chunking at this stage
    row_chunks, tbin_idx, tbin_counts = chunkify_rows(time, utimes_per_chunk)
    
    # chunkify time mapping
    xds = xds_from_ms(args.ms, 
                      columns=('TIME', 'ANTENNA1', 'ANTENNA2', 
                      chunks={"row": -1})[args.fid]  # not chunking for the time being
    
    # subtables
    ddids = xds_from_table((args.ms + "::DATA_DESCRIPTION")
    fields = xds_from_table((args.ms + "::FIELD", group_cols="__row__")
    spws = xds_from_table((args.ms + "::SPECTRAL_WINDOW", group_cols="__row__")
    pols = xds_from_table((args.ms + "::POLARIZATION", group_cols="__row__")
    posn = xds_from_table(args.ms + '::ANTENNA', group_cols="__row__")

    # Get subtable data
    ddids = dask.compute(ddids)[0]
    fields = dask.compute(fields)[0]
    spws = dask.compute(spws)[0]
    pols = dask.compute(pols)[0]
    posn = dask.compute(posns)[0]

    # to store write list
    datasets = []
    for ds in xds:
        if ds.FIELD_ID not in args.field:
            continue

        if ds.DATA_DESC_ID not in args.ddid:
            continue

        field = fields[ds.FIELD_ID]

        phase_dir = field.PHASE_DIR.data.compute().squeeze()

        ddid = ddids[ds.DATA_DESC_ID]

        pol = pols[ddid.POLARIZATION_ID.values[0]]
        corr_type_set = set(pol.CORR_TYPE.data.compute().squeeze())
        if corr_type_set.issubset(set([9, 10, 11, 12])):
            pol_type = 'linear'
        elif corr_type_set.issubset(set([5, 6, 7, 8])):
            pol_type = 'circular'
        else:
            raise ValueError("Cannot determine polarisation type "
                            "from correlations %s. Constructing "
                            "a feed rotation matrix will not be "
                            "possible." % (corr_type_set,))
        
        freq = spws[ddid.SPECTRAL_WINDOW_ID.values[0]].CHAN_FREQ.data.compute()
        freq /= np.mean(freq)

        ant_pos = posn.POSITION.data.compute()

        parangles = parallactic_angles(utimes, ant_pos, phase_dir)
        nant = parangles.shape[-1]
        

        ant1 = ds.ANTENNA1.data.compute()
        ant2 = ds.ANTENNA2.data.compute()

        xi = define_fields(utimes, freq, nant)  # note internal freq and time scaling

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

        # compute visibilities
        Vpq = param2vis(tbin_idx, tbin_counts, ant1, ant2,
                        g0a, g0p, g1a, g1p, 
                        k0, k1, l0, l1, 
                        b00a, b00p, b01a, b01p, b10a, b10p, b11a, b11p,
                        c, d, parangles, R00, R01, R10, R11,
                        I, Q, U, V, kappa, freq)

        # add noise
        noise = np.random.randn(Vpq.shape)/np.sqrt(2) + 1.0j*np.random.randn(Vpq.shape)/np.sqrt(2)
        data = Vpq + noise

        data_vars = {
                'FIELD_ID':(('row',), da.full_like(ds.TIME.data, field_id)),
                'DATA_DESC_ID':(('row',), da.full_like(ds.TIME.data, ddid_id)),
                args.data_out_column:(('row', 'chan'), data),
                args.weight_out_column:(('row', 'chan'), da.full_like(data, 1.0))
            }

        out_ds = Dataset(data_vars)

        out_datasets.append(out_ds)
    
    writes = xds_to_table(out_datasets, args.ms, columns="ALL")
    
    dask.compute(writes)


if __name__=="__main__":
    args = create_parser().parse_args()

    args.ms = args.ms[0]

    main(args)
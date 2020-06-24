import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from smoothcal.utils import symbolic_jones_chain, define_field_dct, domain2param_mapping
from smoothcal.response import jacobian
from africanus.calibration.utils import chunkify_rows
from pyrap.tables import table
import pickle
from time import time as timeit
from scipy.sparse import csr_matrix

if __name__=="__main__":
    # data source
    ms_name = '/home/landman/Data/SmoothCalTests/VLA_3stamps_60s_8chan_20Mhz.MS_p0'
    
    # determine polarisation type
    poltbl = table(ms_name + '::POLARIZATION')
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
    
    # construct symbolic Jones chain
    joness = 'GKDP'
    solvables = '1110'
    cache_dir = '/home/landman/Data/SmoothCalTests/'
    try:
        with open(cache_dir+joness+'_'+solvables+'.pickle', 'rb') as f:
            RIME = pickle.load(f) 
        
        R00 = RIME['R00']
        R01 = RIME['R01']
        R10 = RIME['R10']
        R11 = RIME['R11']
        dR00 = RIME['dR00']
        dR01 = RIME['dR01']
        dR10 = RIME['dR10']
        dR11 = RIME['dR11']
        solvable_names = RIME['solvable_names']
        field_names = RIME['field_names']
        field_inds = RIME['field_inds']
        
    except:
        R00, R01, R10, R11, dR00, dR01, dR10, dR11, solvable_names, field_names, field_inds = symbolic_jones_chain(joness=joness, solvables=solvables, poltype=pol_type)

        RIME = {}
        RIME['R00'] = R00
        RIME['R01'] = R01
        RIME['R10'] = R10
        RIME['R11'] = R11
        RIME['dR00'] = dR00
        RIME['dR01'] = dR01
        RIME['dR10'] = dR10
        RIME['dR11'] = dR11
        RIME['solvable_names'] = solvable_names
        RIME['field_names'] = field_names
        RIME['field_inds'] = field_inds

        with open(cache_dir+joness+'_'+solvables+'.pickle', 'wb') as f:
            pickle.dump(RIME, f)

    # get ms info
    ms = table(ms_name)
    time = ms.getcol('TIME')
    utimes = np.unique(time)
    utimes_per_chunk = utimes.size  # no chunking at this stage
    row_chunks, tbin_idx, tbin_counts = chunkify_rows(time, utimes_per_chunk)
    ntime = utimes.size
    
    antenna1 = ms.getcol('ANTENNA1')
    antenna2 = ms.getcol('ANTENNA2')
    nant = np.maximum(antenna1.max() + 1, antenna2.max() + 1)

    spw = table(ms_name + '::SPECTRAL_WINDOW')
    freq = spw.getcol("CHAN_FREQ").squeeze()
    nchan = freq.size

    # now we need a dct containing field arrays
    xi = define_field_dct(ntime, nchan, nant, joness)

    param_arrays = ()
    for name in field_names:
        param_arrays += (xi[name],)

    # image params
    I = 1.0
    Q = 0.1
    U = 0.1
    V = 0.01
    
    # compute starting indices of stacked solvable params
    from numba import typed
    start_inds = typed.Dict()
    ntot = 0
    for name in solvable_names:
        if name not in start_inds:
            arr = xi[name]
            start_inds[name] = ntot
            ntot += np.prod(arr.shape)


    print(len(solvable_names), len(field_inds))

    quit()

    # evaluate model and Jacobian
    ti = timeit()
    Vpq, J, J2 = jacobian(tbin_idx, tbin_counts, antenna1, antenna2, freq,
                      R00, R01, R10, R11, dR00, dR01, dR10, dR11,
                      xi, field_names, field_inds, solvable_names, start_inds, ntot, param_arrays,
                      I, Q, U, V)
    print(timeit() - ti)

    


    # ti = timeit()
    # Vpq, J = jacobian(tbin_idx, tbin_counts, antenna1, antenna2, freq,
    #                   R00, R01, R10, R11, dR00, dR01, dR10, dR11,
    #                   xi, field_names, field_inds, solvable_names, start_inds, ntot, param_arrays,
    #                   I, Q, U, V)
    # print(timeit() - ti)

    x = np.random.randn(ntot)
    ti = timeit()
    tmp = J.dot(x)
    print(timeit() - ti)

    plt.figure('1')
    plt.imshow(np.abs(J[0:ntot, :]))
    plt.colorbar()

    ti = timeit()
    tmp2 = J2.dot(x)
    print(timeit() - ti)

    plt.figure('2')
    plt.imshow(np.abs(J2.toarray()[0:ntot, :]))
    plt.colorbar()

    print(np.abs(tmp - tmp2).max())

    plt.show()
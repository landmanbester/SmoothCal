import numpy as np
from smoothcal.response import param2vis as np_param2vis
from dask import blockwise

def _param2vis_wrapper(time_bin_indices, time_bin_counts, antenna1, antenna2,
                       g0a, g0p, g1a, g1p, 
                       k0, k1, l0, l1, 
                       b00a, b00p, b01a, b01p, b10a, b10p, b11a, b11p,
                       c, d, phi, R00, R01, R10, R11,
                       I, Q, U, V, kappa, freq):

    return np_param2vis(time_bin_indices, time_bin_counts, antenna1, antenna2,
                        g0a, g0p, g1a, g1p, 
                        k0, k1, l0, l1, 
                        b00a, b00p, b01a, b01p, b10a, b10p, b11a, b11p,
                        c, d, phi, R00, R01, R10, R11,
                        I, Q, U, V, kappa, freq)

def param2vis(time_bin_indices, time_bin_counts, antenna1, antenna2,
              g0a, g0p, g1a, g1p, 
              k0, k1, l0, l1, 
              b00a, b00p, b01a, b01p, b10a, b10p, b11a, b11p,
              c, d, phi, R00, R01, R10, R11,
              I, Q, U, V, kappa, freq):

    out_shape = ("row", "chan", "corr1", "corr2")
    jones_shape = ("row", "ant", "chan", "dir", "corr1", "corr2")

    return blockwise(_param2vis_wrapper, out_shape,
                     time_bin_indices, ("row",),
                     time_bin_counts, ("row",),
                     antenna1, ("row",),
                     antenna2, ("row",),
                     g0a, ('row', 'ant'), 
                     g0p, ('row', 'ant'),
                     g1a, ('row', 'ant'),
                     g1p, ('row', 'ant'),
                     k0, ('row', 'ant'),
                     k1, ('row', 'ant'),
                     l0, ('row', 'ant'),
                     l1, ('row', 'ant'),
                     b00a, ('ant', 'chan'),
                     b00p, ('ant', 'chan'),
                     b01a, ('ant', 'chan'),
                     b01p, ('ant', 'chan'),
                     b10a, ('ant', 'chan'),
                     b10p, ('ant', 'chan'),
                     b11a, ('ant', 'chan'),
                     b11p, ('ant', 'chan'),
                     c, ('row', 'ant'),
                     d, ('row', 'ant'),
                     phi, ('row', 'ant'),
                     R00, None,
                     R01, None, 
                     R10, None,
                     R11, None,
                     I, None,
                     Q, None,
                     U, None,
                     V, None,
                     kappa, None,
                     freq, ('chan',),
                     adjust_chunks={"row": antenna1.chunks[0]},
                     dtype=np.complex128,
                     align_arrays=False)

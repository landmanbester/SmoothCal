"""
Created Wed 11 April

@author Landman Bester

Set of utilities to simulate corrupted visibility data

"""

import numpy as np
import itertools as it
import utils

def sim_uv(Na, Nt, umax, vmax, Autocor=False, rot_params=(1,1)):
    """
    Simulates elliptical uv-tracks coverage 
    :param Na: number of antennae
    :param Nt: number of times
    :param umax: max u coordinate
    :param vmax: max v coordinate
    :param Autocor: whether to include auto-correlations or not
    :param rot_params: tuple or list specifying major and minor axes of rotation
    :returns:
        upq: u coordinates for baseline pq
        vpq: v coordinates for baseline pq
        pqlist: list of antennae pairs ordered in same order as upq and vpq axis 0
        N: number of baselines
    """
    if Autocor:
        pqlist = list(it.combinations_with_replacement(np.arange(Na), 2))
        N = Na * (Na + 1) // 2
    else:
        pqlist = list(it.combinations(np.arange(Na), 2))
        N = Na * (Na - 1) // 2
    # choose random antennae locations
    u = umax*np.random.random(Na)
    v = vmax*np.random.random(Na)
    # create baselines with time axis
    upq = np.zeros([N, Nt])
    vpq = np.zeros([N, Nt])
    phi = np.linspace(0, np.pi, Nt) # to simulate earth rotation
    for i, pq in enumerate(iter(pqlist)):
        p = pq[0]
        q = int(pq[1]) - 1
        upq[i, 0] = u[p] - u[q]
        vpq[i, 0] = v[p] - v[q]
        for j in xrange(1, Nt):
            rottheta = np.array([[rot_params[0]*np.cos(phi[j]), -rot_params[1]*np.sin(phi[j])],
                                 [rot_params[1]*np.sin(phi[j]), rot_params[0]*np.cos(phi[j])]])
            upq[i, j], vpq[i, j] = np.dot(rottheta, np.array([upq[i, 0], vpq[i, 0]]))
    return upq, vpq, pqlist, N

def sim_sky(Npix, Nsource, max_I, lmax, mmax):
    """
    Simulates a sky randomly populated with sources
    :param Npix: 
    :param Nsource: 
    :param max_I: 
    :param lmax: 
    :param mmax: 
    :return: 
    """
    l = np.linspace(-lmax, lmax, Npix)
    m = np.linspace(-mmax, mmax, Npix)
    ll, mm = np.meshgrid(l, m)
    lm = (np.vstack((ll.flatten(), mm.flatten())))
    lmsource = []
    IM = np.zeros([Npix, Npix])
    #IM[Npix//2, Npix//2] = 1.0
    for i in xrange(Nsource):
        locx = np.random.randint(2, Npix-2)
        locy = np.random.randint(2, Npix-2)
        IM[locx, locy] = np.abs(max_I*np.random.randn())
        lmsource.append((ll[locx, locy], mm[locx, locy]))
    return IM, lmsource, lm

def sim_gains(Na, Ns, thetas, lm, bounds=None):
    """
    Simulates DDE's 
    :param Na: number of antennae
    :param Ns: [Nnu, Nt]
    :param thetas: [theta_nu, theta_t, theta_l, theta_m]
    :param lm: 2 x Nsource array of source coordinates (only doing point sources for now)
    :param bounds: [(nu_min, nu_max), (t_min, t_max)]
    :return: 
    """
    if bounds is not None:
        nu = np.linspace(bounds[0][0], bounds[0][1], Ns[0])
        t = np.linspace(bounds[1][0], bounds[1][1], Ns[1])
    else:
        nu = np.linspace(1.0, 2.0, Ns[0])
        t = np.linspace(0,1, Ns[1])

    l = lm[0]
    m = lm[1]

    Nsource = l.size

    x = np.array([nu, t, l, m])

    meanf = lambda x: np.ones([x[0].size, x[1].size, x[2].size, x[3].size], dtype=np.complex128)

    gains = utils.draw_samples_ND_grid(x, thetas, Na, meanf=meanf)

    return gains



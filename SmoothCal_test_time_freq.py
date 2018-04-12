"""
Created 11/04/2018

@author Landman Bester

Testing time + frequency SmoothCal

"""

import numpy as np
import scipy.sparse as sps
import Simulator
import utils

def give_D_and_j(Vpq, Xpq, Wpq, K, Na, Nnu, Nt):
    A = sps.lil_matrix([Na*(Na-1)*Nnu*Nt, Na*Nt*Nnu], dtype=np.complex128)  # to hold per-antenna response
    V = np.zeros([Na*(Na-1)*Nnu*Nt], dtype=np.complex128)  # to hold per-antenna data
    W = np.zeros([Na*(Na-1)*Nnu*Nt], dtype=np.float64)  # to hold weights
    Sigma = np.zeros([Na*(Na-1)*Nnu*Nt], dtype=np.float64)  # to hold per-antenna weights
    Sigmay = np.zeros([Na * Nt * Nnu], dtype=np.float64)  # to hold diagonal of Ad.Sigmainv.A
    return

if __name__=="__main__":
    # set time and freq domain
    Nt = 100
    tmin = 0.0
    tmax = 10.0
    t = np.linspace(tmin, tmax, Nt)

    Nnu = 50
    numin = 1.0
    numax = 10.0
    nu = np.linspace(numin, numax, Nnu)

    Ns = [Nnu, Nt]

    # covariance params
    theta_nu = np.array([0.1, 5])
    theta_t = np.array([0.25, 1.0])

    thetas = np.array([theta_nu, theta_t])

    # model image
    Npix = 35
    Nsource = 5
    max_I = 1.0
    lmax = 0.1
    mmax = 0.1
    IM, lm, locs = Simulator.sim_sky(Npix, Nsource, max_I, lmax, mmax, nu, nu[Nnu//2])

    # gains
    Na = 4
    gains, K = Simulator.sim_DI_gains(Na, Ns, thetas, bounds=[(tmin, tmax), (numin, numax)])

    # uv-coverage
    umax = 10.0
    vmax = 10.0
    upq, vpq, pqlist, N = Simulator.sim_uv(Na, Nt, umax, vmax, rot_params=(2, 1))

    # data
    Xpq = np.zeros([Na, Na, Nnu, Nt], dtype=np.complex128)
    Xpq = utils.R_DI(IM, upq, vpq, lm, pqlist, nu, nu[Nnu//2], np.ones_like(gains), Xpq, Nnu, Nt, Nsource, DD=False)
    sigma = 0.1
    Vpq = utils.R_DI(IM, upq, vpq, lm, pqlist, nu, nu[Nnu//2], gains, Xpq, Nnu, Nt, Nsource, DD=False) + sigma**2*(
            np.random.randn(Na, Na, Nnu, Nt) + 1.0j*np.random.randn(Na, Na, Nnu, Nt))

    Wpq = np.ones_like(Vpq, dtype=np.float64)/(2*sigma**2)






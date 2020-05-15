import numpy as np
import sympy as sm
from numba import njit, typed, types
from africanus.gps.kernels import exponential_squared as expsq

def print_symbolic_jones_chain():
    from sympy import init_printing
    init_printing(use_latex=True, wrap_line=True)
    from IPython.display import display
    # scalars
    I, Q, U, V, k, v = sm.symbols("I Q U V kappa nu", real=True)
    Bs = sm.Matrix([[I + V, sm.exp(2*sm.I*v**2*k)*(Q + sm.I*U)],[sm.exp(2*sm.I*v**2*k)*(Q - sm.I*U), I - V]])
    display(Bs)

    # G
    gp0a, gp0p, gp1a, gp1p, gq0a, gq0p, gq1a, gq1p = sm.symbols("g_{p0a} g_{p0p} g_{p1a} g_{p1p} g_{q0a} g_{q0p} g_{q1a} g_{q1p}", real=True)
    Gp = sm.Matrix([[sm.exp(gp0a)*sm.exp(sm.I*gp0p), 0], [0, sm.exp(gp1a)*sm.exp(sm.I*gp1p)]])
    Gq = sm.Matrix([[sm.exp(gq0a)*sm.exp(sm.I*gq0p), 0], [0, sm.exp(gq1a)*sm.exp(sm.I*gq1p)]])
    display(Gp)
    display(Gq.H)

    # K 
    kp0, kp1, lp0, lp1, kq0, kq1, lq0, lq1 = sm.symbols("k_{p0} k_{p1} l_{p0} l_{p1} k_{q0} k_{q1} l_{q0} l_{q1}", real=True)
    Kp = sm.Matrix([[sm.exp(sm.I*(kp0*v + lp0)), 0], [0, sm.exp(sm.I*(kp1*v + lp1))]])
    Kq = sm.Matrix([[sm.exp(sm.I*(kq0*v + lq0)), 0], [0, sm.exp(sm.I*(kq1*v + lq1))]])
    display(Kp)
    display(Kq.H)

    # Bp
    bp00a, bp00p, bp01a, bp01p, bp10a, bp10p, bp11a, bp11p = sm.symbols("b_{p00a} b_{p00p} b_{p01a} b_{p01p} b_{p10a} b_{p10p} b_{p11a} b_{p11p}", real=True)
    Bp = sm.Matrix([[sm.exp(bp00a)*sm.exp(sm.I*bp00p), sm.exp(bp01a)*sm.exp(sm.I*bp01p)],[sm.exp(bp10a)*sm.exp(sm.I*bp10p), sm.exp(bp11a)*sm.exp(sm.I*bp11p)]])
    display(Bp)

    # Bq
    bq00a, bq00p, bq01a, bq01p, bq10a, bq10p, bq11a, bq11p = sm.symbols("b_{q00a} b_{q00p} b_{q01a} b_{q01p} b_{q10a} b_{q10p} b_{q11a} b_{q11p}", real=True)
    Bq = sm.Matrix([[sm.exp(bq00a)*sm.exp(sm.I*bq00p), sm.exp(bq01a)*sm.exp(sm.I*bq01p)],[sm.exp(bq10a)*sm.exp(sm.I*bq10p), sm.exp(bq11a)*sm.exp(sm.I*bq11p)]])
    display(Bq.H)

    # D
    cp, dp, cq, dq = sm.symbols("c_p d_p c_q d_q", real=True)
    Dp = sm.Matrix([[sm.exp(sm.I*(cp*v + dp)), 0], [0, sm.exp(sm.I*(cp*v + dp))]])
    Dq = sm.Matrix([[sm.exp(sm.I*(cq*v + dq)), 0], [0, sm.exp(sm.I*(cq*v + dq))]])
    display(Dp)
    display(Dq.H)

    # P
    phip, phiq = sm.symbols('phi_p phi_q', real=True)
    Pp = sm.Matrix([[sm.exp(-sm.I*phip), 0],[0, sm.exp(sm.I*phip)]])
    Pq = sm.Matrix([[sm.exp(-sm.I*phiq), 0],[0, sm.exp(sm.I*phiq)]])
    display(Pp)
    display(Pq)

    RIME = Gp*Kp*Bp*Dp*Pp*Bs*Pq.H*Dq.H*Bq.H*Kq.H*Gq.H
    RIME = sm.simplify(RIME) 

    display(RIME[0,0])
    display(RIME[0,1])
    display(RIME[1,0])
    display(RIME[1,1])

    print('00=', sm.latex(RIME[0,0]))


    print('01=', sm.latex(RIME[0,1]))


    print('10=', sm.latex(RIME[1,0]))


    print('11=', sm.latex(RIME[1,1]))

def symbolic_jones_chain(joness='GKBDP', solvables='11110', poltype='linear'):
    assert len(joness)==len(solvables)
    unsolvable_params = ()
    solvable_params = ()
    # brightness matrix (assumed fixed for the time being)
    I, Q, U, V, nu = sm.symbols("I, Q, U, V, nu", real=True)
    if poltype=='circular':
        Bs = sm.Matrix([[I + V, Q + sm.I*U],[Q - sm.I*U, I - V]])
    elif poltype=='linear':
        Bs = sm.Matrix([[I + Q, U + sm.I*V],[U - sm.I*V, I - Q]])
    else:
        raise ValueError('Unrecognised poltype %s'%poltype)

    unsolvable_params += (nu, I, Q, U, V)
    # LB - TODO - rotation measure jones term
    # sm.exp(2*sm.I*nu**2*kappa)

    # G
    def Gterm():
        gp0a, gp0p, gp1a, gp1p, gq0a, gq0p, gq1a, gq1p = sm.symbols("gp0a, gp0p, gp1a, gp1p, gq0a, gq0p, gq1a, gq1p", real=True)
        Gp = sm.Matrix([[sm.exp(gp0a)*sm.exp(sm.I*gp0p), 0], [0, sm.exp(gp1a)*sm.exp(sm.I*gp1p)]])
        Gq = sm.Matrix([[sm.exp(gq0a)*sm.exp(sm.I*gq0p), 0], [0, sm.exp(gq1a)*sm.exp(sm.I*gq1p)]])
        return Gp, Gq, (gp0a, gp0p, gp1a, gp1p, gq0a, gq0p, gq1a, gq1p)

    # K 
    def Kterm():
        kp0, kp1, lp0, lp1, kq0, kq1, lq0, lq1 = sm.symbols("kp0, kp1, lp0, lp1, kq0, kq1, lq0, lq1", real=True)
        Kp = sm.Matrix([[sm.exp(sm.I*(kp0*nu + lp0)), 0], [0, sm.exp(sm.I*(kp1*nu + lp1))]])
        Kq = sm.Matrix([[sm.exp(sm.I*(kq0*nu + lq0)), 0], [0, sm.exp(sm.I*(kq1*nu + lq1))]])
        return Kp, Kq, (kp0, kp1, lp0, lp1, kq0, kq1, lq0, lq1)


    # B 
    def Bterm():
        # Bp
        bp00a, bp00p, bp01a, bp01p, bp10a, bp10p, bp11a, bp11p = sm.symbols("bp00a, bp00p, bp01a, bp01p, bp10a, bp10p, bp11a, bp11p", real=True)
        Bp = sm.Matrix([[sm.exp(bp00a)*sm.exp(sm.I*bp00p), sm.exp(bp01a)*sm.exp(sm.I*bp01p)],[sm.exp(bp10a)*sm.exp(sm.I*bp10p), sm.exp(bp11a)*sm.exp(sm.I*bp11p)]])
        # Bp = sm.Matrix([[sm.exp(bp00a)*sm.exp(sm.I*bp00p), 0],[0, sm.exp(bp11a)*sm.exp(sm.I*bp11p)]])

        # Bq
        bq00a, bq00p, bq01a, bq01p, bq10a, bq10p, bq11a, bq11p = sm.symbols("bq00a, bq00p, bq01a, bq01p, bq10a, bq10p, bq11a, bq11p", real=True)
        Bq = sm.Matrix([[sm.exp(bq00a)*sm.exp(sm.I*bq00p), sm.exp(bq01a)*sm.exp(sm.I*bq01p)],[sm.exp(bq10a)*sm.exp(sm.I*bq10p), sm.exp(bq11a)*sm.exp(sm.I*bq11p)]])
        # Bq = sm.Matrix([[sm.exp(bq00a)*sm.exp(sm.I*bq00p), 0],[0, sm.exp(bq11a)*sm.exp(sm.I*bq11p)]])
        return Bp, Bq, (bp00a, bp00p, bp01a, bp01p, bp10a, bp10p, bp11a, bp11p, bq00a, bq00p, bq01a, bq01p, bq10a, bq10p, bq11a, bq11p)

    # D
    def Dterm():
        cp, dp, cq, dq = sm.symbols("cp, dp, cq, dq", real=True)
        # Dp = sm.exp(sm.I*(cp*v + dp))*sm.eye(2)
        Dp = sm.Matrix([[sm.exp(sm.I*(cp*nu + dp)), 0], [0, sm.exp(sm.I*(cp*nu + dp))]])
        # Dq = sm.exp(sm.I*(cq*v + dq))*sm.eye(2)
        Dq = sm.Matrix([[sm.exp(sm.I*(cq*nu + dq)), 0], [0, sm.exp(sm.I*(cq*nu + dq))]])
        return Dp, Dq, (cp, dp, cq, dq)

    # P
    def Pterm():
        phip, phiq = sm.symbols('phip, phiq', real=True)
        if poltype == 'circular':
            Pp = sm.Matrix([[sm.exp(-sm.I*phip), 0],[0, sm.exp(sm.I*phip)]])
            Pq = sm.Matrix([[sm.exp(-sm.I*phiq), 0],[0, sm.exp(sm.I*phiq)]])
        elif poltype=='linear':
            Pp = sm.Matrix([[sm.cos(phip), -sm.sin(phip)],[sm.sin(phip), sm.cos(phip)]])
            Pq = sm.Matrix([[sm.cos(phiq), -sm.sin(phiq)],[sm.sin(phiq), sm.cos(phiq)]])
        else:
            raise ValueError('Unrecognised poltype %s'%poltype)
        return Pp, Pq, (phip, phiq)

    # build RIME model
    print("Constructing symbolic RIME")
    RIME = Bs
    for jones in joness[::-1]:  # need to traverse Jones chain in reverse
        if jones == 'G':
            print("G")
            Gp, Gq, Gparams = Gterm()
            RIME = sm.simplify(Gp * RIME * Gq.H)
            ind = joness.find('G')
            if int(solvables[ind]):
                solvable_params += Gparams
            else:
                unsolvable_params += Gparams
        elif jones == 'K':
            print("K")
            Kp, Kq, Kparams = Kterm()
            RIME = sm.simplify(Kp * RIME * Kq.H)
            ind = joness.find('K')
            if int(solvables[ind]):
                solvable_params += Kparams
            else:
                unsolvable_params += Kparams
        elif jones == 'B':
            print("B")
            Bp, Bq, Bparams = Bterm()
            RIME = sm.simplify(Bp * RIME * Bq.H)
            ind = joness.find('B')
            if int(solvables[ind]):
                solvable_params += Bparams
            else:
                unsolvable_params += Bparams
        elif jones == 'D':
            print("D")
            Dp, Dq, Dparams = Dterm()
            RIME = sm.simplify(Dp * RIME * Dq.H)
            ind = joness.find('D')
            if int(solvables[ind]):
                solvable_params += Dparams
            else:
                unsolvable_params += Dparams
        elif jones == 'P':
            print('P')
            Pp, Pq, Pparams = Pterm()
            RIME = sm.simplify(Pp * RIME * Pq.H)
            ind = joness.find('P')
            if int(solvables[ind]):
                solvable_params += Pparams
            else:
                unsolvable_params += Pparams            
        else:
            raise ValueError("Unrecognised Jones term %s"%jones)
    
    RIME = sm.simplify(RIME) 

    parameters = unsolvable_params + solvable_params

    print(parameters)

    # symbolic -> LLVM
    from sympy.utilities.lambdify import lambdify

    # 00
    print('Compiling 00')
    tmp = lambdify(parameters, RIME[0,0], 'numpy')
    R00 = njit(nogil=True, fastmath=True)(tmp)
    dR00 = ()
    for param in solvable_params:
        print(str(param))
        tmp = sm.simplify(RIME[0,0].diff(param))
        tmp = lambdify(parameters, tmp, 'numpy')
        dR00 += (njit(nogil=True, fastmath=True)(tmp),)

    # 01
    print('Compiling 01')
    tmp = lambdify(parameters, RIME[0,1], 'numpy')
    R01 = njit(nogil=True, fastmath=True)(tmp)
    dR01 = ()
    for param in solvable_params:
        print(str(param))
        tmp = sm.simplify(RIME[0,1].diff(param))
        tmp = lambdify(parameters, tmp, 'numpy')
        dR01 += (njit(nogil=True, fastmath=True)(tmp),)

    # 10
    print('Compiling 10')
    tmp = lambdify(parameters, RIME[1,0], 'numpy')
    R10 = njit(nogil=True, fastmath=True)(tmp)
    dR10 = ()
    for param in solvable_params:
        print(str(param))
        tmp = sm.simplify(RIME[1,0].diff(param))
        tmp = lambdify(parameters, tmp, 'numpy')
        dR10 += (njit(nogil=True, fastmath=True)(tmp),)

    # 11
    print('Compiling 11')
    tmp = lambdify(parameters, RIME[1,1], 'numpy')
    R11 = njit(nogil=True, fastmath=True)(tmp)
    dR11 = ()
    for param in solvable_params:
        print(str(param))
        tmp = sm.simplify(RIME[1,1].diff(param))
        tmp = lambdify(parameters, tmp, 'numpy')
        dR11 += (njit(nogil=True, fastmath=True)(tmp),)

    return R00, R01, R10, R11, dR00, dR01, dR10, dR11

def domain2param_mapping(joness, solvables):
    @njit(nogil=True, fastmath=True)
    def time_map_p(t, p, q, nu):
        return (p, t, 0)
    @njit(nogil=True, fastmath=True)
    def time_map_q(t, p, q, nu):
        return (q, t, 0)
    @njit(nogil=True, fastmath=True)
    def freq_map_p(t, p, q, nu):
        return (p, 0, nu)
    @njit(nogil=True, fastmath=True)
    def freq_map_q(t, p, q, nu):
        return (q, 0, nu)

    field_names = ()
    field_inds = ()
    solvable_names = ()
    for jones in joness[::-1]:  # need to traverse Jones chain in reverse
        if jones == 'G':
            ind = joness.find('G')
            if int(solvables[ind]):
                solvable_names += 2*('g0a', 'g0p', 'g1a', 'g1p')
            field_names += 2*('g0a', 'g0p', 'g1a', 'g1p')
            field_inds += 4*(time_map_p,) + 4*(time_map_q,) 
        elif jones == 'K':
            ind = joness.find('K')
            if int(solvables[ind]):
                solvable_names += 2*('k0', 'k1', 'l0', 'l1')
            field_names += 2*('k0', 'k1', 'l0', 'l1')
            field_inds += 4*(time_map_p,) + 4*(time_map_q,) 
        elif jones == 'B':
            ind = joness.find('B')
            if int(solvables[ind]):
                solvable_names += 2*('b00a', 'b00p', 'b01a', 'b01p', 'b10a', 'b10p', 'b11a', 'b11p')
            field_names += 2*('b00a', 'b00p', 'b01a', 'b01p', 'b10a', 'b10p', 'b11a', 'b11p')
            field_inds += 8*(freq_map_p,) + 8*(freq_map_q,) 
        elif jones == 'D':
            ind = joness.find('D')
            if int(solvables[ind]):
                solvable_names += 2*('c', 'd')
            field_names += 2*('c', 'd')
            field_inds += 2*(time_map_p,) + 2*(time_map_q,) 
        elif jones == 'P':
            ind = joness.find('P')
            if int(solvables[ind]):
                solvable_names += 2*('phi',)
            field_names += 2*('phi',)
            field_inds += (time_map_p, time_map_q) 
            
        else:
            raise ValueError("Unrecognised Jones term %s"%jones)

    return field_names, field_inds, solvable_names

# def domain2param_mapping2(joness='GKBDP'):
#     field_names = ()
#     field_inds = ()
#     for jones in joness[::-1]:  # need to traverse Jones chain in reverse
#         if jones == 'G':
#             field_names += 2*('g0a', 'g0p', 'g1a', 'g1p')
#             field_inds += (lambda t, p, q, nu: ((t, p, slice(None)),(t, p, slice(None)),(t, p, slice(None)),(t, p, slice(None)),(t, q, slice(None)),(t, q, slice(None)),(t, q, slice(None)),(t, q, slice(None))),)
#         elif jones == 'K':
#             field_names += 2*('k0', 'k1', 'l0', 'l1')
#             field_inds += (lambda t, p, q, nu: ((t, p, slice(None)),(t, p, slice(None)),(t, p, slice(None)),(t, p, slice(None)),(t, q, slice(None)),(t, q, slice(None)),(t, q, slice(None)),(t, q, slice(None))),)
#         elif jones == 'B':
#             field_names += 2*('b00a', 'b00p', 'b01a', 'b01p', 'b10a', 'b10p', 'b11a', 'b11p')
#             field_inds += (lambda t, p, q, nu: ((slice(None), p, nu),(slice(None), p, nu),(slice(None), p, nu),(slice(None), p, nu),(slice(None), p, nu),(slice(None), p, nu),(slice(None), p, nu),(slice(None), p, nu),
#                                                 (slice(None), q, nu),(slice(None), q, nu),(slice(None), q, nu),(slice(None), q, nu),(slice(None), q, nu),(slice(None), q, nu),(slice(None), q, nu),(slice(None), q, nu)),)
#         elif jones == 'D':
#             field_names += 2*('c', 'd')
#             field_inds += (lambda t, p, q, nu: ((t, p, slice(None)),(t, p, slice(None)),(t, q, slice(None)),(t, q, slice(None))),)
#         elif jones == 'P':
#             field_names += 2*('phi',)
#             field_inds += (lambda t, p, q, nu: ((t, p, slice(None)),(t, q, slice(None))),)
            
#         else:
#             raise ValueError("Unrecognised Jones term %s"%jones)

#     # construct (t, p, q, nu) -> field_inds function
#     func = lambda i:field_inds[i]
#     mapping = map(func, range(len(field_inds)))

#     return field_names, mapping


def define_field_dct(ntime, nchan, nant, joness):
    xi = typed.Dict()
    if 'G' in joness:
        xi['g0a'] = np.random.randn(nant, ntime, 1)
        xi['g0p'] = np.random.randn(nant, ntime, 1)
        xi['g1a'] = np.random.randn(nant, ntime, 1)
        xi['g1p'] = np.random.randn(nant, ntime, 1)

    if 'K' in joness:
        xi['k0'] = np.random.randn(nant, ntime, 1)
        xi['k1'] = np.random.randn(nant, ntime, 1)
        xi['l0'] = np.random.randn(nant, ntime, 1)
        xi['l1'] = np.random.randn(nant, ntime, 1)

    if 'B' in joness:
        xi['b00a'] = np.random.randn(nant, 1, nchan)
        xi['b00p'] = np.random.randn(nant, 1, nchan)
        xi['b01a'] = np.random.randn(nant, 1, nchan)
        xi['b01p'] = np.random.randn(nant, 1, nchan)
        xi['b10a'] = np.random.randn(nant, 1, nchan)
        xi['b10p'] = np.random.randn(nant, 1, nchan)
        xi['b11a'] = np.random.randn(nant, 1, nchan)
        xi['b11p'] = np.random.randn(nant, 1, nchan)

    if 'D' in joness:
        xi['c'] = np.random.randn(nant, ntime, 1)
        xi['d'] = np.random.randn(nant, ntime, 1)

    if 'P' in joness:
        xi['phi'] = np.random.randn(nant, ntime, 1)

    return xi
    

def define_fields(time, freq, nant):
    xi = {}
    t = time/np.mean(time)
    ntime = t.size
    nu = freq/np.mean(freq)
    nchan = nu.size

    # G 
    xi['g0a'] = {}
    xi['g0a']['theta'] = (0.1, 1.0)  # (sigma_f, l) for all antennas
    xi['g0a']['domain'] = t
    xi['g0a']['pos'] = np.random.randn(ntime, nant)  # white field per antenna
    

    xi['g0p'] = {}
    xi['g0p']['theta'] = (0.1, 1.0)  
    xi['g0p']['domain'] = t
    xi['g0p']['pos'] = np.random.randn(ntime, nant)  
    

    xi['g1a'] = {}
    xi['g1a']['theta'] = (0.1, 1.0)  
    xi['g1a']['domain'] = t
    xi['g1a']['pos'] = np.random.randn(ntime, nant)
    

    xi['g1p'] = {}
    xi['g1p']['theta'] = (0.1, 1.0)  
    xi['g1p']['domain'] = t
    xi['g1p']['pos'] = np.random.randn(ntime, nant)  
    

    # K
    xi['k0'] = {}
    xi['k0']['theta'] = (0.25, 1.0)
    xi['k0']['domain'] = t
    xi['k0']['pos'] = np.random.randn(ntime, nant)  
    

    xi['k1'] = {}
    xi['k1']['theta'] = (0.25, 1.0)
    xi['k1']['domain'] = t
    xi['k1']['pos'] = np.random.randn(ntime, nant)  
    

    xi['l0'] = {}
    xi['l0']['theta'] = (0.25, 1.0)
    xi['l0']['domain'] = t
    xi['l0']['pos'] = np.random.randn(ntime, nant)  
    

    xi['l1'] = {}
    xi['l1']['theta'] = (0.25, 1.0)
    xi['l1']['domain'] = t
    xi['l1']['pos'] = np.random.randn(ntime, nant)  
    

    # B
    xi['b00a'] = {}
    xi['b00a']['theta'] = (0.1, 0.25)
    xi['b00a']['domain'] = nu
    xi['b00a']['pos'] = np.random.randn(nchan, nant)

    xi['b00p'] = {}
    xi['b00p']['theta'] = (0.1, 0.25)
    xi['b00p']['domain'] = nu
    xi['b00p']['pos'] = np.random.randn(nchan, nant)  

    xi['b11a'] = {}
    xi['b11a']['theta'] = (0.1, 0.25)
    xi['b11a']['domain'] = nu
    xi['b11a']['pos'] = np.random.randn(nchan, nant)

    xi['b11p'] = {}
    xi['b11p']['theta'] = (0.1, 0.25)
    xi['b11p']['domain'] = nu
    xi['b11p']['pos'] = np.random.randn(nchan, nant) 

    xi['b01a'] = {}
    xi['b01a']['theta'] = (0.01, 0.25)
    xi['b01a']['domain'] = nu
    xi['b01a']['pos'] = np.random.randn(nchan, nant)

    xi['b01p'] = {}
    xi['b01p']['theta'] = (0.01, 0.25)
    xi['b01p']['domain'] = nu
    xi['b01p']['pos'] = np.random.randn(nchan, nant) 

    xi['b10a'] = {}
    xi['b10a']['theta'] = (0.01, 0.25)
    xi['b10a']['domain'] = nu
    xi['b10a']['pos'] = np.random.randn(nchan, nant)

    xi['b10p'] = {}
    xi['b10p']['theta'] = (0.01, 0.25)
    xi['b10p']['domain'] = nu
    xi['b10p']['pos'] = np.random.randn(nchan, nant)

    # D
    xi['c'] = {}
    xi['c']['theta'] = (0.1, 0.1)
    xi['c']['domain'] = t
    xi['c']['pos'] = np.random.randn(ntime, nant)

    xi['d'] = {}
    xi['d']['theta'] = (0.1, 0.1)
    xi['d']['domain'] = t
    xi['d']['pos'] = np.random.randn(ntime, nant)

    return xi

def field2param(xi):
    """
    Maps a white field to a correlated field
    """
    x = xi['domain']
    theta = xi['theta']
    K = expsq(x, x, theta[0], theta[1])
    A = np.linalg.cholesky(K + 1e-12*np.eye(x.size))
    #A = np.zeros(K.shape)
    return np.einsum('ab,bc->ac', A, xi['pos'])


def params2jones(xi, freq, I, Q, U, V, kappa):
    """
    Takes parameters as defined in pdf doc to matrix valued gains

    gia - (ntime, nant)
    gip - (ntime, nant)
    
    biia - (nchan, nant)
    biip - (nchan, nant)
    
    ki - (ntime, nant)
    li - (ntime, nant)
    
    c - (ntime, nant)
    d - (ntime, nant)
    
    freq - (nchan)
    
    I - float
    Q - float
    U - float
    V - float
    kappa - float

    Function is wasteful and only meant for testing purposes

    """
    ntime, nant = xi['g1a']['pos'].shape
    nchan = freq.size

    # brightness matrix (assuming flat spectra for now)
    Bs = np.zeros((nchan, 2, 2), dtype=np.complex128)
    Bs[:, 0, 0] = I+V
    Bs[:, 1, 1] = I-V
    Bs[:, 0, 1] = np.exp(2j*freq**2*kappa)*(Q + 1j*U)
    Bs[:, 1, 0] = np.exp(2j*freq**2*kappa)*(Q - 1j*U)

    # get colored fields and broadcast to expected shape (ntime, nant, nchan)
    g0a = field2param(xi['g0a'])[:, :, None]
    g0p = field2param(xi['g0p'])[:, :, None]
    g1a = field2param(xi['g1a'])[:, :, None]
    g1p = field2param(xi['g1p'])[:, :, None]

    b00a = field2param(xi['b00a']).T[None]
    b00p = field2param(xi['b00p']).T[None]
    b01a = field2param(xi['b01a']).T[None]
    b01p = field2param(xi['b01p']).T[None]
    b10a = field2param(xi['b10a']).T[None]
    b10p = field2param(xi['b10p']).T[None]
    b11a = field2param(xi['b11a']).T[None]
    b11p = field2param(xi['b11p']).T[None]
    
    k0 = field2param(xi['k0'])[:, :, None]
    k1 = field2param(xi['k1'])[:, :, None]
    l0 = field2param(xi['l0'])[:, :, None]
    l1 = field2param(xi['l1'])[:, :, None]
    
    c = field2param(xi['c'])[:, :, None]
    d = field2param(xi['d'])[:, :, None]
    
    freq = freq[None, None]

    # fill in correlations
    G = np.zeros((ntime, nant, nchan, 2, 2), dtype=np.complex128)
    K = np.zeros((ntime, nant, nchan, 2, 2), dtype=np.complex128)
    B = np.zeros((ntime, nant, nchan, 2, 2), dtype=np.complex128)
    D = np.zeros((ntime, nant, nchan, 2, 2), dtype=np.complex128)

    G[:, :, :, 0, 0] = np.tile(np.exp(g0a + 1.0j*g0p), (1, 1, nchan))
    G[:, :, :, 1, 1] = np.tile(np.exp(g1a + 1.0j*g1p), (1, 1, nchan))
    
    K[:, :, :, 0, 0] = np.exp(1.0j*(k0*freq + l0))
    K[:, :, :, 1, 1] = np.exp(1.0j*(k1*freq + l1))
    
    B[:, :, :, 0, 0] = np.tile(np.exp(b00a + 1j*b00p), (ntime, 1, 1))
    B[:, :, :, 0, 1] = np.tile(np.exp(b01a + 1j*b01p), (ntime, 1, 1))
    B[:, :, :, 1, 0] = np.tile(np.exp(b10a + 1j*b10p), (ntime, 1, 1))
    B[:, :, :, 1, 1] = np.tile(np.exp(b11a + 1j*b11p), (ntime, 1, 1))

    D[:, :, :, 0, 0] = np.exp(1j*(c*freq + d))
    D[:, :, :, 1, 1] = np.exp(1j*(c*freq + d))

    return G, K, B, D, Bs

def rime00(cp, cq, dp, dq, phip, phiq, I, V, nu):
    return (I+V)*np.exp(1.0j*(cp*nu - cq*nu + dp - dq - phip + phiq))

def rime01(cp, cq, dp, dq, phip, phiq, Q, U, kappa, nu):
    return (Q + 1.0j*U)*np.exp(1.0j*(cp*nu - cq*nu + dp - dq - phip - phiq + 2*kappa*nu**2))

def rime10(cp, cq, dp, dq, phip, phiq, Q, U, kappa, nu):
    return (Q - 1.0j*U)*np.exp(1.0j*(cp*nu - cq*nu + dp - dq + phip + phiq + 2*kappa*nu**2))

def rime11(cp, cq, dp, dq, phip, phiq, I, V):
    return (I-V)*np.exp(1.0j*(cp*nu - cq*nu + dp - dq + phip - phiq))
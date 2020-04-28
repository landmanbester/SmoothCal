#  Coefficients for
#  log(S) = a + b*log(f) +c*log(f)**2 + d*log(f)**3
#  S in Jy
#  f in MHz (Perley versions in GHz !!)
# 
#  Generally valid from 843MHz (SUMSS) to 20GHz (ATCA).
#  Some have data points from PAPER (145MHz)

import numpy as np

def coord2rad(ra, dec):
    ra_h, tmp = ra.split('h')
    ra_m, tmp = tmp.split('m')
    ra_s = tmp.replace('s','')

    dec_d, tmp = dec.split('d')
    dec_m, tmp = tmp.split('m')
    dec_s = tmp.reapce('s', '')

    return np.deg2rad(float(ra_h)*15.0 + float(ra_m)*15/60 + float(ra_s)*15/60**2), 
           np.deg2rad(float(dec_d) + float(dec_m)/60 + float(dec_s)/60**2)

def logpoly(freq, a, b, c, d):
    return a + b*np.log(freq) + c*np.log(freq)**2 + d*np.log(freq)**3

def src_0056_001():
    """
    name=0056-001 epoch=2016 ra=00h59m05.5s dec=+00d06m52s a=-0.9051
    b=1.8362 c=-0.6141 d=0.0518
    """
    ra, dec= coord2rad('00h59m05.5s', '+00d06m52s') 
    a=-0.9051
    b=1.8362 
    c=-0.6141 
    d=0.0518 
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_0201_440():
    """
    name=0201-440 epoch=2016 ra=02h03m40.8s dec=-43d49m52s a=1.4053
    b=0.1893 c=-0.1654 d=0.0030
    """
    ra, dec= coord2rad('02h03m40.8s', '-43d49m52s') 
    a=1.4053
    b=0.1893 
    c=-0.1654 
    d=0.0030
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_0237_233():
    """
    name=0237-233 epoch=2016 ra=02h40m08.2s dec=-23d09m16s a=-23.5170
    b=20.5047 c=-5.5855 d=0.4851
    """
    ra, dec= coord2rad('02h40m08.2s', '-23d09m16s') 
    a=-23.5170
    b=20.5047 
    c=-5.5855 
    d=0.4851
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_0252_712():
    """
    name=0252-712 epoch=2016 ra=02h52m46.1s dec=-71d04m35s a=1.1245
    b=0.7190 c=-0.2955 d=0.0096
    """
    ra, dec= coord2rad('02h52m46.1s', '-71d04m35s') 
    a=1.1245
    b=0.7190 
    c=-0.2955 
    d=0.0096
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_CTA21():
    """
    name=CTA21    epoch=2016 ra=03h18m57.8s dec=+16d28m33s a=-13.4672
    =12.4587 c=-3.4110 d=0.2867
    """
    ra, dec= coord2rad('03h18m57.8s', '+16d28m33s') 
    a=-13.4672
    b=12.4587 
    c=-3.4110 
    d=0.2867
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_0320_053():
    """
    name=0320+053 epoch=2016 ra=03h23m20.3s dec=+05d34m12s a=-6.9043
    b=7.8471 c=-2.4713 d=0.2296
    """
    ra, dec= coord2rad('03h23m20.3s', '+05d34m12s') 
    a=-6.9043
    b=7.8471 
    c=-2.4713 
    d=0.2296
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_0403_132():
    """
    name=0403-132 epoch=2016 ra=04h05m34.0s dec=-13d08m14s a=-0.8395
    b=2.4438 c=-0.9451 d=0.1006
    """
    ra, dec= coord2rad('04h05m34.0s', '-13d08m14s') 
    a=-0.8395
    b=2.4438 
    c=-0.9451 
    d=0.1006
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_0408_65():
    """
    name=0408-65 epoch=2016 ra=04h08m20.4s dec=-65d45m09s a=-0.9790
    b=3.3662 c=-1.1216 d=0.0861
    """
    ra, dec= coord2rad('04h08m20.4s', '-65d45m09s') 
    a=-0.9790
    b=3.3662 
    c=-1.1216 
    d=0.0861
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_PKS1934_63():
    """
    name=PKS1934-63 epoch=1994 ra=19h39m25.0s dec=-63d42m46s a=-30.7667
    b=26.4908 c=-7.0977 d=0.605334
    """
    ra, dec= coord2rad('19h39m25.0s', '-63d42m46s') 
    a=-30.7667
    b=26.4908 
    c=-7.0977 
    d=0.605334
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_0410_752():
    """
    name=0410-752 epoch=2016 ra=04h08m48.5s dec=-75d07m19s a=8.1443
    b=-5.2122 c=1.4472 d=-0.1584
    """
    ra, dec= coord2rad('04h08m48.5s', '-75d07m19s') 
    a=8.1443
    b=-5.2122 
    c=1.4472 
    d=-0.1584
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_0420_625():
    """
    name=0420-625 epoch=2016 ra=04h20m56.0s dec=-62d23m40s a=4.7772
    b=-2.1190 c=0.3788 d=-0.0433
    """
    ra, dec= coord2rad('04h20m56.0s', '-62d23m40s') 
    a=4.7772
    b=-2.1190 
    c=0.3788 
    d=-0.0433
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_0531_194():
    """
    name=0531+194 epoch=2016 ra=05h34m44.5s dec=+19d27m22s a=-3.4915
    b=5.0261 c=-1.6476 d=0.1549
    """
    ra, dec= coord2rad('05h34m44.5s', '+19d27m22s') 
    a=-3.4915
    b=5.0261 
    c=-1.6476 
    d=0.1549
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_0823_500():
    """
    name=0823-500 epoch=2016 ra=08h25m26.9s dec=-50d10m38s a=-33.5795
    b=27.9753 c=-7.3670 d=0.6198
    """
    ra, dec= coord2rad('08h25m26.9s', '-50d10m38s') 
    a=-33.5795
    b=27.9753 
    c=-7.3670 
    d=0.6198
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_0906_682():
    """
    name=0906-682 epoch=2016 ra=09h06m52.4s dec=-68d29m40s a=0.7680
    b=1.3416 c=-0.6601 d=0.0573
    """
    ra, dec= coord2rad('09h06m52.4s', '-68d29m40s') 
    a=0.7680
    b=1.3416 
    c=-0.6601 
    d=0.0573
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_3C237():
    """
    name=3C237    epoch=2016 ra=10h08m00.0s dec=+07d30m16s a=3.1549
    b=-1.0771 c=0.2183 d=-0.0356
    """
    ra, dec= coord2rad('10h08m00.0s', '+07d30m16s') 
    a=3.1549
    b=-1.0771 
    c=0.2183 
    d=-0.0356
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_1127_145():
    """
    name=1127-145 epoch=2016 ra=11h30m07.0s dec=-14d49m27s a=-4.7962
    b=4.3260 c=-1.0387 d=0.0710
    """
    ra, dec= coord2rad('11h30m07.0s', '-14d49m27s') 
    a=-4.7962
    b=4.3260 
    c=-1.0387 
    d=0.0710
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_1139_285():
    """
    name=1139-285 epoch=2016 ra=11h41m34.5s dec=-28d50m52s a=0.7452
    b=1.2832 c=-0.6408 d=0.0632
    """
    ra, dec= coord2rad('11h41m34.5s', '-28d50m52s') 
    a=0.7452
    b=1.2832
    c=-0.6408
    d=0.0632
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_1151_348():
    """
    name=1151-348 epoch=2016 ra=11h54m21.8s dec=-35d05m29s a=-2.9686
    b=3.9700 c=-1.2009 d=0.1020
    """
    ra, dec= coord2rad('11h54m21.8s', '-35d05m29s') 
    a=-2.9686
    b=3.9700 
    c=-1.2009 
    d=0.1020
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_1240_209():
    """
    name=1240-209 epoch=2016 ra=12h43m12.3s dec=-21d12m13s a=5.2689
    b=-4.0352 c=1.2650 d=-0.1561
    """
    ra, dec= coord2rad('12h43m12.3s', '-21d12m13s') 
    a=5.2689
    b=-4.0352 
    c=1.2650 
    d=-0.1561
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_3C283():
    """
    name=3C283    epoch=2016 ra=13h11m40.1s dec=-22d17m04s a=-3.3516
    b=6.4132 c=-2.4230 d=0.2522
    """
    ra, dec= coord2rad('13h11m40.1s', '-22d17m04s') 
    a=-3.3516
    b=6.4132 
    c=-2.4230 
    d=0.2522
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_1421_490():
    """
    name=1421-490 epoch=2016 ra=14h24m32.2s dec=-49d13m50s a=6.0550
    b=-4.1963 c=1.1975 d=-0.1218
    """
    ra, dec= coord2rad('14h24m32.2s', '-49d13m50s') 
    a=6.0550
    b=-4.1963 
    c=1.1975 
    d=-0.1218
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_1740_517():
    """
    name=1740-517 epoch=2016 ra=17h44m25.4s dec=-51d44m44s a=-3.6941
    b=2.7447 c=-0.3655 d=-0.0153
    """
    ra, dec= coord2rad('17h44m25.4s', '-51d44m44s') 
    a=-3.6941
    b=2.7447 
    c=-0.3655 
    d=-0.0153
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_1827_360():
    """
    name=1827-360 epoch=2016 ra=18h30m58.8s dec=-36d02m30s a=0.4218
    b=1.4761 c=-0.4243 d=0.0002
    """
    ra, dec= coord2rad('18h30m58.8s', '-36d02m30s') 
    a=0.4218
    b=1.4761 
    c=-0.4243 
    d=0.0002
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_1830_210():
    """
    name=1830-210 epoch=2016 ra=18h33m39.9s dec=-21d03m40s a=-6.2117
    b=6.0429 c=-1.6037 d=0.1333
    """
    ra, dec= coord2rad('18h33m39.9s', '-21d03m40s') 
    a=-6.2117
    b=6.0429 
    c=-1.6037 
    d=0.1333
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_1921_293():
    """
    name=1921-293 epoch=2016 ra=19h24m51.0s dec=-29d14m30s a=3.0213
    b=-2.2634 c=0.7651 d=-0.0800
    """
    ra, dec= coord2rad('19h24m51.0s', '-29d14m30s') 
    a=3.0213
    b=-2.2634 
    c=0.7651 
    d=-0.0800
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_1938_155():
    """
    name=1938-155 epoch=2016 ra=19h41m15.1s dec=-15d24m31s a=-2.1724
    b=4.1589 c=-1.4511 d=0.1368
    """
    ra, dec= coord2rad('19h41m15.1s', '-15d24m31s') 
    a=-2.1724
    b=4.1589 
    c=-1.4511 
    d=0.1368
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_2203_188():
    """
    name=2203-188 epoch=2016 ra=22h06m10.4s dec=-18d35m39s a=2.1847
    b=-0.7118 c=0.1563 d=-0.0215
    """
    ra, dec= coord2rad('22h06m10.4s', '-18d35m39s') 
    a=2.1847
    b=-0.7118 
    c=0.1563 
    d=-0.0215
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_3C446():
    """
    name=3C446    epoch=2016 ra=22h25m47.2s dec=-04d57m01s a=3.8594
    b=-1.8462 c=0.3468 d=-0.0192
    """
    ra, dec= coord2rad('22h25m47.2s', '-04d57m01s') 
    a=3.8594
    b=-1.8462 
    c=0.3468 
    d=-0.0192
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func

def src_CTA102():
    """
    name=CTA102   epoch=2016 ra=22h32m36.4s dec=+11d43m51s a=-2.0888
    b=2.9478 c=-0.9102 d=0.0855
    """
    ra, dec= coord2rad('22h25m47.2s', '-04d57m01s') 
    a=3.8594
    b=-1.8462 
    c=0.3468 
    d=-0.0192
    func = lambda nu:logpoly(nu, a, b, c, d)
    return ra, dec, func
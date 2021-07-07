import numpy as np
import sympy as sp
from sympy import Rational
from copy import deepcopy

import sys
from os.path import realpath, dirname

full_path = realpath(__file__)
dir_path = dirname(full_path)

sys.path.append(dir_path+'/..')

from cosmology.redshift import redshift_from_distance

## define physical units
cc = 299792458 ## m/s (exact by decree)
msun = 1.98855e+30 #Kg
G = 6.67430e-11 ## m^3/kg/s from <https://physics.nist.gov/cuu/Constants/index.html> 
rsun = msun*G/cc**2 ## m
Mpc = 3.1e22 ## m

## define symbols
eta, delta, M, c = sp.symbols('eta, delta, M, c')
chi_s, chi_a, kappa_s, kappa_a, Lamda, Lamda2 =\
 sp.symbols('chi_s, chi_a, kappa_s, kappa_a, Lamda, Lamda_2')
v, f = sp.symbols('v, f')
tc, phic, DL = sp.symbols('t_c, phi_c, D_L')

class CompactObject():
    '''
    mass: in units of Msun
    spin: dimensionless
    '''
    def __init__(self, mass, spin, Lamda=0.):
        self.mass = mass
        self.spin = spin
        self.kappa = 1.
        self.Lamda = Lamda


def amplitude_coefficients():
    '''v= (pi*M*f/cc)**(1/3)'''
    ## numerical factors
    pi = sp.pi
    ## spin corrections
    sigma = (8*eta - Rational(81,32))*chi_a**2 + (17*eta/8 - Rational(81,32))*chi_s**2\
            - Rational(81,16)*delta*chi_s*chi_a
    
    gamma = (Rational(285197,16128) - 1579*eta/4032)*delta*chi_a +\
            (Rational(285197,16128) - 15317*eta/672 - 2227*eta**2/1008)*chi_s
    
    xi = (Rational(1614569,64512) - 1873643*eta/16128 + 2167*eta**2/42)*chi_a**2 + \
         (31*pi/12 - 7*pi*eta/3)*chi_s +\
         (Rational(1614569,64512) - 61392*eta/1344 + 57451*eta**2/4032)*chi_s**2 +\
         (31*pi/12 + (Rational(1614569,32256) - 165961*eta/2688)*chi_s)*delta*chi_a
    
    ## output
    out = []
    out.append(1)
    out.append(0)
    out.append(-Rational(323,224) + 451*eta/168)
    out.append((27*delta*chi_a/8 + (Rational(27,8) - 11*eta/6)*chi_s))
    out.append(-Rational(27312085,8128512) - 1975055*eta/338688 + 105271*eta**2/24192 + sigma)
    out.append(-85*pi/64 + 85*pi*eta/16 + gamma)
    out.append(-Rational(177520268561,8583708672) + (Rational(545384828789,5007163392)\
             - 205*pi**2/48)*eta - 3248849057*eta**2/178827264 + 34473079*eta**3/6386688 + xi)
    return out


def phase_coefficients():
    '''v= (pi*M*f/cc)**(1/3)'''
    ## numerical factors
    pi = sp.pi
    gE = sp.EulerGamma
    ## spin corrections
    beta = Rational(113,3)*delta*chi_a + (113-76*eta)/3*chi_s

    sigma = - 5*chi_s**2/8*(1 + 156*eta + 80*delta*kappa_a + 80*(1-2*eta)*kappa_s)\
        - 5*chi_a**2/8*((1-160*eta) + 80*delta*kappa_a + 80*(1-2*eta)*kappa_s)\
        - 5*chi_s*chi_a/4*(delta + 80*(1-2*eta)*kappa_a + 80*delta*kappa_s)

    gamma = chi_a*delta*(-Rational(732985,2268) - 140*eta/9)\
            + chi_s*(-Rational(732985,2268) + 24260*eta/81 + 340*eta**2/9)

    xi = sp.pi*( 2270*chi_a*delta/3 + chi_s*(Rational(2270,3) - 520*eta) )\
         + chi_s**2*( -Rational(1344475,2016) + 829705*eta/504\
        + 3415*eta**2/9\
        + delta*kappa_a*(Rational(26015,28) - 1495*eta/6)\
        + kappa_s*(Rational(26015,28) - 44255*eta/21 - 240*eta**2) )\
        + chi_a**2*( -Rational(1344475,2016) + 267815*eta/252 - 240*eta**2\
        + delta*kappa_a*(Rational(26015,28) - 1495*eta/6)\
        + kappa_s*(Rational(26015,28) - 44255*eta/21 - 240*eta**2) )\
        + chi_s*chi_a*( kappa_a*(Rational(26015,14) - 88510*eta/21 - 480*eta**2)\
        + delta*(-Rational(1344475,1008) + 745*eta/18\
        + kappa_s*(Rational(26015,14) - 1495*eta/3)) )

    zeta = delta*chi_a*( -Rational(25150083775,3048192) + 26804935*eta/6048\
        - 1985*eta**2/48 ) + chi_s*( -Rational(25150083775,3048192)\
        + 10566655595*eta/762048 - 1042165*eta**2/3024 + 5345*eta**3/36 )\
        + chi_s**3*( Rational(265,24) + 4035*eta/2 - 20*eta**2/3\
        + (Rational(3110,3) - 10250*eta/3 + 40*eta**2)\
        - 440*(1-3*eta) )\
        + chi_a**3*( delta*(Rational(265,24) - 2070*eta\
        + (Rational(3110,3) - 750*eta) - 440*(1-eta)) )\
        + chi_s**2*chi_a*( delta*(Rational(265,8) + 12055*eta/6\
        + (3110 - 10310*eta/3) - 1320*(1-eta)) )\
        + chi_s*chi_a**2*( Rational(265,8) - 6500*eta/3 + 40*eta**2\
        + (3110 - 27190*eta/3 + 40*eta**2)\
        - 1320*(1-3*eta) )
    
    ## output
    out = []
    out.append(1.)
    out.append(0.)
    out.append(Rational(3715,756) + 55*eta/9)
    ## leading SO
    out.append(-16*pi + beta)
    ## leading SS
    out.append(Rational(15293365,508032) + 27145*eta/504 + 3085*eta**2/72 + sigma)
    out.append((38645*pi/756 - 65*pi*eta/9 + gamma)*(1+3*sp.log(v)))
    out.append(Rational(11583231236531,4694215680) - 6848*gE/21 -\
        640*pi**2/3 + (2255*pi**2/12 - Rational(15737765635,3048192))*eta +\
        76055*eta**2/1728 - 127825*eta**3/1296 + xi - Rational(6848,63)*sp.log(64*v**3))
    out.append( 77096675*pi/254016 + 378515*pi/1512*eta\
        -74045*pi/756*eta**2 + zeta )
    return out


class TaylorF2():

    def __init__(self,obj1,obj2,DL=100,tc=0,phic=0,redshift=False):
        if redshift:
            self.redshift = redshift_from_distance(DL)
        else:
            self.redshift = 0.
        self.keys = ['t_c','phi_c','M_c','eta','chi_s','chi_a']
        self.__dict__['t_c'] = tc
        self.__dict__['phi_c'] = phic
        self.__dict__['D_L'] = DL*Mpc
        self.__dict__['M'] = (obj1.mass + obj2.mass)*rsun*(1+self.redshift)
        self.__dict__['kappa_s'] = 0.5*(obj1.kappa + obj2.kappa)
        self.__dict__['kappa_a'] = 0.5*(obj1.kappa - obj2.kappa)
        self.__dict__['eta'] = obj1.mass*obj2.mass/(obj1.mass + obj2.mass)**2
        #self.__dict__['mu'] = self.__dict__['eta']*self.__dict__['M']
        self.__dict__['M_c'] = self.__dict__['M']*self.__dict__['eta']**0.6*(1+self.redshift)
        #self.__dict__['delta'] = (obj1.mass - obj2.mass)/(obj1.mass + obj2.mass)
        self.__dict__['q'] = obj1.mass / obj2.mass
        self.__dict__['chi_s'] = 0.5*(obj1.spin +  obj2.spin)
        self.__dict__['chi_a'] = 0.5*(obj1.spin - obj2.spin)
        self.__dict__['Lamda'] = 16/13*((1+12/self.__dict__['q'])*(obj1.mass**5)*obj1.Lamda +\
                (1+12*self.__dict__['q'])*(obj2.mass**5)*obj2.Lamda)/(obj1.mass+obj2.mass)**5
        self.__dict__['Lamda_2'] = 16/4361*( (-919+3179*(1+self.__dict__['q'])/self.__dict__['q']-\
                2286*self.__dict__['q']/(1+self.__dict__['q'])+\
                260*(self.__dict__['q']/(1+self.__dict__['q']))**2)*(obj1.mass**5)*obj1.Lamda +\
                (-919+3179*(1+self.__dict__['q'])-2286/(1+self.__dict__['q'])+\
                260/(1+self.__dict__['q'])**2)*(obj2.mass**5)*obj2.Lamda )/(obj1.mass+obj2.mass)**5
        if self.__dict__['Lamda']:
            self.tidal = True
            self.keys += ['Lamda','Lamda_2']
        else:
            self.tidal = False
        self.eval = False

    def ISCO(self,mode='static'):
        if mode == 'static':
            return cc/(self.M*6**1.5*np.pi)

    def frequency_from_obs_time(self,obs_time=1):
        ## return minimum frequency from observational time
        ## as per eq. (2.15) in
        ## https://arxiv.org/abs/gr-qc/0411129v2
        fmin = 4.149e-5*(obs_time)**(-3/8)*(self.M_c*1e-6)**(-5/8)
        return fmin

    def __call__(self,f):
        if not self.eval:
            ## update eval
            self.eval = False
            params = {k:self.__dict__[k] for k in self.keys}
            ## phase
            self.phase_eval = self.phase().subs(params)
            self.phase_eval = sp.lambdify('f',self.phase_eval,modules='numpy')
            ## amplitude
            self.amplitude_eval = self.amplitude().subs(params)
            self.amplitude_eval = sp.lambdify('f',self.amplitude_eval,modules='numpy')
        amplitude = self.amplitude_eval(f)
        phase = self.phase_eval(f)
        return amplitude*np.exp(1j*phase)

    def evaluate_Nabla(self,keys=None):
        self.Nabla = {}
        if not keys:
            keys = self.keys
        for parameter in keys:
            self.Nabla[parameter] = self.diff_(parameter)
        return None

    def diff_(self,parameter):
        keys = deepcopy(self.keys)
        keys.remove(parameter)
        params = {k:self.__dict__[k] for k in keys}
        ## phase
        ph_diff = self.phase().subs(params)
        ph_diff = sp.diff(ph_diff,parameter)
        ph_diff = ph_diff.subs({parameter:self.__dict__[parameter]})
        ph_diff = sp.lambdify('f',ph_diff,modules='numpy')
        ##
        out = lambda f: self(f)*ph_diff(f)
        return out

    def amplitude(self,PN=0):
        cfs = amplitude_coefficients()
        ## restrict to PN order
        out = 0
        for i in range(int(2*PN)+1):
            out += cfs[i]*v**i
        ## add normalization
        out *= sp.sqrt(sp.pi*eta)*v**Rational(-7,2)
        out *= sp.sqrt(Rational(5,24))
        out *= M**2/c/DL
        out = out.subs('v','(pi*M*f/c)**Rational(1,3)')
        ## change variables
        out = out.subs([('M','M_c/(eta)**Rational(3,5)'),('delta','sqrt(1-4*eta)')])
        ## replace numerical physical quantities
        out = out.subs([('c',cc),('D_L',self.__dict__['D_L']),\
                       ('kappa_a',self.__dict__['kappa_a']),\
                       ('kappa_s',self.__dict__['kappa_s'])])
        return out

    def phase(self,PN=3.5):
        cfs = phase_coefficients()
        ## restrict to PN order
        out = 0
        for i in range(int(2*PN)+1):
            out += cfs[i]*v**i
        ## add tidal terms
        if self.tidal == True:
            out += -Rational(39,2)*Lamda*v**10 -Rational(3115,64)*Lamda*v**12 + Rational(6595,364)*delta*Lamda2*v**12
        ## add normalization
        out *= 3/(128*eta*v**5)
        ## add constant phase terms
        out += -sp.pi/4 + 2*c*tc/M*v**3 - phic
        out = out.subs('v', '(pi*M*f/c)**Rational(1,3)')
        ## change variables
        out = out.subs([('M','M_c/(eta)**Rational(3,5)'),('delta','sqrt(1-4*eta)')])
        ## replace numerical physical quantities
        out = out.subs([('c',cc),('D_L',self.__dict__['D_L']),\
                       ('kappa_a',self.__dict__['kappa_a']),\
                       ('kappa_s',self.__dict__['kappa_s'])])
        return out

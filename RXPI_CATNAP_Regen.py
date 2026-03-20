# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:53:30 2026

@author: Maximilian Slavik
"""

import numpy as np
import scipy.interpolate
import CoolProp.CoolProp as CP
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import math
from math import pi, tan
from scipy.optimize import brentq
from scipy.optimize import root

from RXPI_CATNAP_Combustion import SolvePC, CombustionPerformance, ChamberTransport, TPRhoStag, Props_obj, Transport_obj






def RegenGeom(lfmin,wcmin,Rthroat,numchannels,twall,R):
    
    #NOTE: Constant rib height/thickness channels, throat is minimum width - size carefully for DFAM!
    
    # Will add support for variable height channels at a future time
    
    trib = (2*pi*(Rthroat + twall) - wcmin*numchannels)/numchannels
    
    def channelheight(z):
        return lfmin
    
    def channelwidth(z):
        widc = (2*pi*(R(z) + twall) - numchannels*trib)/numchannels
        
        return widc
    
    return channelheight,channelwidth, trib



    




def DittusB(z,mdotchannel,T1,P1,Regen_obj):
    
    channelheight = Regen_obj.channelheight
    
    channelwidth = Regen_obj.channelwidth
    
    fluid = Regen_obj.coolant
    
    channelheight1 = channelheight(z)
    
    channelwidth1 = channelwidth(z)
    
    mu = CP.PropsSI('viscosity','T',T1,'P',P1,fluid)
    
    Dh = 4*(channelwidth1*channelheight1)/(2*channelwidth1 + 2*channelheight1)
    
    Reynolds = Dh*mdotchannel/(channelheight1*channelwidth1*mu)
    
    if Reynolds < 2300:
        raise ValueError('Laminar flow probable!, Re < 2300')
        
    Prantl = CP.PropsSI('Prandtl','T',T1,'P',P1,fluid)
    
    k_fluid = CP.PropsSI('conductivity','T',T1,'P',P1,fluid)
    
    Nusselt = 0.023*(Reynolds**0.8)*(Prantl**0.4)
    
    hcoolant = Nusselt*k_fluid/Dh
    
    return hcoolant


def Gneilinski(z,mdotchannel,Regen_obj,T1,P1):
    
    channelheight = Regen_obj.channelheight
    
    channelwidth = Regen_obj.channelwidth
    
    fluid = Regen_obj.coolant
    
    epsilon_rough = Regen_obj.epsilon_rough
    
    channelheight1 = channelheight(z)
    
    channelwidth1 = channelwidth(z)
    
    mu = CP.PropsSI('viscosity','T',T1,'P',P1,fluid)
    
    Dh = 4*(channelwidth1*channelheight1)/(2*channelwidth1 + 2*channelheight1)
    
    Reynolds = Dh*mdotchannel/(channelheight1*channelwidth1*mu)
    
    if Reynolds < 2300:
        raise ValueError('Laminar flow probable!, Re < 2300')
    
    Prantl = CP.PropsSI('Prandtl','T',T1,'P',P1,fluid)
    
    k_fluid = CP.PropsSI('conductivity','T',T1,'P',P1,fluid)
    
    f = 0.25*((np.log10(epsilon_rough/(3.7*Dh) + 5.74/(Reynolds**0.9)))**(-2))
    
    Nusselt = ((f/8)*(Reynolds - 1000)*Prantl)/(1 + 12.7*((f/8)**0.5)*(Prantl**(2/3) - 1))
    
    
    hcoolant = Nusselt*k_fluid/Dh
    
    return hcoolant
    


def Etafin(z,hcoolant,Regen_obj):
    
    
    finthickness = Regen_obj.trib
    
    kcond_fin = Regen_obj.k_chamber
    
    channelheight = Regen_obj.channelheight
    channelheight1 = channelheight(z)
    
    msqr = 2*hcoolant/(kcond_fin*finthickness)
    
    m = np.sqrt(msqr)
    
    etafin = (np.tanh(m*channelheight1))/(m*channelheight1)

    
    return etafin


 
def bartz_hg(z, Mach, R, Dt, rc, Pc, cstar, mu, cp, Pr, gamma,Tcomb,TempsC):
    """
    Computes the hot-gas side heat transfer coefficient using the Bartz correlation.
    
    Parameters
    ----------
    z : float
        Axial position along the engine [m]
    Mach : callable
        Mach number as a function of z, Mach(z) [-]
    R : callable
        Local inner wall radius as a function of z, R(z) [m]
    Dt : float
        Throat diameter [m]
    rc : float
        Radius of curvature at the throat [m]
    Pc : float
        Chamber pressure [Pa]
    cstar : float
        Characteristic exhaust velocity [m/s]
    mu : callable
        Dynamic viscosity of combustion gas as a function of z, mu(z) [Pa·s]
    cp : callable
        Specific heat at constant pressure as a function of z, cp(z) [J/kg·K]
    Pr : callable
        Prandtl number of combustion gas as a function of z, Pr(z) [-]
    gamma : callable
        Ratio of specific heats as a function of z, gamma(z) [-]
    T0 : float
        Stagnation (chamber) temperature [K]
    Tw : float
        Wall temperature on the hot-gas side [K]
    
    Returns
    -------
    h_g : float
        Hot-gas side heat transfer coefficient [W/m²·K]
    
    """
    
    M   = Mach(z)
    g   = gamma(z)
    mu_ = mu(z)      
    cp_ = cp(z)  # J/kg-K
    Pr_ = Pr(z)

    T0 = Tcomb

    Tw, _ = TempsC(z,Mach)

    Rad = R(z)
    
    A_ = pi*(Rad**2)
    
    At = 0.25*pi*(Dt**2)
 
    mach_term = 1.0 + (g - 1.0) / 2.0 * (M**2)
 
    sigma = (0.5 * (Tw / T0) * mach_term + 0.5)**(-0.68) \
            * mach_term**(-0.12)
 
    h_g = (
        (0.026 / Dt**0.2)
        * (mu_**0.2 * cp_ / Pr_**0.6)
        * (Pc / cstar)**0.8
        * (Dt / rc)**0.1
        * (At / A_)**0.9
        * sigma
    )

    return h_g



def Resistances(z,hc,hg,R,Regen_obj):
    
    channelheight = Regen_obj.channelheight

    rad_inner = R(z)

    k_chamber = Regen_obj.k_chamber

    dz = Regen_obj.dz

    twall = Regen_obj.twall

    trib = Regen_obj.trib

    numchannels = Regen_obj.numchannels

    channelheight1 = channelheight(z)
    
    Rg = 1/(hg*2*pi*rad_inner*dz)
    
    Rwall = np.log((rad_inner + twall)/rad_inner)/(2*pi*k_chamber*dz)
    
    etafin = Etafin(z, hc, Regen_obj)
    
    Abase = (2*pi*(rad_inner + twall) - numchannels*trib)*dz
    
    Atotal = Abase + 2*numchannels*channelheight1*dz
    
    etanaught = (1 - ((2*numchannels*channelheight1*dz)/(Atotal))*(1 - etafin))
    
    Rc = 1/(etanaught*hc*Atotal)
    
    Rtotal = Rg + Rwall + Rc
    
    return Rtotal,Rg,Rwall,Rc



def DeltaP(z,mdotchannel,T1,P1,Regen_obj):

    channelheight = Regen_obj.channelheight

    channelwidth = Regen_obj.channelwidth

    fluid = Regen_obj.coolant

    epsilon_rough = Regen_obj.epsilon_rough

    dz = Regen_obj.dz
    
    channelheight1 = channelheight(z)
    channelheight2 = channelheight(z + dz)
    
    channelwidth1 = channelwidth(z)
    channelwidth2 = channelwidth(z + dz)
    

    mu = CP.PropsSI('viscosity','T',T1,'P',P1,fluid)
    
    Dh = 4*(channelwidth1*channelheight1)/(2*channelwidth1 + 2*channelheight1)
    
    Reynolds = Dh*mdotchannel/(channelheight1*channelwidth1*mu)
    
    if Reynolds < 2300:
        raise ValueError('Laminar flow probable!, Re < 2300')
    
    f = 0.25*((np.log10(epsilon_rough/(3.7*Dh) + 5.74/(Reynolds**0.9)))**(-2))
    
    rho = CP.PropsSI('D','T',T1,'P',P1,fluid)
    
    u = mdotchannel/(rho*channelwidth1*channelheight1)
    
    u2 = mdotchannel/(rho*channelwidth2*channelheight2)
    
    darcydP = -(f/Dh)*(0.5*rho*(u**2))*dz
    
    areadP = 0.5*(rho)*(u**2 - u2**2)        
    
    DeltaP = darcydP + areadP
    
    return DeltaP





def RectStress(z,twall,Q,Rwall):

    pass




class Regen_obj:
    def __init__(self, twall, lfmin, wcmin,R, Rthroat, throat_radcurv, numchannels, coolant, k_chamber, epsilon_rough,numpts_z,enginelength,genangle):

        self.genrad = (genangle)*pi/180
        
        self.twall = twall
        self.R = R #function R(z)
        self.throat_radcurv = throat_radcurv
        self.numchannels = numchannels
        self.Rthroat = Rthroat
        self.coolant = coolant
        self.k_chamber = k_chamber
        self.epsilon_rough = epsilon_rough
        self.numpts_z = numpts_z
        self.channelheight, self.channelwidth, self.trib = RegenGeom(lfmin,wcmin,Rthroat,numchannels,twall,R)
        self.z_array = np.linspace(0,enginelength,numpts_z)
        self.dz = (enginelength/numpts_z)/np.cos(self.genrad)

    


    def BalanceEnth(self,z,mdot_coolant,T1,P1,Transport_obj,Correlation_flag='Gneil'):

        MachArea = lambda z: Transport_obj.Mach(z, self.R)

        Cptransport, viscositytransport, thermalcondtransport, prantltransport, gammatransport = Transport_obj.Chambertransport()

        cstar = Transport_obj.getCstar()

        Pc = Transport_obj.Pc

        Tcomb = Transport_obj.Tcomb()

        TempsC,_,_,_ = Transport_obj.TPRhostag()

        mdotchannel = mdot_coolant/self.numchannels

        hg = bartz_hg(z,MachArea,self.R,2*self.Rthroat,self.throat_radcurv,Pc,cstar,viscositytransport,
                      Cptransport,prantltransport,gammatransport,Tcomb,TempsC)


        hcGneil = Gneilinski(z, mdotchannel, self, T1, P1)

        hcDB = DittusB(z, mdotchannel, T1, P1, self)
        
        Taw,_ = TempsC(z,MachArea)

        if Correlation_flag == 'DB':
        
            Rtotal,_,_,_ = Resistances(z,hcDB,hg,self.R,self)

        if Correlation_flag == 'Gneil':
        
            Rtotal,_,_,_ = Resistances(z,hcGneil,hg,self.R,self)

        Cpcool = CP.PropsSI('CPMASS','T',T1,'P',P1,Transport_obj.Props_obj.fuel)
        
        deltaTc = (Taw - T1)/(mdot_coolant*(Cpcool)*Rtotal)

        T2 = T1 + deltaTc

        return T2
    
    def DeltaPstep(self,z,T1,P1,mdot_coolant):

        mdot_channel = mdot_coolant/self.numchannels

        Deltap = DeltaP(z,mdot_channel,T1,P1,self)

        return Deltap

    '''
    def SOLVE_REGEN(self,mdot_coolant,Tcool_init,Pcool_init,Transport_obj):

        T1 = Tcool_init

        P1 = Pcool_init

        z1 = 0.0

        Tcool_array = np.zeros(self.numpts_z)
        Pcool_array = np.zeros(self.numpts_z)

        for i in range(self.numpts_z):
            T2 = self.BalanceEnth(z1,mdot_coolant,T1,P1,Transport_obj)

            P2 = P1 + self.DeltaPstep(z1,T1,P1,mdot_coolant)

            Tcool_array[i] = T2
            Pcool_array[i] = P2

            T1 = T2
            P1 = P2
            z1 = z1 + self.dz


        return Tcool_array, Pcool_array

        '''
    
    def dP_channel_Approx(self,Tc_init,Pcool_init,mdot_coolant):

        #NOTE: Do not use for normal regen circuit calculation, this loop approximates regen channel loss
        # without accounting for small changes in coolant properties due to temperature. Meant to be used 
        # outside of snapshot loop

        P1 = Pcool_init
        
        for i,z1 in enumerate(self.z_array):
            mdotchannel = mdot_coolant / self.numchannels

            P2 = P1 + self.DeltaPstep(z1,Tc_init,P1,mdot_coolant)

            z1 = z1 + self.dz

            P1 = P2

        return P2 - Pcool_init
    

    def SOLVE_REGEN(self, mdot_coolant, Tcool_init, Pcool_init, Transport_obj):
        T1 = Tcool_init
        P1 = Pcool_init

        numz = self.numpts_z
        Tcool_array = np.zeros(numz)
        Pcool_array = np.zeros(numz)
        hg_array    = np.zeros(numz)
        Twall_array = np.zeros(numz)
        Qflux_array = np.zeros(numz)

        TempsC, _, _, _ = Transport_obj.TPRhostag()
        cstar  = Transport_obj.getCstar()
        Pc     = Transport_obj.Pc
        Tcomb  = Transport_obj.Tcomb()
        Cptransport, viscositytransport, _, prantltransport, gammatransport = Transport_obj.Chambertransport()
        MachFunc = lambda z: Transport_obj.Mach(z, self.R)

        for i, z1 in enumerate(self.z_array):
            mdotchannel = mdot_coolant / self.numchannels

            hg = bartz_hg(z1, MachFunc, self.R, 2*self.Rthroat, self.throat_radcurv,
                        Pc, cstar, viscositytransport, Cptransport,
                        prantltransport, gammatransport, Tcomb, TempsC)

            hcDB = DittusB(z1, mdotchannel, T1, P1, self)
            
            Rtotal, Rg, Rwall, Rc = Resistances(z1, hcDB, hg, self.R, self)

            Taw, _ = TempsC(z1, MachFunc)

            Q = (Taw - T1) / Rtotal                      

            Twall = T1 + Q * (Rwall + Rc)                           

            Cpcool = CP.PropsSI('CPMASS', 'T', T1, 'P', P1, self.coolant)
            T2 = T1 + Q / (mdot_coolant * Cpcool)
            P2 = P1 + DeltaP(z1, mdotchannel, T1, P1, self)

            Tcool_array[i] = T2
            Pcool_array[i] = P2
            hg_array[i]    = hg
            Twall_array[i] = Twall
            Qflux_array[i] = Q / (2 * pi * self.R(z1) * self.dz)  # W/m² (this is approximate based on hotwall area)

            T1 = T2
            P1 = P2

        return Tcool_array, Pcool_array, hg_array, Twall_array, Qflux_array
    
    
        



        





# def transformdz(z,dz,gen_angles):
#     ### Only constant generatrix/helix angle channels supported for now, will add variability later
    
#     genanglelocal = gen_angles(z) #radians

#     dz = dz*(1/(np.cos(genanglelocal)))
    
    
    
    




        

    























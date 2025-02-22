import jax.numpy as jnp
import jax.scipy.special as jsp
from jax import jit, vmap
from functools import partial
from jax.lax import while_loop, fori_loop
from jax import config
config.update("jax_enable_x64", True)
from .Tools import grid, legendre
from .Channel import Channel
import time

import matplotlib.pyplot as plt

# import Christian's modules for the chiral potential
import sys
import os

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
chiral_modules_path = os.path.join(os.path.dirname(script_directory), "chiral", "modules")
sys.path.append(chiral_modules_path)

from Potential import chiralms, chiralms_affine
from Channel import Channel as CD_Channel

'''
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
print(script_directory)
sys.path.append(os.path.join(script_directory, "chiral"))

#import chiralPot
'''

class Potential:

    # the potential has units of MeV^-2 AFTER being multiplied by LECs


    def __init__(self, config):
    
        self.chan = Channel(config)
        self.hbarc = config.hbarc
        self.Nc = self.chan.Nsingle
        self.Ncc = self.chan.Ncoupled
        self.aux = False  # if potential has constant operator
        
        
        # choose potential
        if config.pot == 'chiral':
        
            self.Voc = self.Voc_Chiral
            self.No = 12        # there are 11 LECs plus one auxiliary variable
            self.Naux = True    # one auxiliary (constant) variable, always assumed to be first operator
            self.potargs = {"label": "chiral", "kwargs": {"potId": 213}} # N2LO, R0=1.0 fm, SFR cutoff=1 GeV
            
            self.CS = 5.43850     # fm^2
            self.CT = 0.27672     # fm^2
            self.C1 = -0.14084    # fm^4
            self.C2 = 0.04243     # fm^4
            self.C3 = -0.12338    # fm^4
            self.C4 = 0.11018     # fm^4
            self.C5 = -2.11254    # fm^4
            self.C6 = 0.15898     # fm^4
            self.C7 = -0.26994    # fm^4
            self.CNN = 0.04344    # fm^2
            self.CPP = 0.062963   # fm^2
            
            self.LECs = jnp.array([1.0, self.CS, self.CT, self.C1, self.C2, self.C3, self.C4,
                                   self.C5, self.C6, self.C7, self.CNN, self.CPP])
                
            self.chan.print_single_channels()
            self.chan.print_coupled_channels()
            
            self.compute_single = True if self.chan.Nsingle > 0 else False
            self.compute_coupled = True if self.chan.Ncoupled > 0 else False
            
            

                
            
        elif config.pot == 'mt':
        
            self.VR = 7.2910                   # 1
            self.VA = -3.1769                  # 1
            self.muR = 613.69                  # MeV
            self.muA = 305.86                  # MeV
            self.Voc = self.Voc_MalflietTjon   # MeV^-2
            self.LECs = jnp.array([self.VR, self.VA])
            self.No = 2
            self.Nx = config.Nx
            self.x, self.wx = grid(-1., 1., self.Nx)
                
            # precompute Pl(x) for all l in single channels
            self.compute_single = True if self.chan.Nsingle > 0 else False
            self.compute_coupled = False
            self.chan.print_single_channels()
            J, S, T, Tz, self.L = self.chan.single_channels
            self.Plx, _ = legendre(config.Jmax + 1, self.x)
            self.Plx = self.Plx[self.L]
            
        elif config.pot == 'mn':
        
            self.V0R = 200.0                                    # MeV
            self.V0S = -91.85                                   # MeV
            self.KR = 1.487                                     # fm^-2
            self.KS = 0.465                                     # fm^-2
            self.KR *= self.hbarc**2                            # MeV^2
            self.KS *= self.hbarc**2                            # MeV^2
            self.Voc = self.Voc_Minnesota                       # MeV^-3
            self.prefactorR = 0.5 / jnp.sqrt(jnp.pi * self.KR)  # MeV^-1
            self.prefactorS = 0.5 / jnp.sqrt(jnp.pi * self.KS)  # MeV^-1
            self.LECs = jnp.array([self.V0R, self.V0S])   # MeV
            self.No = 2
            
            if (config.channel != 'nn') or (config.Jmax != 0):
                raise ValueError('Minnesota potential is analytically decomposed for the 1S0 and 3P0 channels only.')
                
            else:
                self.chan.print_single_channels()
                self.compute_single = True if self.chan.Nsingle > 0 else False
                self.compute_coupled = False
            
       

    def Voc_Chiral(self, p1, p2=None, diag=False):
        
        # change momentum units from MeV to fm^-1
        p1 = p1 / self.hbarc
        
        if diag is True:
        
            if self.compute_single:
                Vocp = jnp.zeros((self.No, self.Nc, p1.shape[0]))
                J, S, T, Tz, L = self.chan.single_channels
                for c in range(self.Nc):
                    channel = CD_Channel(S=S[c], L=L[c], LL=L[c], J=J[c], channel=Tz[c])
                    for i in range(p1.shape[0]):
                        Vocp = Vocp.at[:,c,i].set( chiralms_affine(p1[i], p1[i], channel, **self.potargs["kwargs"]) )
            else:
                Vocp = None
                
            if self.compute_coupled:
                Voccp = jnp.zeros((4, self.No, self.Ncc, p1.shape[0]))
                J, S, T, Tz, L1, L2 = self.chan.coupled_channels
                for cc in range(self.Ncc):
                    print(f"{cc} S={S[cc]} L={L1[cc]} L'={L2[cc]} J={J[cc]} Tz={Tz[cc]}")
                    channel_mm = CD_Channel(S=S[cc], L=L1[cc], LL=L1[cc], J=J[cc], channel=Tz[cc])
                    channel_pp = CD_Channel(S=S[cc], L=L2[cc], LL=L2[cc], J=J[cc], channel=Tz[cc])
                    channel_mp = CD_Channel(S=S[cc], L=L1[cc], LL=L2[cc], J=J[cc], channel=Tz[cc])
                    channel_pm = CD_Channel(S=S[cc], L=L2[cc], LL=L1[cc], J=J[cc], channel=Tz[cc])
                    
                    for i in range(p1.shape[0]):
                        Voccp = Voccp.at[0,:,cc,i].set( chiralms_affine(p1[i], p1[i], channel_mm, **self.potargs["kwargs"]) )
                        Voccp = Voccp.at[1,:,cc,i].set( chiralms_affine(p1[i], p1[i], channel_pp, **self.potargs["kwargs"]) )
                        Voccp = Voccp.at[2,:,cc,i].set( chiralms_affine(p1[i], p1[i], channel_mp, **self.potargs["kwargs"]) )
                        Voccp = Voccp.at[3,:,cc,i].set( chiralms_affine(p1[i], p1[i], channel_pm, **self.potargs["kwargs"]) )
                        
            else:
                Voccp = None
                        
            return Vocp, Voccp
        
        else:
        
            if p2 is None:
            
                if self.compute_single:
                    Vocpp = jnp.zeros((self.No, self.Nc, p1.shape[0], p1.shape[0]))
                    J, S, T, Tz, L = self.chan.single_channels
                    for c in range(self.Nc):
                        channel = CD_Channel(S=S[c], L=L[c], LL=L[c], J=J[c], channel=Tz[c])
                        for i in range(p1.shape[0]):
                            for j in range(i, p1.shape[0]):
                                temp = chiralms_affine(p1[i], p1[j], channel, **self.potargs["kwargs"])
                                Vocpp = Vocpp.at[:,c,i,j].set( temp )
                                Vocpp = Vocpp.at[:,c,j,i].set( temp )
                                
                else:
                    Vocpp = None
                        
                if self.compute_coupled:
                    Voccpp = jnp.zeros((4, self.No, self.Ncc, p1.shape[0], p1.shape[0]))
                    J, S, T, Tz, L1, L2 = self.chan.coupled_channels
                    for cc in range(self.Ncc):
                    
                        channel_mm = CD_Channel(S=S[cc], L=L1[cc], LL=L1[cc], J=J[cc], channel=Tz[cc])
                        channel_pp = CD_Channel(S=S[cc], L=L2[cc], LL=L2[cc], J=J[cc], channel=Tz[cc])
                        channel_mp = CD_Channel(S=S[cc], L=L1[cc], LL=L2[cc], J=J[cc], channel=Tz[cc])
                        channel_pm = CD_Channel(S=S[cc], L=L2[cc], LL=L1[cc], J=J[cc], channel=Tz[cc])
                        
                        for i in range(p1.shape[0]):
                            for j in range(i, p1.shape[0]):
                            
                                V_mm = chiralms_affine(p1[i], p1[j], channel_mm, **self.potargs["kwargs"])
                                V_pp = chiralms_affine(p1[i], p1[j], channel_pp, **self.potargs["kwargs"])
                                V_mp = chiralms_affine(p1[i], p1[j], channel_mp, **self.potargs["kwargs"])
                                V_pm = chiralms_affine(p1[i], p1[j], channel_pm, **self.potargs["kwargs"])
                                
                                Voccpp = Voccpp.at[0,:,cc,i,j].set( V_mm )
                                Voccpp = Voccpp.at[0,:,cc,j,i].set( V_mm )
                                Voccpp = Voccpp.at[1,:,cc,i,j].set( V_pp )
                                Voccpp = Voccpp.at[1,:,cc,j,i].set( V_pp )
                                Voccpp = Voccpp.at[2,:,cc,i,j].set( V_mp )
                                Voccpp = Voccpp.at[2,:,cc,j,i].set( V_mp )
                                Voccpp = Voccpp.at[3,:,cc,i,j].set( V_pm )
                                Voccpp = Voccpp.at[3,:,cc,j,i].set( V_pm )
                                
                                
                else:
                    Voccpp = None
                
            
            else:
                                    
                # change momentum units from MeV to fm^-1
                p2 = p2 / self.hbarc

                if self.compute_single:
                    Vocpp = jnp.zeros((self.No, self.Nc, p1.shape[0], p2.shape[0]))
                    J, S, T, Tz, L = self.chan.single_channels
                    for c in range(self.Nc):
                        channel = CD_Channel(S=S[c], L=L[c], LL=L[c], J=J[c], channel=Tz[c])
                        for i in range(p1.shape[0]):
                            for j in range(p2.shape[0]):
                                Vocpp = Vocpp.at[:,c,i,j].set( chiralms_affine(p1[i], p2[j], channel, **self.potargs["kwargs"]) )
                                
                else:
                    Vocpp = None
                        
                if self.compute_coupled:
                    Voccpp = jnp.zeros((4, self.No, self.Ncc, p1.shape[0], p2.shape[0]))
                    J, S, T, Tz, L1, L2 = self.chan.coupled_channels
                    
                    for cc in range(self.Ncc):
                    
                        channel_mm = CD_Channel(S=S[cc], L=L1[cc], LL=L1[cc], J=J[cc], channel=Tz[cc])
                        channel_pp = CD_Channel(S=S[cc], L=L2[cc], LL=L2[cc], J=J[cc], channel=Tz[cc])
                        channel_mp = CD_Channel(S=S[cc], L=L1[cc], LL=L2[cc], J=J[cc], channel=Tz[cc])
                        channel_pm = CD_Channel(S=S[cc], L=L2[cc], LL=L1[cc], J=J[cc], channel=Tz[cc])
                        
                        for i in range(p1.shape[0]):
                            for j in range(p2.shape[0]):
                            
                                Voccpp = Voccpp.at[0,:,cc,i,j].set( chiralms_affine(p1[i], p2[j], channel_mm, **self.potargs["kwargs"]) )
                                Voccpp = Voccpp.at[1,:,cc,i,j].set( chiralms_affine(p1[i], p2[j], channel_pp, **self.potargs["kwargs"]) )
                                Voccpp = Voccpp.at[2,:,cc,i,j].set( chiralms_affine(p1[i], p2[j], channel_mp, **self.potargs["kwargs"]) )
                                Voccpp = Voccpp.at[3,:,cc,i,j].set( chiralms_affine(p1[i], p2[j], channel_pm, **self.potargs["kwargs"]) )

                        '''
                        # plot each operator
                        for o in range(self.No):
                            fig, ax = plt.subplots(2, 2, figsize=(8,8))
                            
                            ax[0,0].imshow(Voccpp[0,o,cc])
                            ax[1,1].imshow(Voccpp[1,o,cc])
                            ax[0,1].imshow(Voccpp[2,o,cc])
                            ax[1,0].imshow(Voccpp[3,o,cc])
                            
                            plt.show()
                        '''
                else:
                    Voccpp = None
            return Vocpp, Voccpp
            
            
        
    
        

    @partial(jit, static_argnames=['self', 'diag'])
    def Voc_Minnesota(self, p1, p2=None, diag=False):
        """
        Analytically decomposed for the 1S0 and 3P0 channels.
        """
        
        if diag is True:
        
            PP_inv = 1./p1**2     # MeV^-2
            
            exp_plus_R = jnp.exp(-p1**2 / self.KR)
            exp_plus_S = jnp.exp(-p1**2 / self.KS)
            
            V0p_R = self.prefactorR * PP_inv * (1. - exp_plus_R)
            V0p_S = self.prefactorS * PP_inv * (1. - exp_plus_S)
            
            V1p_R = self.prefactorR * PP_inv * (1. + exp_plus_R - 2. * self.KR * PP_inv * (1. - exp_plus_R))
            V1p_S = self.prefactorS * PP_inv * (1. + exp_plus_S - 2. * self.KS * PP_inv * (1. - exp_plus_S))
            
            Vcp_R = jnp.stack((V0p_R, V1p_R))
            Vcp_S = jnp.stack((V0p_S, V1p_S))
            
            return jnp.stack((Vcp_R, Vcp_S)), None
            
        else:
        
            if p2 is None:
            
                P1, P2 = jnp.meshgrid(p1, p1)
                
            else:
                P1, P2 = jnp.meshgrid(p2, p1)
            
                
            PP_inv = 1./(P1 * P2)
            exp_plus_R = jnp.exp(-0.25 * (P1 + P2)**2 / self.KR)
            exp_plus_S = jnp.exp(-0.25 * (P1 + P2)**2 / self.KS)
            exp_minus_R = jnp.exp(-0.25 * (P1 - P2)**2 / self.KR)
            exp_minus_S = jnp.exp(-0.25 * (P1 - P2)**2 / self.KS)
            
            V0pp_R = self.prefactorR * PP_inv * (exp_minus_R - exp_plus_R)
            V0pp_S = self.prefactorS * PP_inv * (exp_minus_S - exp_plus_S)
                
            V1pp_R = self.prefactorR * PP_inv * (exp_minus_R + exp_plus_R - 2. * self.KR * PP_inv * (exp_minus_R - exp_plus_R))
            V1pp_S = self.prefactorS * PP_inv * (exp_minus_S + exp_plus_S - 2. * self.KS * PP_inv * (exp_minus_S - exp_plus_S))

            Vcpp_R = jnp.stack((V0pp_R, V1pp_R))
            Vcpp_S = jnp.stack((V0pp_S, V1pp_S))

            
            return jnp.stack((Vcpp_R, Vcpp_S)), None


    #@partial(jit, static_argnames=['self', 'V_func', 'diag'])
    def PWD(self, V_func, p1, p2=None, diag=False):
        """
        Decomposes a potential as a function of Q_sq = (p1-p2)^2 in partial waves.
        Only compatible with single channels.
        """
    
        # ignore p2 for diagonal case
        if diag is True:
        
            P, X = jnp.meshgrid(p1, self.x)
            Q_sq = 2. * P**2 * (1. - X)
            Vlp = jnp.einsum('lx,x,xi->li', self.Plx, self.wx, V_func(Q_sq))
            return Vlp / jnp.pi
            
        else:
        
            # take advantage of p1 <-> p2 symmetry
            if p2 is None:
            
                P1, P2, X = jnp.meshgrid(p1, p1, self.x)
                i, j = jnp.triu_indices(p1.shape[0])
                P1 = P1[i,j,:]
                P2 = P2[i,j,:]
                X = X[i,j,:]
                Q_sq = P1**2 + P2**2 - 2. * P1 * P2 * X
                Vlp = jnp.einsum('lx,x,ix->li', self.Plx, self.wx, V_func(Q_sq))
                Vlpp = jnp.zeros((self.Nc, p1.shape[0], p1.shape[0]))
                Vlpp = Vlpp.at[:,i,j].set(Vlp)
                Vlpp = Vlpp.at[:,j,i].set(Vlp)
                
            # compute for generic p1 and p2
            else:
                P1, P2, X = jnp.meshgrid(p2, p1, self.x)
                Q_sq = P1**2 + P2**2 - 2. * P1 * P2 * X
                Vlpp = jnp.einsum('lx,x,ijx->lij', self.Plx, self.wx, V_func(Q_sq))
            
            return Vlpp / jnp.pi

    #@partial(jit, static_argnames=['self', 'diag'])
    def Voc_MalflietTjon(self, p1, p2=None, diag=False):
    
        def V1_func(Q_sq):
            return 1. / (Q_sq + self.muR**2)
        
        def V2_func(Q_sq):
            return 1. / (Q_sq + self.muA**2)
            
        Vc1 = self.PWD(V1_func, p1, p2=p2, diag=diag)
        Vc2 = self.PWD(V2_func, p1, p2=p2, diag=diag)
        
        return jnp.stack((Vc1, Vc2)), None

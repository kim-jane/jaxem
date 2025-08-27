import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from typing import Optional

# import Christian's modules for the chiral potential
import sys
import os

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
chiral_modules_path = os.path.join(os.path.dirname(script_directory), "chiral", "modules")
sys.path.append(chiral_modules_path)

from Potential import chiralms, chiralms_affine
from Channel import Channel as CD_Channel


class Potential:

    """
    Base class for nucleon-nucleon potentials.
    Outputs potentials in units of fm^2 after being multiplied with LECs. 
    """
    
    def __init__(
        self,
        hbarc: float,      # MeV fm
        mass: float,       # MeV
        name: str
    ):
    
        self.hbarc = hbarc
        self.mass = mass
        self.name = name
        
        
        

class Chiral(Potential):

    """
    N2LO Chiral Potential from Gazerlis, Tews, et al.
    """
    
    def __init__(
        self,
        hbarc: float = 197.327,         # MeV fm
        mass: float = 938.918755,       # MeV (default: avg nucleon mass)
    ):
        super().__init__(hbarc, mass, "chiral")
        
        self.potId = 213            # N2LO, R0=1.0 fm, SFR cutoff=1 GeV
        self.CS = 5.43850           # fm^2
        self.CT = 0.27672           # fm^2
        self.C1 = -0.14084          # fm^4
        self.C2 = 0.04243           # fm^4
        self.C3 = -0.12338          # fm^4
        self.C4 = 0.11018           # fm^4
        self.C5 = -2.11254          # fm^4
        self.C6 = 0.15898           # fm^4
        self.C7 = -0.26994          # fm^4
        self.CNN = 0.04344          # fm^2
        self.CPP = 0.062963         # fm^2
        self.n_operators = 12       # extra operator is auxiliary
        self.factor = 2.0/jnp.pi    # normalization convention
        
        self.LECs = jnp.array([1.0, self.CS, self.CT, self.C1, self.C2, self.C3, self.C4,
                               self.C5, self.C6, self.C7, self.CNN, self.CPP])
        

    def single_channel_operators(
        self,
        q1: jnp.ndarray,
        q2: Optional[jnp.ndarray] = None,
        diag: Optional[bool] = False,
        J: int = 0,
        S: int = 0,
        T: int = 1,
        Tz: int = 0,
        L: int = 0
    ):
        channel = CD_Channel(S=S, L=L, LL=L, J=J, channel=Tz)
        chiral = lambda qi, qj: chiralms_affine(qi, qj, channel, potId=self.potId)
    
        if diag is True:
            n_q = q1.shape[0]
            operators = jnp.zeros((n_q, self.n_operators))
            for i in range(n_q):
                operators = operators.at[i].set( chiral(q1[i], q1[i]) )
            
        else:
            if q2 is None:
                n_q = q1.shape[0]
                operators = jnp.zeros((n_q, n_q, self.n_operators))
                for i in range(n_q):
                    for j in range(i, n_q):
                        operators = operators.at[i,j].set( chiral(q1[i], q1[j]) )
                        operators = operators.at[j,i].set( operators[i,j] )
                        
            else:
                n_q1, n_q2 = q1.shape[0], q2.shape[0]
                operators = jnp.zeros((n_q1, n_q2, self.n_operators))
                for i in range(n_q1):
                    for j in range(n_q2):
                        operators = operators.at[i,j].set( chiral(q1[i], q2[j]) )
                
        return operators * self.hbarc**2 # fm^2 once multiplied by LECs
    

    def coupled_channel_operators(
        self,
        q1: jnp.ndarray,
        q2: Optional[jnp.ndarray] = None,
        diag: Optional[bool] = False,
        J: int = 1,
        S: int = 1,
        T: int = 0,
        Tz: int = 0,
        L1: int = 0,
        L2: int = 2
    ):
        channel_mm = CD_Channel(S=S, L=L1, LL=L1, J=J, channel=Tz)
        channel_pp = CD_Channel(S=S, L=L2, LL=L2, J=J, channel=Tz)
        channel_mp = CD_Channel(S=S, L=L1, LL=L2, J=J, channel=Tz)
        chiral_mm = lambda qi, qj: chiralms_affine(qi, qj, channel_mm, potId=self.potId)
        chiral_pp = lambda qi, qj: chiralms_affine(qi, qj, channel_pp, potId=self.potId)
        chiral_mp = lambda qi, qj: chiralms_affine(qi, qj, channel_mp, potId=self.potId)
    
        if diag is True:
        
            n_q = q1.shape[0]
            operators = jnp.zeros((2, 2, n_q, self.n_operators))
            for i in range(n_q):
            
                operators = operators.at[0,0,i].set( chiral_mm(q1[i], q1[i]) )
                operators = operators.at[1,1,i].set( chiral_pp(q1[i], q1[i]) )
                
                temp = chiral_mp(q1[i], q1[i])
                operators = operators.at[0,1,i].set( temp )
                operators = operators.at[1,0,i].set( temp )
            
        else:
            if q2 is None:
                n_q = q1.shape[0]
                operators = jnp.zeros((2, 2, n_q, n_q, self.n_operators))
                for i in range(n_q):
                    for j in range(i, n_q):
                    
                        temp = chiral_mm(q1[i], q1[j])
                        operators = operators.at[0,0,i,j].set( temp )
                        operators = operators.at[0,0,j,i].set( temp )
                        
                        temp = chiral_pp(q1[i], q1[j])
                        operators = operators.at[1,1,i,j].set( temp )
                        operators = operators.at[1,1,j,i].set( temp )
                        
                        temp = chiral_mp(q1[i], q1[j])
                        operators = operators.at[0,1,i,j].set( temp )
                        operators = operators.at[1,0,j,i].set( temp )
                        
                        temp = chiral_mp(q1[j], q1[i])
                        operators = operators.at[1,0,i,j].set( temp )
                        operators = operators.at[0,1,j,i].set( temp )
                        
            else:
                n_q1, n_q2 = q1.shape[0], q2.shape[0]
                operators = jnp.zeros((2, 2, n_q1, n_q2, self.n_operators))
                for i in range(n_q1):
                    for j in range(n_q2):
                        operators = operators.at[0,0,i,j].set( chiral_mm(q1[i], q2[j]) )
                        operators = operators.at[1,1,i,j].set( chiral_pp(q1[i], q2[j]) )
                        operators = operators.at[0,1,i,j].set( chiral_mp(q1[i], q2[j]) )
                        operators = operators.at[1,0,i,j].set( chiral_mp(q2[j], q1[i]) )
                
        return operators * self.hbarc**2 # fm^2 once multiplied by LECs

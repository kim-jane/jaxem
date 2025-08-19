import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
import time
from .channels import Channels
from .mesh import Mesh
from .potential import Potential
from typing import Optional, Dict, Tuple
import matplotlib.pyplot as plt
import glob


class Solver:

    def __init__(
        self,
        mesh: Mesh,
        channels: Channels,
        potential: Potential,
        write_potential: bool = True,
        load_potential: bool = True
    ):
    
        self.mesh = mesh
        self.channels = channels
        self.potential = potential
        self.write_potential = write_potential
        self.load_potential = load_potential
        
            
    def t(
        self,
        channel: str = '1S0',
        Elab: float = 10.0,
        LECs: jnp.ndarray = None
    ) -> jnp.ndarray:
    
        if LECs is None:
            LECs = self.potential.LECs
            
            
        if channel in self.channels.single:
            
            VGqqo, Vqo = self.setup_single_channel(channel, Elab)
            return self.solve_single_channel(LECs, VGqqo, Vqo)
            
            
        elif channel in self.channels.coupled:
        
            VGqqo, Vqo = self.setup_coupled_channel(channel, Elab)
            return self.solve_coupled_channel(LECs, VGqqo, Vqo)
            
        else:
            raise ValueError(f"Channel {channel} not found.")
            
            
        
    def total_cross_section(
        self,
        Elab
    ):
        sigma_tot = 0.0
    
        for channel in self.channels.all:
        
            sigma_tot += self.scattering_params(channel, Elab)[2]
            
        return sigma_tot
            
            

    def scattering_params(
        self,
        channel,
        Elab,
        Tq: jnp.ndarray = None
    ):
    
        if Tq is None:
            Tq = self.t(channel, Elab)
    
        k = self.momentum_pole(Elab) # fm^-1
        m = self.potential.mass # MeV
        f = self.potential.factor # 1
        
        if channel in self.channels.single:
            
            # extract on-shell element of T-matrix
            T = Tq[0] # fm^2
            
            # convert to S-matrix
            S = 1 - 1j * jnp.pi * f * m * k * T / self.potential.hbarc # 1
            delta = 0.5 * jnp.arctan2(S.imag, S.real)
            eta = jnp.abs(S)
        
            J = self.channels.single[channel]["J"]
            sigma = 0.5 * jnp.pi * (2*J+1) * (1-S).real / k**2 # fm^2
            
            return delta, eta, sigma
            
            
        elif channel in self.channels.coupled:
            
            T = Tq[:,:,0]
            J = self.channels.coupled[channel]["J"]
            S = jnp.identity(2) - 1j * jnp.pi * f * m * k * T / self.potential.hbarc

            Z = 0.5 * ( S[0,1] + S[1,0] ) / jnp.sqrt(S[0,0] * S[1,1])
            epsilon = -0.25 * 1j * jnp.log( (1 + Z) / (1 - Z) )
            epsilon = epsilon.real
            
            if J == 1:
                epsilon = jnp.where(epsilon < -1e-8, jnp.abs(epsilon), epsilon)
            
            S_minus = S[0,0] / jnp.cos(2 * epsilon)
            S_plus = S[1,1] / jnp.cos(2 * epsilon)
            
            delta_minus = 0.5 * jnp.arctan2(S_minus.imag, S_minus.real)
            delta_plus = 0.5 * jnp.arctan2(S_plus.imag, S_plus.real)
            delta_minus = jnp.where(delta_minus < -1e-8, jnp.pi + delta_minus, delta_minus)
            
            eta_minus = jnp.abs(S_minus)
            eta_plus = jnp.abs(S_plus)
            
            sigma = 0.5 * jnp.pi * (2*J+1) * (2 - S_minus - S_plus).real / k**2
            
            return (delta_minus, delta_plus, epsilon), (eta_minus, eta_plus), sigma
            
        else:
            raise ValueError(f"Channel {channel} not found.")
        

        
        
    def setup_single_channel(
        self,
        channel: str,
        Elab: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns the operator-decomposed tensors (VGqqo, Vqo) for a given channel and energy.
        """
        Gq = self.propagator(Elab)
        Vqqo = self.potential_operators_single(channel, Elab)
        VGqqo = Vqqo * Gq[None,:,None]
        return (VGqqo, Vqqo[:,0,:])
        
        
        

    def setup_coupled_channel(
        self,
        channel: str,
        Elab: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns the operator-decomposed tensors (VGqqo, Vqo) for a given channel and energy.
        """
        Gq = self.propagator(Elab)
        Vqqo = self.potential_operators_coupled(channel, Elab)
        VGqqo = Vqqo * jnp.tile(Gq, (2,))[None,:,None]
        return (VGqqo, Vqqo[:,[0, self.mesh.n_mesh + 1],:])
        
        
    @partial(jax.jit, static_argnames=("self"))
    def momentum_pole(
        self,
        Elab: jnp.ndarray # one element JAX-array for tracing
    ):
        k = jnp.sqrt( 0.5 * self.potential.mass * Elab / self.potential.hbarc**2 ) # fm^-1
        return jnp.array([k]) # traceable
        
    
        
        
    @partial(jax.jit, static_argnames=("self"))
    def propagator(
        self,
        Elab: jnp.ndarray # one element JAX-array for tracing
    ):
    
        k = self.momentum_pole(Elab)
        
        # correction for finite map
        if self.mesh.inf:
            C = 0.0 - 1j * jnp.pi
            
        else:
            qmax = jnp.max(self.mesh.q)
            C = jnp.log( (qmax + k) / (qmax - k) ) - 1j * jnp.pi
            
        Bq = self.mesh.wq / ( self.mesh.q**2 - k**2 )  # fm
        
        Gq = jnp.zeros((self.mesh.n_mesh + 1,), dtype=jnp.complex128)
        Gq = Gq.at[0].set( jnp.squeeze(self.potential.mass * k * (0.5 * C + k * jnp.sum(Bq))) ) # MeV fm^-1
        Gq = Gq.at[1:].set( - self.potential.mass * Bq * self.mesh.q**2 ) # MeV fm^-1
        Gq *= (self.potential.factor / self.potential.hbarc) # fm^-2, for different normalization conventions
        
        return Gq # fm^-2
        
        
    def load_compute_write_potential(
        self,
        filename,
        validate,
        load,
        compute,
    ):
        if self.load_potential and glob.glob(filename):
            print(f"Loading file: {filename}")
            t = time.perf_counter()
            data = jnp.load(filename)
            valid_cache = validate(data)
            t = time.perf_counter() - t
            print(f"Done in {t:.5f} sec. Valid: {valid_cache}.")
            
        else:
            valid_cache = False
            print(f"File not found: {filename}")
            
        if valid_cache:
            V = load(data)
            
        else:
            print(f"Computing potential matrix elements...")
            t = time.perf_counter()
            kwargs = compute()
            V = next(iter(kwargs.values()))
            t = time.perf_counter() - t
            print(f"Done in {t:.5f} sec.")
            
            if self.write_potential:
                print(f"Writing potential to file: {filename}")
                jnp.savez(filename, **kwargs)
        print("")
        return V
        
    
    def potential_operators_single(
        self,
        channel,
        Elab # one element JAX-array for tracing
    ):
    
        filebase = f"saved_potentials/{self.potential.name}_Nq{self.mesh.n_mesh}"
        qn = self.channels.single[channel]
        rtol = 0.0
        atol = 1e-8
        
        m = self.mesh.n_mesh
        n = self.potential.n_operators
        k = self.momentum_pole(Elab)
        
        # load Vqqo
        filename = filebase + f"_Vqqo_{channel}.npz"
        validate = lambda data: (
                data['Vqqo'].shape == (m, m, n)
            and jnp.allclose(data['q'], self.mesh.q, rtol=rtol, atol=atol)
        )
        load = lambda data: data['Vqqo']
        compute = lambda: {
            'Vqqo': self.potential.single_channel_operators(self.mesh.q, diag=False, **qn),
            'q': self.mesh.q
        }
        Vqqo = self.load_compute_write_potential(
            filename, validate, load, compute
        )

        # load Vkqo
        filename = filebase + f"_Vkqo_{channel}_{Elab:.3f}MeV.npz"
        validate = lambda data: (
            data['Vkqo'].shape == (1, m, n)
            and jnp.allclose(data['k'], k, rtol=rtol, atol=atol)
            and jnp.allclose(data['q'], self.mesh.q, rtol=rtol, atol=atol)
        )
        load = lambda data: data['Vkqo']
        compute = lambda: {
            'Vkqo': self.potential.single_channel_operators(k, self.mesh.q, **qn),
            'k': k,
            'q': self.mesh.q,
        }
        Vkqo = self.load_compute_write_potential(
            filename, validate, load, compute
        )
        
        # load Vkko
        filename = filebase + f"_Vkko_{channel}_{Elab:.3f}MeV.npz"
        validate = lambda data: (
            data['Vkko'].shape == (1, n)
            and jnp.allclose(data['k'], k, rtol=rtol, atol=atol)
        )
        load = lambda data: data['Vkko']
        compute = lambda: {
            'Vkko': self.potential.single_channel_operators(k, diag=True, **qn),
            'k': k,
        }
        Vkko = self.load_compute_write_potential(
            filename, validate, load, compute
        )
        
        Vkko = jnp.squeeze(Vkko)
        Vkqo = jnp.squeeze(Vkqo)
        
        operators = jnp.zeros((m+1, m+1, n), dtype=jnp.complex128)
        operators = operators.at[0,0].set( Vkko )
        operators = operators.at[0,1:].set( Vkqo )
        operators = operators.at[1:,0].set( Vkqo )
        operators = operators.at[1:,1:].set( Vqqo )
        
        return operators
        
        
    def potential_operators_coupled(
        self,
        channel,
        Elab # one element JAX-array for tracing
    ):
    
        filebase = f"saved_potentials/{self.potential.name}_Nq{self.mesh.n_mesh}"
        qn = self.channels.coupled[channel]
        rtol = 0.0
        atol = 1e-8
        
        m = self.mesh.n_mesh
        n = self.potential.n_operators
        k = self.momentum_pole(Elab)
        
        # load Vqqo
        filename = filebase + f"_Vqqo_{channel}.npz"
        validate = lambda data: (
                data['Vqqo'].shape == (2, 2, m, m, n)
            and jnp.allclose(data['q'], self.mesh.q, rtol=rtol, atol=atol)
        )
        load = lambda data: data['Vqqo']
        compute = lambda: {
            'Vqqo': self.potential.coupled_channel_operators(self.mesh.q, diag=False, **qn),
            'q': self.mesh.q
        }
        Vqqo = self.load_compute_write_potential(
            filename, validate, load, compute
        )

        # load Vkqo
        filename = filebase + f"_Vkqo_{channel}_{Elab:.3f}MeV.npz"
        validate = lambda data: (
            data['Vkqo'].shape == (2, 2, 1, m, n)
            and jnp.allclose(data['k'], k, rtol=rtol, atol=atol)
            and jnp.allclose(data['q'], self.mesh.q, rtol=rtol, atol=atol)
        )
        load = lambda data: data['Vkqo']
        compute = lambda: {
            'Vkqo': self.potential.coupled_channel_operators(k, self.mesh.q, **qn),
            'k': k,
            'q': self.mesh.q,
        }
        Vkqo = self.load_compute_write_potential(
            filename, validate, load, compute
        )
        
        # load Vkko
        filename = filebase + f"_Vkko_{channel}_{Elab:.3f}MeV.npz"
        validate = lambda data: (
            data['Vkko'].shape == (2, 2, 1, n)
            and jnp.allclose(data['k'], k, rtol=rtol, atol=atol)
        )
        load = lambda data: data['Vkko']
        compute = lambda: {
            'Vkko': self.potential.coupled_channel_operators(k, diag=True, **qn),
            'k': k,
        }
        Vkko = self.load_compute_write_potential(
            filename, validate, load, compute
        )
        
        Vkko = jnp.squeeze(Vkko)
        Vkqo = jnp.squeeze(Vkqo)
    
        
        operators = jnp.zeros((2, 2, m+1, m+1, n), dtype=jnp.complex128)
        
        operators = operators.at[0,0,0,0].set( Vkko[0,0] )
        operators = operators.at[1,1,0,0].set( Vkko[1,1] )
        operators = operators.at[0,1,0,0].set( Vkko[0,1] )
        operators = operators.at[1,0,0,0].set( Vkko[1,0] )
        
        operators = operators.at[0,0,1:,0].set( Vkqo[0,0] )
        operators = operators.at[1,1,1:,0].set( Vkqo[1,1] )
        operators = operators.at[0,1,1:,0].set( Vkqo[1,0] ) # mp
        operators = operators.at[1,0,0,1:].set( Vkqo[1,0] ) # pm

        operators = operators.at[0,0,0,1:].set( Vkqo[0,0] )
        operators = operators.at[1,1,0,1:].set( Vkqo[1,1] )
        operators = operators.at[0,1,0,1:].set( Vkqo[0,1] ) # mp
        operators = operators.at[1,0,1:,0].set( Vkqo[0,1] ) # pm
        
        operators = operators.at[0,0,1:,1:].set( Vqqo[0,0] )
        operators = operators.at[1,1,1:,1:].set( Vqqo[1,1] )
        operators = operators.at[0,1,1:,1:].set( Vqqo[0,1] )
        operators = operators.at[1,0,1:,1:].set( Vqqo[1,0] )
        
        operators = jnp.transpose(operators, (0,2,1,3,4))
        operators = jnp.reshape(operators, (2*m+2, 2*m+2, n))

        return operators



    @partial(jax.jit, static_argnames=('self'))
    def solve_single_channel(
        self,
        LECs: jnp.ndarray,
        VGqqo: jnp.ndarray,
        Vqo: jnp.ndarray,
    ) -> jnp.ndarray:
    
        Tq = jnp.linalg.solve(
            jnp.identity(VGqqo.shape[0]) - jnp.tensordot(VGqqo, LECs, axes=([2], [0])),
            jnp.tensordot(Vqo, LECs, axes=([1], [0]))
        )
        return Tq # fm^2
    

    
    @partial(jax.jit, static_argnames=('self'))
    def solve_coupled_channel(
        self,
        LECs: jnp.ndarray,
        VGqqo: jnp.ndarray,
        Vqo: jnp.ndarray,
    ) -> jnp.ndarray:
    
        """
        LECs = (No,)
        VGqqo = (2Nq+2, 2Nq+2, No)
        Vqo = (2Nq+2, 2, No)
        Tq = (2, 2, Nq+1)
            
        """

        Tq = jnp.linalg.solve(
            jnp.identity(VGqqo.shape[0]) - jnp.tensordot(VGqqo, LECs, axes=([2], [0])),
            jnp.tensordot(Vqo, LECs, axes=([2], [0]))
        )
        Tq = jnp.reshape(Tq, (2, self.mesh.n_mesh+1, 2))
        Tq = jnp.transpose(Tq, (0,2,1))

        return Tq # fm^2
        


    

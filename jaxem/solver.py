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
        
        # precompute all potential operators for all
        # channels and store in a pytree
        
        filebase = "saved_potentials/" + self.potential.name + f"_Nq{self.mesh.n_mesh}"
        
        print("filebase = ", filebase)
        
        self.Vqqo = {}
        
        # TODO: put this in a separate function
        
        filename = filebase + "_q.npy"
        if glob.glob(filename):
            loaded_q = jnp.load(filename)
            mesh_matches = jnp.allclose(self.mesh.q, loaded_q, rtol=0.0, atol=1e-15)
        else:
            mesh_matches = False
        
        for channel, quantum_nums in self.channels.single.items():
            filename = filebase + f"_Vqqo_{channel}.npy"
            if load_potential and glob.glob(filename) and mesh_matches:
                print(f"Loading potential matrix elements for {channel} channel...", end="")
                start_time = time.perf_counter()
                self.Vqqo[channel] = jnp.load(filename)
                print(f"Done in {time.perf_counter()-start_time} sec.")
            else:
                print(f"Precomputing potential matrix elements for {channel} channel...", end="")
                start_time = time.perf_counter()
                self.Vqqo[channel] = potential.single_channel_operators(
                    mesh.q, diag=False, **quantum_nums
                )
                print(f"Done in {time.perf_counter()-start_time} sec.")
            
        for channel, quantum_nums in self.channels.coupled.items():
            filename = filebase + f"_Vqqo_{channel}.npy"
            if load_potential and glob.glob(filename) and mesh_matches:
                print(f"Loading potential matrix elements for {channel} channel...", end="")
                start_time = time.perf_counter()
                self.Vqqo[channel] = jnp.load(filename)
                print(f"Done in {time.perf_counter()-start_time} sec.")
            else:
                print(f"Precomputing potential matrix elements for {channel} channel...", end="")
                start_time = time.perf_counter()
                self.Vqqo[channel] = potential.coupled_channel_operators(
                    mesh.q, diag=False, **quantum_nums
                )
                print(f"Done in {time.perf_counter()-start_time} sec.")
        
        if write_potential:
        
            jnp.save(filebase + "_q.npy", mesh.q, allow_pickle=False)
        
            for channel in self.Vqqo.keys():
                jnp.save(
                    filebase + f"_Vqqo_{channel}.npy",
                    self.Vqqo[channel], allow_pickle=False
                )
                
                

        
        
    def t(
        self,
        channel: str = '1S0',
        Elab: float = 10.0,
        LECs: jnp.ndarray = None,
        linear_system: Tuple[jnp.ndarray, jnp.ndarray] = None
    ) -> jnp.ndarray:
    
        if LECs is None:
            LECs = self.potential.LECs
            
            
        if channel in self.channels.single.keys():
            
            if linear_system is None:
                Aqqo, Vqo = self.setup_single_channel(channel, Elab)
                
            else:
                Aqqo, Vqo = linear_system
                
            return self.solve_single_channel(LECs, Aqqo, Vqo)
            
            
        elif channel in self.channels.coupled.keys():
            
            if linear_system is None:
                Aqqo, Vqo, k = self.setup_coupled_channel(channel, Elab)
                
            else:
                Aqqo, Vqo = linear_system
                
            return self.solve_coupled_channel(LECs, Aqqo, Vqo)
            
        else:
            raise ValueError(f"Channel {channel} not found.")
        
        
    def setup_single_channel(
        self,
        channel: str,
        Elab: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns the decomposed linear system (Aqqo, Vqo) for a given channel and energy.
        """

        k = self.calc_momentum_pole(Elab) # jit
        Gq = self.calc_propagator(k) # jit
        
        qn = self.channels.single[channel]
        Vkko = self.potential.single_channel_operators(k, diag=True, **qn)   # no jit
        Vkqo = self.potential.single_channel_operators(k, self.mesh.q, **qn) # no jit
        Vqqo = self.extend_onshell_single(Vkko, Vkqo, self.Vqqo[channel])    # jit
        
        Aqqo, Vqo = self.decomposed_linear_system_single(Vqqo, Gq) # jit
        return Aqqo, Vqo
        

    def setup_coupled_channel(
        self,
        channel: str,
        Elab: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns the decomposed linear system (Aqqo, Vqo) for a given channel and energy.
        """
        k = jnp.sqrt(0.5 * self.potential.mass * Elab / self.potential.hbarc**2)
        k = jnp.array([k])  # traceable
        
        Gq = self.calc_propagator(k) # jit
        
        qn = self.channels.single[channel]
        Vkko = self.potential.single_channel_operators(k, diag=True, **qn)   # no jit
        Vkqo = self.potential.single_channel_operators(k, self.mesh.q, **qn) # no jit
        Vqqo = self.extend_onshell_coupled(Vkko, Vkqo, self.Vqqo[channel])    # jit
        
        Aqqo, Vqo = self.decomposed_linear_system_single(Vqqo, Gq) # jit
        return Aqqo, Vqo
        
        
    @partial(jax.jit, static_argnames=("self"))
    def calc_momentum_pole(
        self,
        Elab: float
    ):
        k = jnp.sqrt(0.5 * self.potential.mass * Elab / self.potential.hbarc**2)
        return jnp.array([k])  # traceable
        
        
    @partial(jax.jit, static_argnames=("self"))
    def calc_propagator(
        self,
        k: jnp.ndarray
    ):
        # correction for finite map
        if self.mesh.inf:
            C = 0.0 - 1j * jnp.pi
            
        else:
            qmax = jnp.max(self.mesh.q)
            C = jnp.log( (qmax + k) / (qmax - k) ) - 1j * jnp.pi
            
        Bq = self.mesh.wq / ( self.mesh.q**2 - k**2 )  # fm
        Gq = jnp.zeros((self.mesh.n_mesh + 1,), dtype=jnp.complex128)
        Gq = Gq.at[0].set( jnp.squeeze(self.potential.mass * k * (0.5 * C + k * jnp.sum(Bq))) ) # fm^-2
        Gq = Gq.at[1:].set( - self.potential.mass * Bq * self.mesh.q**2 ) # fm^-2
        Gq *= (self.potential.factor / self.potential.hbarc)
        
        return Gq
        
        
        
    @partial(jax.jit, static_argnames=("self"))
    def extend_onshell_single(
        self,
        Vkko: jnp.ndarray,
        Vkqo: jnp.ndarray,
        Vqqo: jnp.ndarray
    ):
        # combine with precomputed potential elements
        m = self.mesh.n_mesh
        n = self.potential.n_operators
        
        Vkko = jnp.squeeze(Vkko)
        Vkqo = jnp.squeeze(Vkqo)
        
        operators = jnp.zeros((m+1, m+1, n), dtype=jnp.complex128)
        operators = operators.at[0,0].set( Vkko )
        operators = operators.at[0,1:].set( Vkqo )
        operators = operators.at[1:,0].set( Vkqo )
        operators = operators.at[1:,1:].set( Vqqo ) # precomputed, independent of k
        
        return operators
        

    @partial(jax.jit, static_argnames=('self'))
    def extend_onshell_coupled(
        self,
        Vkko: jnp.ndarray,
        Vkqo: jnp.ndarray,
        Vqqo: jnp.ndarray
    ) -> jnp.ndarray:

        # combine with precomputed potential elements
        m = self.mesh.n_mesh
        n = self.potential.n_operators
        
        Vkko = jnp.squeeze(Vkko)
        Vkqo = jnp.squeeze(Vkqo)
        
        operators = jnp.zeros((2, 2, m+1, m+1, n), dtype=jnp.complex128)
        
        operators = operators.at[0,0,0,0].set( Vkko[0,0] )
        operators = operators.at[1,1,0,0].set( Vkko[1,1] )
        operators = operators.at[0,1,0,0].set( Vkko[0,1] )
        operators = operators.at[1,0,0,0].set( Vkko[1,0] )
        
        operators = operators.at[0,0,1:,0].set( Vkqo[0,0] )
        operators = operators.at[1,1,1:,0].set( Vkqo[1,1] )
        operators = operators.at[0,1,1:,0].set( Vkqo[0,1] )
        operators = operators.at[1,0,0,1:].set( Vkqo[1,0] ) # not a bug

        operators = operators.at[0,0,0,1:].set( Vkqo[0,0] )
        operators = operators.at[1,1,0,1:].set( Vkqo[1,1] )
        operators = operators.at[0,1,0,1:].set( Vkqo[0,1] )
        operators = operators.at[1,0,1:,0].set( Vkqo[1,0] ) # not a bug
        
        operators = operators.at[0,0,1:,1:].set( Vqqo[0,0] )
        operators = operators.at[1,1,1:,1:].set( Vqqo[1,1] )
        operators = operators.at[0,1,1:,1:].set( Vqqo[0,1] )
        operators = operators.at[1,0,1:,1:].set( Vqqo[1,0] )
        
        operators = jnp.transpose(operators, (0,2,1,3,4))
        operators = jnp.reshape(operators, (2*m+2, 2*m+2, n))
        
        return operators


    @partial(jax.jit, static_argnames=('self'))
    def decomposed_linear_system_single(
        self,
        Vqqo: jnp.ndarray,
        Gq: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        Aqqo = -Vqqo * Gq[None,:,None]
        iq = jnp.arange(Aqqo.shape[0])
        Aqqo = Aqqo.at[iq,iq].add(1.0)
        return (Aqqo, Vqqo[:,0,:])
        
        
    @partial(jax.jit, static_argnames=('self'))
    def decomposed_linear_system_coupled(
        self,
        LECs: jnp.ndarray,
        Vqqo: jnp.ndarray,
        Gq: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    
        Aqqo = -Vqqo * jnp.tile(propagator, (2,))[None,:,None]
        iq = jnp.arange(Aqqo.shape[0])
        Aqqo = Aqqo.at[iq,iq].add(1.0)
        return (Aqqo, Vqqo[:,[0, n + 1],:])


    @partial(jax.jit, static_argnames=('self'))
    def solve_single_channel(
        self,
        LECs: jnp.ndarray,
        Aqqo: jnp.ndarray,
        Vqo: jnp.ndarray,
    ) -> jnp.ndarray:
    
        Tq = jnp.linalg.solve(
            jnp.tensordot(Aqqo, LECs, axes=([2], [0])),
            jnp.tensordot(Vqo, LECs, axes=([1], [0]))
        )
        return Tq
    
    @partial(jax.jit, static_argnames=('self'))
    def batch_solve_single_channel(
        self,
        LECs_batch: jnp.ndarray,
        Aqqo: jnp.ndarray,
        Vqo: jnp.ndarray,
    ) -> jnp.ndarray:

        return jax.vmap(self.solve_single_channel, in_axes=(0,None,None))(LECs_batch, Aqqo, Vqo)
    
    @partial(jax.jit, static_argnames=('self'))
    def solve_coupled_channel(
        self,
        LECs: jnp.ndarray,
        Aqqo: jnp.ndarray,
        Vqo: jnp.ndarray,
    ) -> jnp.ndarray:
    
        Tq = jnp.linalg.solve(
            jnp.tensordot(Aqqo, LECs, axes=([2], [0])),
            jnp.tensordot(Vqo, LECs, axes=([1], [0]))
        )
        Tq = jnp.reshape(Tq, (2, self.mesh.n_mesh+1, 2))
        Tq = jnp.transpose(Tq, (0,2,1))
        return Tq
        
        
    

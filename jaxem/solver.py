import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
import time
from .channels import Channels
from .mesh import Mesh
from .potential import Potential
from typing import Optional, Dict, Tuple, List
import matplotlib.pyplot as plt
import glob

class Solver:

    def __init__(
        self,
        mesh: Mesh,
        channels: Channels,
        potential: Potential,
        Elabs: List,
        write_potential: bool = True,
        load_potential: bool = True
    ):
    
        self.mesh = mesh
        self.channels = channels
        self.potential = potential
        self.Elabs = jnp.array(Elabs)
        self.write_potential = write_potential
        self.load_potential = load_potential
        
        self.setup()
        
    
    def setup(self):
        
        self.setup_propagator()
        
        nc = self.channels.n_single
        ncc = self.channels.n_coupled
        nk = self.n_poles
        nq = self.mesh.n_mesh
        no = self.potential.n_operators
        
        self.Vcqqo = jnp.zeros((nc,nq,nq,no), dtype=jnp.complex128)
        self.Vckqo = jnp.zeros((nc,nk,nq,no), dtype=jnp.complex128)
        self.Vcko = jnp.zeros((nc,nk,no), dtype=jnp.complex128)

        self.Vccqqo = jnp.zeros((ncc,2,2,nq,nq,no), dtype=jnp.complex128)
        self.Vcckqo = jnp.zeros((ncc,2,2,nk,nq,no), dtype=jnp.complex128)
        self.Vccko = jnp.zeros((ncc,2,2,nk,no), dtype=jnp.complex128)
        
        for c in range(nc):
            self.setup_single_channel(c)
            
        for cc in range(ncc):
            self.setup_coupled_channel(cc)


    def setup_propagator(self):

        # momentum poles
        self.k = jnp.sqrt( 0.5 * self.potential.mass * self.Elabs / self.potential.hbarc**2 ) # fm^-1
        self.n_poles = self.k.shape[0]
        
        k = self.k[:,None]
        q = self.mesh.q[None,:]
        Bkq = self.mesh.wq[None,:] / ( q**2 - k**2 )  # fm
        
        # correction for finite map
        if self.mesh.inf:
            Ck = 0.0 - 1j * jnp.pi
            
        else:
            qmax = jnp.max(self.mesh.q)
            Ck = jnp.log( (qmax + self.k) / (qmax - self.k) ) - 1j * jnp.pi

        # fill in propagator   fm^-2
        self.Gkq = jnp.zeros((self.n_poles, self.mesh.n_mesh + 1), dtype=jnp.complex128)
        
        self.Gkq = self.Gkq.at[:,0].set( self.potential.mass * self.k * (0.5 * Ck + self.k * jnp.sum(Bkq, axis=1) ) ) # MeV fm^-1
        self.Gkq = self.Gkq.at[:,1:].set( - self.potential.mass * Bkq * q**2 ) # MeV fm^-1
        self.Gkq *= (self.potential.factor / self.potential.hbarc) # fm^-2, for different normalization conventions
    
        
        
    @partial(jax.jit, static_argnames=("self"))
    def t(
        self,
        LECs: jnp.ndarray,
    ):
    
        ic = jnp.arange(self.channels.n_single)
        icc = jnp.arange(self.channels.n_coupled)
        ik = jnp.arange(self.n_poles)
        
        Tckq = jax.vmap(
            jax.vmap(self.single_channel_t, in_axes=(None,None,0)),
            in_axes=(None,0,None)
        )(LECs, ic, ik)
    
        Tcckq = jax.vmap(
            jax.vmap(self.coupled_channel_t, in_axes=(None,None,0)),
            in_axes=(None,0,None)
        )(LECs, icc, ik)
        
        return Tckq, Tcckq # fm^2


    @partial(jax.jit, static_argnames=("self"))
    def onshell_t(
        self,
        LECs: jnp.ndarray,
    ):
    
        Tckq, Tcckq = self.t(LECs)
        
        return Tckq[:,:,0], Tcckq[:,:,:,:,0] # fm^2


    @partial(jax.jit, static_argnames=("self"))
    def scattering_params(
        self,
        t_onshell: Tuple[jnp.ndarray]
    ):
    
        Tck, Tcck = t_onshell # fm^2
        
        hc = self.potential.hbarc # MeV fm
        k = self.k # fm^-1
        m = self.potential.mass # MeV
        f = self.potential.factor # 1
        Jc = self.channels.single_quantum_nums[:,0]
        Jcc = self.channels.coupled_quantum_nums[:,0]
        
        # single channels
        Sck = 1 - 1j * jnp.pi * f * m * k * Tck / hc
        delta_ck = 0.5 * jnp.arctan2(Sck.imag, Sck.real)
        eta_ck = jnp.abs(Sck)
        sigma_ck = 0.5 * jnp.pi * (2*Jc[:,None]+1) * (1-Sck).real / k[None,:]**2 # fm^2
        sigma_ck *= 10. # convert from fm^2 to mb
        single_output = jnp.stack((delta_ck, eta_ck, sigma_ck))

        # coupled channels
        Scck = - 1j * jnp.pi * f * m * k[None,:,None,None] * Tcck / hc
        Scck = Scck.at[:,:,0,0].add(1.0)
        Scck = Scck.at[:,:,1,1].add(1.0)
        

        Z = 0.5 * ( Scck[:,:,0,1] + Scck[:,:,1,0] ) / jnp.sqrt(Scck[:,:,0,0] * Scck[:,:,1,1])
        epsilon = -0.25 * 1j * jnp.log( (1 + Z) / (1 - Z) )
        epsilon = epsilon.real
        cond = (epsilon < -1e-8) & (Jcc == 1)[..., None]
        epsilon = jnp.where(cond, jnp.abs(epsilon), epsilon)
        
        S_minus = Scck[:,:,0,0] / jnp.cos(2 * epsilon)
        S_plus = Scck[:,:,1,1] / jnp.cos(2 * epsilon)
        
        delta_minus = 0.5 * jnp.arctan2(S_minus.imag, S_minus.real)
        delta_plus = 0.5 * jnp.arctan2(S_plus.imag, S_plus.real)
        delta_minus = jnp.where(delta_minus < -1e-8, jnp.pi + delta_minus, delta_minus)
        
        eta_minus = jnp.abs(S_minus)
        eta_plus = jnp.abs(S_plus)
        
        sigma_cck = 0.5 * jnp.pi * (2*Jcc[:,None]+1) * (2 - S_minus - S_plus).real / k[None,:]**2
        sigma_cck *= 10. # convert from fm^2 to mb        
        coupled_output = jnp.stack((delta_minus, delta_plus, epsilon, eta_minus, eta_plus, sigma_cck))

        
        # total cross section
        sigma_tot = jnp.sum(sigma_ck, axis=0) + jnp.sum(sigma_cck, axis=0)
        
        return sigma_tot, single_output, coupled_output
    

    
        
    @partial(jax.jit, static_argnames=("self"))
    def single_channel_t(
        self,
        LECs: jnp.ndarray,
        c: int, # channel index,
        k: int # pole index
    ):
    
        nq = self.mesh.n_mesh
        
        # dot LECs with precomputed operators
        V_onshell   =  self.Vcko[c,k] @ LECs
        V_halfshell = self.Vckqo[c,k] @ LECs
        V_offshell  = self.Vcqqo[c] @ LECs
        
        Vqq = jnp.zeros((nq+1, nq+1), dtype=jnp.complex128)
        Vqq = Vqq.at[0,0].set( V_onshell )
        Vqq = Vqq.at[0,1:].set( V_halfshell )
        Vqq = Vqq.at[1:,0].set( V_halfshell )
        Vqq = Vqq.at[1:,1:].set( V_offshell )
        
        Gq = self.Gkq[k,:]
        
        iq = jnp.arange(nq+1)
        Aqq = -Vqq * Gq[None,:]
        Aqq = Aqq.at[iq,iq].add(1.0)
        Tq = jnp.linalg.solve(Aqq, Vqq[:,0])

        return Tq # fm^2
        

    @partial(jax.jit, static_argnames=("self"))
    def single_channel_operators(
        self,
        c: int, # channel index,
        k: int # pole index
    ):
    
        nq = self.mesh.n_mesh
        no = self.potential.n_operators
        
        V_onshell   =  self.Vcko[c,k]
        V_halfshell = self.Vckqo[c,k]
        V_offshell  = self.Vcqqo[c]
        
        Vqqo = jnp.zeros((nq+1, nq+1, no), dtype=jnp.complex128)
        Vqqo = Vqqo.at[0,0].set( V_onshell )
        Vqqo = Vqqo.at[0,1:].set( V_halfshell )
        Vqqo = Vqqo.at[1:,0].set( V_halfshell )
        Vqqo = Vqqo.at[1:,1:].set( V_offshell )
        
        Gq = self.Gkq[k,:]
        VGqqo = Vqqo * Gq[None,:,None]
        Vqo = Vqqo[:,0,:]
        
        return VGqqo, Vqo


    @partial(jax.jit, static_argnames=("self"))
    def coupled_channel_t(
        self,
        LECs: jnp.ndarray,
        cc: int, # channel index
        k: int
    ):
    
        nq = self.mesh.n_mesh
        
        # dot LECs with precomputed operators
        V_onshell = self.Vccko[cc,:,:,k] @ LECs
        V_halfshell = self.Vcckqo[cc,:,:,k] @ LECs
        V_offshell  = self.Vccqqo[cc] @ LECs
        
        Vqq = jnp.zeros((2, 2, nq+1, nq+1), dtype=jnp.complex128)
        
        Vqq = Vqq.at[:,:,0,0].set( V_onshell )
        
        Vqq = Vqq.at[0,0,1:,0].set( V_halfshell[0,0] )
        Vqq = Vqq.at[1,1,1:,0].set( V_halfshell[1,1] )
        Vqq = Vqq.at[0,1,1:,0].set( V_halfshell[1,0] )
        Vqq = Vqq.at[1,0,0,1:].set( V_halfshell[1,0] )
        
        Vqq = Vqq.at[0,0,0,1:].set( V_halfshell[0,0] )
        Vqq = Vqq.at[1,1,0,1:].set( V_halfshell[1,1] )
        Vqq = Vqq.at[0,1,0,1:].set( V_halfshell[0,1] )
        Vqq = Vqq.at[1,0,1:,0].set( V_halfshell[0,1] )
        
        Vqq = Vqq.at[:,:,1:,1:].set( V_offshell )
        
        Vqq = jnp.transpose(Vqq, (0,2,1,3))
        Vqq = jnp.reshape(Vqq, (2*nq+2, 2*nq+2))
        
        Gq = jnp.tile(self.Gkq[k,:], (2,))
        
        iq = jnp.arange(2*nq+2)
        Aqq = -Vqq * Gq[None,:]
        Aqq = Aqq.at[iq,iq].add(1.0)
        Tq = jnp.linalg.solve(Aqq, Vqq[:,[0,nq+1]])

        Tq = jnp.reshape(Tq, (2, self.mesh.n_mesh+1, 2))
        Tq = jnp.transpose(Tq, (0,2,1))
                
        return Tq # fm^2
        
    @partial(jax.jit, static_argnames=("self"))
    def coupled_channel_t_flat(
        self,
        LECs: jnp.ndarray,
        cc: int, # channel index
        k: int
    ):
    
        Tq = self.coupled_channel_t(LECs, cc, k)
        
        return jnp.concatenate(
            (Tq[0,0], Tq[1,0], Tq[0,1], Tq[1,1]),
            axis=0
        )
        

    @partial(jax.jit, static_argnames=("self"))
    def coupled_channel_operators(
        self,
        cc: int, # channel index
        k: int
    ):
        nq = self.mesh.n_mesh
        no = self.potential.n_operators
        
        V_onshell = self.Vccko[cc,:,:,k]
        V_halfshell = self.Vcckqo[cc,:,:,k]
        V_offshell  = self.Vccqqo[cc]

        Vqqo = jnp.zeros((2, 2, nq+1, nq+1, no), dtype=jnp.complex128)
        
        Vqqo = Vqqo.at[:,:,0,0].set( V_onshell )
        
        Vqqo = Vqqo.at[0,0,1:,0].set( V_halfshell[0,0] )
        Vqqo = Vqqo.at[1,1,1:,0].set( V_halfshell[1,1] )
        Vqqo = Vqqo.at[0,1,1:,0].set( V_halfshell[1,0] )
        Vqqo = Vqqo.at[1,0,0,1:].set( V_halfshell[1,0] )
        
        Vqqo = Vqqo.at[0,0,0,1:].set( V_halfshell[0,0] )
        Vqqo = Vqqo.at[1,1,0,1:].set( V_halfshell[1,1] )
        Vqqo = Vqqo.at[0,1,0,1:].set( V_halfshell[0,1] )
        Vqqo = Vqqo.at[1,0,1:,0].set( V_halfshell[0,1] )
        
        Vqqo = Vqqo.at[:,:,1:,1:].set( V_offshell )
        
        Vqqo = jnp.transpose(Vqqo, (0,2,1,3,4))
        Vqqo = jnp.reshape(Vqqo, (2*nq+2, 2*nq+2, no))
        
        Gq = jnp.tile(self.Gkq[k,:], (2,))

        VGqqo = Vqqo * Gq[None,:,None]
        Vqo = Vqqo[:,[0,nq+1],:]
        
        return VGqqo, Vqo
        
        
    @partial(jax.jit, static_argnames=("self"))
    def coupled_channel_operators_flat(
        self,
        cc: int, # channel index
        k: int
    ):
        VGqqo, Vqo = self.coupled_channel_operators(cc, k)
        VGqqo = jax.vmap(
            jax.scipy.linalg.block_diag, in_axes=(2,2),
            out_axes=2
        )(VGqqo, VGqqo)
        Vqo = jnp.concatenate([Vqo[:,0,:], Vqo[:,1,:]], axis=0)
        return VGqqo, Vqo
        
        
    def setup_single_channel(
        self,
        channel_index: int
    ):
    
        label = self.channels.single_labels[channel_index]
        J, S, T, Tz, L = self.channels.single_quantum_nums[channel_index]
        qn = {"J": J, "S": S, "T": T, "Tz": Tz, "L": L}
        filebase = f"saved_potentials/{self.potential.name}_Nq{self.mesh.n_mesh}_{label}"
        rtol = 0.0
        atol = 1e-8
        
        nk = self.k.shape[0]
        nq = self.mesh.n_mesh
        no = self.potential.n_operators
        
        
        # load Vqqo
        filename = filebase + f"_Vqqo.npz"
        validate = lambda data: (
                data['Vqqo'].shape == (nq, nq, no)
            and jnp.allclose(data['q'], self.mesh.q, rtol=rtol, atol=atol)
        )
        load = lambda data: data['Vqqo']
        compute = lambda: {
            'Vqqo': self.potential.single_channel_operators(self.mesh.q, diag=False, **qn),
            'q': self.mesh.q
        }
        self.Vcqqo = self.Vcqqo.at[channel_index].set(
            self.load_compute_write_potential(
                filename, validate, load, compute
            )
        )

        # load Vkqo
        filename = filebase + f"_Vkqo.npz"
        validate = lambda data: (
            data['Vkqo'].shape == (nk, nq, no)
            and jnp.allclose(data['k'], self.k, rtol=rtol, atol=atol)
            and jnp.allclose(data['q'], self.mesh.q, rtol=rtol, atol=atol)
        )
        load = lambda data: data['Vkqo']
        compute = lambda: {
            'Vkqo': self.potential.single_channel_operators(self.k, self.mesh.q, **qn),
            'k': self.k,
            'q': self.mesh.q,
        }
        self.Vckqo = self.Vckqo.at[channel_index].set(
            self.load_compute_write_potential(
                filename, validate, load, compute
            )
        )
        
        # load Vkko
        filename = filebase + f"_Vko.npz"
        validate = lambda data: (
            data['Vko'].shape == (nk, no)
            and jnp.allclose(data['k'], self.k, rtol=rtol, atol=atol)
        )
        load = lambda data: data['Vko']
        compute = lambda: {
            'Vko': self.potential.single_channel_operators(self.k, diag=True, **qn),
            'k': self.k,
        }
        self.Vcko = self.Vcko.at[channel_index].set(
            self.load_compute_write_potential(
                filename, validate, load, compute
            )
        )

        
    def setup_coupled_channel(
        self,
        channel_index: int
    ):
    
        label = self.channels.coupled_labels[channel_index]
        J, S, T, Tz, L1, L2 = self.channels.coupled_quantum_nums[channel_index]
        qn = {"J": J, "S": S, "T": T, "Tz": Tz, "L1": L1, "L2": L2}
        filebase = f"saved_potentials/{self.potential.name}_Nq{self.mesh.n_mesh}_{label}"
        rtol = 0.0
        atol = 1e-8
        
        nk = self.k.shape[0]
        nq = self.mesh.n_mesh
        no = self.potential.n_operators
        
        # load Vqqo
        filename = filebase + f"_Vqqo.npz"
        validate = lambda data: (
                data['Vqqo'].shape == (2, 2, nq, nq, no)
            and jnp.allclose(data['q'], self.mesh.q, rtol=rtol, atol=atol)
        )
        load = lambda data: data['Vqqo']
        compute = lambda: {
            'Vqqo': self.potential.coupled_channel_operators(self.mesh.q, diag=False, **qn),
            'q': self.mesh.q
        }
        self.Vccqqo = self.Vccqqo.at[channel_index].set(
            self.load_compute_write_potential(
                filename, validate, load, compute
            )
        )

        # load Vkqo
        filename = filebase + f"_Vkqo.npz"
        validate = lambda data: (
            data['Vkqo'].shape == (2, 2, nk, nq, no)
            and jnp.allclose(data['k'], self.k, rtol=rtol, atol=atol)
            and jnp.allclose(data['q'], self.mesh.q, rtol=rtol, atol=atol)
        )
        load = lambda data: data['Vkqo']
        compute = lambda: {
            'Vkqo': self.potential.coupled_channel_operators(self.k, self.mesh.q, **qn),
            'k': self.k,
            'q': self.mesh.q,
        }
        self.Vcckqo = self.Vcckqo.at[channel_index].set(
            self.load_compute_write_potential(
                filename, validate, load, compute
            )
        )
        
        # load Vkko
        filename = filebase + f"_Vko.npz"
        validate = lambda data: (
            data['Vko'].shape == (2, 2, nk, no)
            and jnp.allclose(data['k'], self.k, rtol=rtol, atol=atol)
        )
        load = lambda data: data['Vko']
        compute = lambda: {
            'Vko': self.potential.coupled_channel_operators(self.k, diag=True, **qn),
            'k': self.k,
        }
        self.Vccko = self.Vccko.at[channel_index].set(
            self.load_compute_write_potential(
                filename, validate, load, compute
            )
        )
        

    

        
    def load_compute_write_potential(
        self,
        filename,
        validate,
        load,
        compute,
    ):
        print("")
        
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
        
        return V
        

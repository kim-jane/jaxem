import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial
import time
from contextlib import contextmanager
from .solver import Solver
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt

class Sampler:

    def __init__(
        self,
        emulator,
        models,
        metas,
        Elabs,
        sigmas,
        prior_scale : float = 0.1,
        likelihood_scale : float = 0.1,
        static_indices = None
    ):
        self.emulator = emulator
        self.potential = emulator.potential
        self.models = models
        self.metas = metas
        self.Elab = Elabs
        self.sigmas = sigmas / 10.0 # convert mb to fm^2
        self.n_energies = Elabs.shape[0]
        self.prior_variance = (prior_scale * self.potential.LECs)**2
        self.likelihood_scale = likelihood_scale
        self.static_indices = static_indices or []
        

    def sample(
        self,
        seed: int = 17,
        n_chains: int = 20,
        n_equil: int = 100,
        n_skip: int = 10,
        n_samples_per_chain = 500,
        init_noise = 0.002,
        step_scale = 0.005
    ):
    
    
        scaled_step_size = step_scale * self.potential.LECs
        
        for i in self.static_indices:
            scaled_step_size = scaled_step_size.at[i].set(0.0)
        
    
        def propagate_chains(state, _):
            
            key, chains_o, logP_o, sigmas_o, acc_o = state
            
            # propose moves for all chains
            key, subkey = jax.random.split(key)
            
            chains_n = chains_o + scaled_step_size[None,:] * jax.random.normal(
                subkey, shape=(n_chains, self.potential.n_operators)
            )
            
            logP_n, sigmas_n = self.batch_log_posterior(chains_n)
            #logP_n = self.batch_log_prior(chains_n)
            
            # accept or reject
            key, subkey = jax.random.split(key)
            log_unif = jnp.log(jax.random.uniform(subkey, shape=(n_chains,)))
            accept = log_unif < (logP_n - logP_o)
            
            chains_o = jnp.where(accept[:,None], chains_n, chains_o)
            logP_o = jnp.where(accept, logP_n, logP_o)
            sigmas_o = jnp.where(accept[:,None], sigmas_n, sigmas_o)
            acc_o = acc_o + accept.astype(jnp.int32)
            
            state = (key, chains_o, logP_o, sigmas_o, acc_o)
            
            return state, (chains_o, sigmas_o)
        
    
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        chains_o = jax.random.normal(subkey, shape=(n_chains, self.potential.n_operators))
        chains_o = self.potential.LECs[None,:] * (1 + init_noise * chains_o)
        
        logP_o, sigmas_o = self.batch_log_posterior(chains_o)
        acc_o = jnp.zeros((n_chains,), jnp.int32)
        
        state_o = (key, chains_o, logP_o, sigmas_o, acc_o)
        
        '''
                
        # run all chains
        total_steps = n_skip * (n_equil + n_samples_per_chain)
        state_n, (chain_history, sigma_history) = jax.lax.scan(
            propagate_chains,
            state_o,
            xs=None,
            length=total_steps
        )
        
        key, chains_n, logP_n, sigmas_n, acc_n = state_n
        chain_history = chain_history[n_skip*n_equil::n_skip]
        chain_history = jnp.reshape(chain_history, (-1, self.potential.n_operators))
        sigma_history = sigma_history[n_skip*n_equil::n_skip]
        sigma_history = jnp.reshape(sigma_history, (-1, self.n_energies))
                
        return chain_history, sigma_history, jnp.mean(acc_n)/total_steps
        
        '''
        
        # 1) burn-in (and thinning within burn-in): run n_equil * n_skip steps, store nothing
        burn_steps = n_equil * n_skip
        state_burn, _ = jax.lax.scan(propagate_chains, state_o, xs=None, length=burn_steps)

        # helper: advance n_skip steps (no outputs), then collect current state
        def advance_and_collect(state, _):
            state, _ = jax.lax.scan(propagate_chains, state, xs=None, length=n_skip)
            key, chains, logP, sigmas, acc = state
            # collect the thinned draw
            return state, (chains, sigmas)

        # 2) sampling: repeat (advance n_skip, then collect) n_samples_per_chain times
        state_end, (chains_collected, sigmas_collected) = jax.lax.scan(
            advance_and_collect, state_burn, xs=None, length=n_samples_per_chain
        )

        # flatten across chains if you want the same return shape as before
        chain_history = chains_collected.reshape(-1, self.potential.n_operators)
        sigma_history = sigmas_collected.reshape(-1, self.n_energies)

        # acceptance over the whole run
        _, _, _, _, acc_end = state_end
        total_steps = n_skip * (n_equil + n_samples_per_chain)
        accept_rate = jnp.mean(acc_end) / total_steps

        return chain_history, sigma_history, accept_rate
    
    
    def batch_log_prior(
        self,
        LECs_batch
    ):
        return jax.vmap(self.log_prior, in_axes=(0,))(LECs_batch)


    def log_prior(
        self,
        LECs
    ):
        logP = - 0.5 * (
            (LECs - self.potential.LECs) ** 2 / self.prior_variance
            + jnp.log(2 * jnp.pi * self.prior_variance)
        )
        return jnp.sum(logP)

    def batch_log_likelihood(
        self,
        LECs_batch
    ):
        return jax.vmap(self.log_likelihood, in_axes=(0,))(LECs_batch)
        
        
    def log_likelihood(
        self,
        LECs
    ):
    
        # can this be vmapped?
        
        sigmas = jnp.zeros(self.n_energies)
        
        for i in range(self.n_energies):
            sigmas = sigmas.at[i].set( self.emulator.total_cross_section(
                self.models[i], self.metas[i], LECs=LECs
                )
            )
            
        likelihood_variance = (self.likelihood_scale * sigmas)**2
            
        logP = -0.5 * (
            (sigmas - self.sigmas) ** 2 / likelihood_variance
            + jnp.log(2 * jnp.pi * likelihood_variance)
        )
        
        return jnp.sum(logP), sigmas


    def log_posterior(
        self,
        LECs
    ):
        logP, sigmas = self.log_likelihood(LECs)
        return logP + self.log_prior(LECs), sigmas
        
    def batch_log_posterior(
        self,
        LECs_batch
    ):
        return jax.vmap(self.log_posterior, in_axes=(0,))(LECs_batch)
        

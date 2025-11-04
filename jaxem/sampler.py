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
from jax.scipy.optimize import minimize

class Sampler:

    def __init__(
        self,
        emulator,
        sigmas,
        prior_scale : float = 0.5,
        likelihood_scale : float = 0.1,
        static_indices = None
    ):
        self.emulator = emulator
        self.sigmas = sigmas
        self.prior_scale = prior_scale
        self.likelihood_scale = likelihood_scale
        self.static_indices = static_indices or []
        self.static_indices = jnp.array(self.static_indices, dtype=int)
        
        self.solver = emulator.solver
        self.potential = emulator.potential
        self.channels = emulator.channels
        self.Elabs = self.solver.Elabs
        self.n_poles = self.solver.n_poles
        self.prior_variance = (self.prior_scale * self.potential.LECs)**2
        

    def sample(
        self,
        seed: int = 17,
        n_chains: int = 20,
        n_equil: int = 100,
        n_skip: int = 10,
        n_samples_per_chain = 500,
        init_noise = 0.1,
        step_scale = 0.1,
        use_emulator = True
    ):
    
        if use_emulator:
            onshell_t_and_err_func = self.emulator.onshell_t_and_err
            
        else:
            onshell_t_and_err_func = self.solver.onshell_t_and_err
        
        # get MAP LECs
        @jax.jit
        def objective(LECs):
            LECs = LECs.at[self.static_indices].set(self.potential.LECs[self.static_indices])
            onshell_t_and_err = onshell_t_and_err_func(LECs)
            sigma_tot, err_sigma_tot, _, _ = self.solver.scattering_params(onshell_t_and_err)
            return -self.log_posterior(LECs, sigma_tot, err_sigma_tot)


        print("Estimating maximum posterior...")
        self.MAP_LECs = minimize(
            objective,
            x0=self.potential.LECs,
            method='BFGS',
            options={'gtol': 1e-5}
        ).x

        print("Best Fit LECs = ", self.potential.LECs)
        print("MAP LECs = ", self.MAP_LECs)
        
        
        # get best step size in each direction using hessian at MAP
        H = jax.hessian(objective)(self.MAP_LECs)
        H = 0.5 * (H + H.T)
        
        
        jitter = 1e-6
        Sigma = jnp.linalg.inv(H + jitter * jnp.eye(H.shape[0]))
        print("Sigma = ", Sigma)
        L = jnp.linalg.cholesky(Sigma)
        L = L.at[self.static_indices,:].set(jnp.zeros((self.static_indices.shape[0], L.shape[1])))
        print("L = ", L)
        
        print("Step = ", L @ jnp.ones(L.shape[1]))
        
        #step = self.potential.LECs # for debugging
        
        #step = jnp.sqrt(1.0 / jnp.diag(H)) # uncorrelated proposals
        #step = step.at[self.static_indices].set(jnp.zeros_like(self.static_indices))
        
        #print("Step = ", step)
        #print("Scaled step = ", step_scale * step)
        
            
        batch_onshell_t_and_err = jax.jit(jax.vmap(onshell_t_and_err_func, in_axes=(0,)))
        batch_scattering_params = jax.jit(jax.vmap(self.solver.scattering_params, in_axes=(0,)))
    
        @jax.jit
        def propagate_chains(state, _):
            
            key, chains_o, logP_o, sigma_tot_o, err_sigma_tot_o, single_params_o, coupled_params_o, acc_o = state
            
            # propose new chains
            key, subkey = jax.random.split(key)
            chains_n = jax.random.normal(subkey, (n_chains, self.potential.n_operators))
            #chains_n = chains_o + step_scale * chains_n * step[None, :] # for uncorrelated proposals
            chains_n = chains_o + step_scale * jnp.einsum('ij,cj->ci', L, chains_n)

            # forward model
            onshell_t_and_err = batch_onshell_t_and_err(chains_n)
            sigma_tot_n, err_sigma_tot_n, single_params_n, coupled_params_n = batch_scattering_params(onshell_t_and_err)
            logP_n = self.batch_log_posterior(chains_n, sigma_tot_n, err_sigma_tot_n)

            # accept/reject
            key, subkey = jax.random.split(key)
            accept = jnp.log(jax.random.uniform(subkey, (n_chains,))) < (logP_n - logP_o)

            def choose(new, old):
                return jnp.where(accept[(...,) + (None,) * (new.ndim - 1)], new, old)

            # update all state pieces in one place
            chains_o = choose(chains_n, chains_o)
            sigma_tot_o = choose(sigma_tot_n, sigma_tot_o)
            err_sigma_tot_o = choose(err_sigma_tot_n, err_sigma_tot_o)
            logP_o = jnp.where(accept, logP_n, logP_o)
            single_params_o = choose(single_params_n, single_params_o)
            coupled_params_o = choose(coupled_params_n, coupled_params_o)
            acc_o = acc_o + accept.astype(jnp.int32)

            new_state = (key, chains_o, logP_o, sigma_tot_o, err_sigma_tot_o, single_params_o, coupled_params_o, acc_o)
            
            return new_state, (chains_o, sigma_tot_o, err_sigma_tot_o, single_params_o, coupled_params_o)
        
    
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        chains_o = jax.random.normal(subkey, shape=(n_chains, self.potential.n_operators))
        #chains_o = self.MAP_LECs[None,:] + init_noise * chains_o * step[None,:]
        chains_o = self.MAP_LECs[None,:] + init_noise * jnp.einsum('ci,ij->cj', chains_o, L)
        
        onshell_t_and_err = batch_onshell_t_and_err(chains_o)
        sigma_tot_o, err_sigma_tot_o, single_params_o, coupled_params_o = batch_scattering_params(onshell_t_and_err)
        logP_o = self.batch_log_posterior(chains_o, sigma_tot_o, err_sigma_tot_o)
        acc_o = jnp.zeros((n_chains,), jnp.int32)
        
        state_o = (key, chains_o, logP_o, sigma_tot_o, err_sigma_tot_o, single_params_o, coupled_params_o, acc_o)

        burn_steps = n_equil * n_skip
        state_burn, _ = jax.lax.scan(propagate_chains, state_o, xs=None, length=burn_steps)

        @jax.jit
        def advance_and_collect(state, _):
            state, _ = jax.lax.scan(propagate_chains, state, xs=None, length=n_skip)
            (key, chains_o, logP_o, sigma_tot_o, err_sigma_tot_o, single_params_o, coupled_params_o, acc_o) = state
            return state, (chains_o, sigma_tot_o, err_sigma_tot_o, single_params_o, coupled_params_o)

        state_end, (chains, sigma_tot, err_sigma_tot, single_params, coupled_params) = jax.lax.scan(
            advance_and_collect, state_burn, xs=None, length=n_samples_per_chain
        )

        # flatten the chain and sigma_tot arrays
        chains = chains.reshape(-1, self.potential.n_operators)
        sigma_tot = sigma_tot.reshape(-1, self.n_poles)
        err_sigma_tot = err_sigma_tot.reshape(-1, self.n_poles)
        single_params = single_params.reshape(-1, 3, self.channels.n_single, self.n_poles)
        single_params = jnp.transpose(single_params, (1,0,2,3))
        coupled_params = coupled_params.reshape(-1, 6, self.channels.n_coupled, self.n_poles)
        coupled_params = jnp.transpose(coupled_params, (1,0,2,3))

        acc_end = state_end[-1]
        total_steps = n_skip * (n_equil + n_samples_per_chain)
        accept_rate = jnp.mean(acc_end) / total_steps

        return chains, sigma_tot, err_sigma_tot, single_params, coupled_params, accept_rate
    

    @partial(jax.jit, static_argnames=("self"))
    def log_prior(
        self,
        LECs
    ):
        logP = - 0.5 * (
            (LECs - self.potential.LECs) ** 2 / self.prior_variance
            + jnp.log(2 * jnp.pi * self.prior_variance)
        )
        return jnp.sum(logP)

    @partial(jax.jit, static_argnames=("self"))
    def log_likelihood(
        self,
        sigmas,
        err_sigmas
    ):
    
        likelihood_variance = (self.likelihood_scale * sigmas)**2 + err_sigmas**2
            
        logP = -0.5 * (
            (sigmas - self.sigmas) ** 2 / likelihood_variance
            + jnp.log(2 * jnp.pi * likelihood_variance)
        )
        
        return jnp.sum(logP)
    
    
    @partial(jax.jit, static_argnames=("self"))
    def log_posterior(
        self,
        LECs,
        sigmas,
        err_sigmas
    ):
        return self.log_prior(LECs) + self.log_likelihood(sigmas, err_sigmas)
        

    @partial(jax.jit, static_argnames=("self"))
    def batch_log_posterior(
        self,
        LECs_batch,
        sigmas_batch,
        err_sigmas_batch,
    ):
        return jax.vmap(self.log_posterior, in_axes=(0,0,0))(LECs_batch, sigmas_batch, err_sigmas_batch)

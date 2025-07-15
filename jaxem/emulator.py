import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial
import time
from .solver import Solver
from typing import Tuple, Dict, List, Optional

import matplotlib.pyplot as plt

class Emulator:

    def __init__(
        self,
        solver: Solver,
        
    ):
        self.solver = solver
        self.mesh = solver.mesh
        self.channels = solver.channels
        self.potential = solver.potential

        
    @partial(jax.jit, static_argnames=("self"))
    def project_petrov_galerkin(
        self,
        Xqb: jnp.ndarray,
        Aqqo: jnp.ndarray,      # lhs of fom
        Vqo: jnp.ndarray,       # rhs of fom
    ):
                               
        # apply AX for all operators
        AXqob = jnp.einsum('ijo,jb->iob', Aqqo, Xqb)
        
        # construct residual subspace (also test basis in LSPG)
        # inactive columns are at the end
        Yqb = jnp.concatenate((
            Vqo,
            jnp.reshape(AXqob, (Aqqo.shape[0], -1), order='F'),
            ), axis=1
        )
        
        # orthogonalize residual subspace
        Yqb = self.qr(Yqb)
        
        # project onto residual subspace
        YAXbbo = jnp.einsum('ia,ijo,jb->abo', jnp.conjugate(Yqb), Aqqo, Xqb)
        YVbo = jnp.einsum('ib,io->bo', jnp.conjugate(Yqb), Vqo)
        
        return YAXbbo, YVbo
        
        
    @partial(jax.jit, static_argnames=("self"))
    def project_galerkin(
        self,
        Xqb: jnp.ndarray,
        Aqqo: jnp.ndarray,      # lhs of fom
        Vqo: jnp.ndarray,       # rhs of fom
    ):
        # project onto residual subspace
        XAXbbo = jnp.einsum('ia,ijo,jb->abo', jnp.conjugate(Xqb), Aqqo, Xqb)
        XVbo = jnp.einsum('ib,io->bo', jnp.conjugate(Xqb), Vqo)
        
        return XAXbbo, XVbo
        
        
    @partial(jax.jit, static_argnames=("self"))
    def estimate_error(
        self,
        LECs_candidates: jnp.ndarray,
        Csb: jnp.ndarray,
        YAXbbo: jnp.ndarray,
        YVbo: jnp.ndarray,
    ):
        YAXsb = jnp.einsum('abo,so,sb->sa', YAXbbo, LECs_candidates, Csb)
        YVsb = jnp.einsum('bo,so->sb', YVbo, LECs_candidates)
        return jnp.linalg.norm(YAXsb - YVsb, axis=1)
        
    def fit(
        self,
        LECs_candidates,
        channel: str = "1S0",
        Elab: float = 10.0,
        mode = "greedy",   # "greedy" or "pod"
        rom = "g",         # "g" or "lspg"
        **kwargs
    ):
    
        if channel in self.channels.single.keys():
        
            if mode == "greedy":
                func = self._fit_greedy_single
                
            elif mode == "pod":
                func = self._fit_pod_single
                
        elif channel in self.channels.coupled.keys():
            
            if mode == "greedy":
                func = self._fit_greedy_coupled
                
            elif mode == "pod":
                func = self._fit_pod_coupled
                
        return func(LECs_candidates, channel, Elab, mode=mode, **kwargs)
        
    
    
    def _fit_greedy_single(
        self,
        LECs_candidates,
        channel: str = "1S0",
        Elab: float = 10.0,
        n_init: int = 2,
        n_max: int = 20,
        err_tol: float = 1e-5,
        rom = "g", # or "lspg"
        mode = "greedy",
        linear_system: Tuple[jnp.ndarray, jnp.ndarray] = None
    ):

        # TODO: assert n_max <= n_mesh + 1
        
        # containers for errors and times
        errors = jnp.zeros((n_max,))
        times = {
            'setup': [],
            'orth': [],
            'g proj': [],
            'lspg proj': [],
            'solve': [],
            'error': []
        }
        setup_start = time.perf_counter()
        
        if linear_system is None:
        
            # precompute decomposed linear system for single channel
            Aqqo, Vqo = self.solver.setup_single_channel(channel, Elab)
            
        else:
            Aqqo, Vqo = linear_system
        
        
        # reserve space for maximum number of snapshots and corresponding LECs
        Tqb = jnp.zeros((self.mesh.n_mesh+1, n_max), dtype=jnp.complex128)
        LECs_train = jnp.zeros((n_max, self.potential.n_operators), dtype=jnp.complex128)
        
        # gather initial snapshots
        Tqb = Tqb.at[:,:n_init].set(
            jnp.transpose(
                self.solver.batch_solve_single_channel(
                    LECs_candidates[:n_init], Aqqo, Vqo
                )
            )
        )
        LECs_train = LECs_train.at[:n_init].set( LECs_candidates[:n_init] )
        
        times['setup'].append( time.perf_counter() - setup_start )
        
        # greedy iterations
        for i in range(n_init, n_max):
        
            # orthogonalize snapshots & zero out inactive columns
            orth_start = time.perf_counter()
            Xqb = self.qr(Tqb)
            Xqb = Xqb.at[:,i:].set(0.0)
            times['orth'].append(time.perf_counter() - orth_start)
            
            # project onto residual subspace
            lspg_proj_start = time.perf_counter()
            YAXbbo, YVbo = self.project_petrov_galerkin(Xqb, Aqqo, Vqo)
            times['lspg proj'].append(time.perf_counter() - lspg_proj_start)
        
            if rom == "g":
            
                g_proj_start = time.perf_counter()
                XAXbbo, XVbo = self.project_galerkin(Xqb, Aqqo, Vqo)
                times['g proj'].append(time.perf_counter() - g_proj_start)
                
                solve_start = time.perf_counter()
                def solve(LECs):
                    return jnp.linalg.solve(
                        jnp.einsum('abo,o->ab', XAXbbo[:i,:i], LECs),
                        jnp.einsum('bo,o->b', XVbo[:i], LECs)
                    )
                    
                
            elif rom == "lspg":
            
                solve_start = time.perf_counter()
            
                j = self.potential.n_operators * (i + 1)
            
                def solve(LECs):
                    return jnp.linalg.lstsq(
                        jnp.einsum('abo,o->ab', YAXbbo[:j,:i], LECs),
                        jnp.einsum('bo,o->b', YVbo[:j], LECs)
                    )[0]

            else:
                raise ValueError("Unknown reduced-order model {rom}. Choose 'g' or 'lspg'.")
            
            Csb = jax.vmap(solve, in_axes=(0,))(LECs_candidates)
            Csb = jnp.concatenate((Csb, jnp.zeros((LECs_candidates.shape[0], n_max-i))), axis=1)
            times['solve'].append(time.perf_counter() - solve_start)
            
            error_start = time.perf_counter()
            errors_est = self.estimate_error(LECs_candidates, Csb, YAXbbo, YVbo)
            
            # find index of worst performing candidate
            index_max_error = jnp.argmax(errors_est)
            
            # convert worst emulated candidate to full space
            Tq_em = jnp.einsum('ib,b->i', Xqb, Csb[index_max_error])
            
            # compute exact solution at candidate to add
            Tq_add = self.solver.solve_single_channel(LECs_candidates[index_max_error], Aqqo, Vqo)
            Tqb = Tqb.at[:,i].set(Tq_add)
            
            # calibrate error using real solution
            errors_cal = errors_est * jnp.linalg.norm(Tq_em - Tq_add) / errors_est[index_max_error]
            max_error_cal = jnp.max(errors_cal)
            
            times['error'].append(time.perf_counter() - error_start)
        
            if max_error_cal < err_tol:
            
                if rom == 'g':
                    emulator = (XAXbbo, XVbo)
                elif rom == 'lspg':
                    emulator = (YAXbbo, YVbo)
                    
                break
                    
                    
        stats = {
            'basis size': i+1,
            'setup': jnp.mean(jnp.array(times['setup'])),
            'orth': jnp.mean(jnp.array(times['orth'])),
            'g proj': jnp.mean(jnp.array(times['g proj'])),
            'lspg proj': jnp.mean(jnp.array(times['lspg proj'])),
            'solve': jnp.mean(jnp.array(times['solve'])),
            'error': jnp.mean(jnp.array(times['error'])),
        }
        
        stats['total'] = (
            stats['setup'] + stats['orth'] + stats['g proj']
            + stats['lspg proj'] + stats['solve'] + stats['error']
        )
        
        return errors[:i+1], Tqb[:,:i+1], LECs_train[:i+1], stats
                
                
    
        
    @partial(jax.jit, static_argnames=("self"))
    def qr(self, A):
        Q, _ = jnp.linalg.qr(A)
        return Q
        
    @partial(jax.jit, static_argnames=("self"))
    def svd(self, A):
        U, s, _ = jnp.linalg.svd(A, full_matrices=False)
        return U, s

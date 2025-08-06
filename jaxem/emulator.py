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

@contextmanager
def timed_section(time_list):
    start = time.perf_counter()
    yield
    time_list.append(time.perf_counter() - start)
    

class Emulator:

    def __init__(
        self,
        solver: Solver,
    ):
        self.solver = solver
        self.mesh = solver.mesh
        self.channels = solver.channels
        self.potential = solver.potential
        
        
    def project(
        self,
        A: jnp.ndarray,
        V: jnp.ndarray,
        test_basis: jnp.ndarray,
        trial_basis: jnp.ndarray,
    ):
        """
        Projects full-space linear system (A, V) onto reduced basis.

        Parameters
        ----------
        A : jnp.ndarray
            Full operator tensor with shape (q, q, o)
        V : jnp.ndarray
            Full right-hand side with shape (q, o)
        test_basis : jnp.ndarray
            Test basis matrix (q, a), spans residual weighting space (a = b for Galerkin)
        trial_basis : jnp.ndarray
            Trial basis matrix (q, b), spans solution space


        Returns
        -------
        A_reduced : jnp.ndarray
            Reduced operator with shape (a, b, o)
        V_reduced : jnp.ndarray
            Reduced RHS with shape (a, o)
        """
    
        A_reduced = jnp.einsum('ia,ijo,jb->abo', jnp.conjugate(test_basis), A, trial_basis)
        V_reduced  = jnp.einsum('ia,io->ao', jnp.conjugate(test_basis), V)

        return A_reduced, V_reduced
        
    def petrov_galerkin_test_basis(
        self,
        X: jnp.ndarray,
        A: jnp.ndarray,      # lhs of fom
        V: jnp.ndarray,       # rhs of fom
    ):
    
        # apply AX for all operators
        AX = jnp.einsum('ijo,jb->iob', A, X)
        
        # construct residual subspace (also test basis in LSPG)
        # inactive columns are at the end
        Y = jnp.concatenate((
            V,
            jnp.reshape(AX, (A.shape[0], -1), order='F'),
            ), axis=1
        )
        
        # orthogonalize residual subspace
        Y = self.qr(Y)
        
        return Y
        
        
    #@partial(jax.jit, static_argnames=("self"))
    def estimate_error(
        self,
        LECs_candidates: jnp.ndarray,
        C: jnp.ndarray,
        A_pg: jnp.ndarray,
        V_pg: jnp.ndarray,
    ):
        A_residual = jnp.einsum('abo,so,sb->sa', A_pg, LECs_candidates, C)
        V_residual = jnp.einsum('bo,so->sb', V_pg, LECs_candidates)
        return jnp.linalg.norm(A_residual - V_residual, axis=1)
        

    def fit(
        self,
        LECs_candidates,
        channel: str = "1S0",
        Elab: float = 10.0,
        n_init: int = 2,
        n_max: int = 25,
        tol: float = 1e-5,
        rom = "g", # or "lspg"
        mode = "greedy", # or "pod"
        linear_system: Tuple[jnp.ndarray, jnp.ndarray] = None,
    ):
        
    
        # empty containers
        model = {
            'reduced A': None,
            'reduced V': None,
            'snapshots': None,
            'LECs': None,
            'trial basis': None,
            'test basis': None,
        }
        
        errors = {
            'max': [],
            'med': [],
            'min': [],
            'p95': [],
            'p75': [],
            'p25': [],
            'p5': [],
        }
        
        times = {
            'setup': [],
            'reduce': [],
            'emulate': [],
            'solve': [],
            'error': [],
        }
        
        # choose the appropriate functions for single/coupled channels
        if channel in self.channels.single.keys():
            assert n_max <= self.mesh.n_mesh+1, "Maximum basis size is too large."
            setup_func = self.solver.setup_single_channel
            solve_func = self.solver.solve_single_channel
            
        elif channel in self.channels.coupled.keys():
            assert n_max <= 4*self.mesh.n_mesh+4, "Maximum basis size is too large."
            setup_func = self.solver.setup_coupled_channel
            solve_func = lambda LECs, A, V: (
                self.solver.solve_coupled_channel(LECs, A, V).reshape(4, -1).T.reshape(-1)
            )
            
        else:
            print("Channel {channel} not found in list.")
            
        # choose appropriate functions for galerkin or petrov-galerkin rom
        if rom == "g":
        
            test_basis_func = lambda X, A, V: X
            
            def emulate_func(LECs, A_reduced, V_reduced, i=None):
                i = i or A_reduced.shape[0]
                return jnp.linalg.solve(
                    jnp.einsum('abo,o->ab', A_reduced[:i,:i], LECs),
                    jnp.einsum('bo,o->b', V_reduced[:i], LECs)
                )
                
            def residual_basis_func(X, A, V, A_reduced, V_reduced):
                Y = self.petrov_galerkin_test_basis(X, A, V)
                print("in residual basis func for galerkin Y = ", Y.shape)
                return self.project(A, V, Y, X)

        elif rom == "lspg":
        
            test_basis_func = lambda X, A, V: self.petrov_galerkin_test_basis(X, A, V)
            
            def emulate_func(LECs, A_reduced, V_reduced, i=None):
                i = i or A_reduced.shape[0]
                j = self.potential.n_operators * (i + 1)
                return jnp.linalg.lstsq(
                    jnp.einsum('abo,o->ab', A_reduced[:j, :i], LECs),
                    jnp.einsum('bo,o->b', V_reduced[:i], LECs)
                )[0]
                
            def residual_basis_func(X, A, V, A_reduced, V_reduced):
                return A_reduced, V_reduced
        else:
            print("Unknown reduced-order model {rom}.")
        
        # setup the linear system regardless of mode
        with timed_section(times['setup']):
            if linear_system is None:
                A, V = setup_func(channel, Elab)
            else:
                A, V = linear_system
                
        # execute training
        if mode == "pod":
        
            with timed_section(times['setup']):
                LECs_train = LECs_candidates[:n_max]
                T = jax.vmap(solve_func, in_axes=(0,None,None))(LECs_train, A, V).T
                
            with timed_section(times['reduce']):
                X, singular_vals = self.svd(T)
                index = int(jnp.argmax(singular_vals/singular_vals[0] <= tol))
                n_basis = index + 1 if index > 0 else n_max
                X = X[:,:n_basis]
                Y = test_basis_func(X, A, V)
                A_reduced, V_reduced = self.project(A, V, Y, X)
                
            with timed_section(times['emulate']):
                C = jax.vmap(emulate_func, in_axes=(0,None,None))(LECs_candidates, A_reduced, V_reduced)
                
            with timed_section(times['solve']):
                # this is just to compare with emulation time
                T_ex = jax.vmap(solve_func, in_axes=(0,None,None))(LECs_candidates, A, V)
                
            with timed_section(times['error']):

                A_residual, V_residual = residual_basis_func(X, A, V, A_reduced, V_reduced)
                
                errors_est = self.estimate_error(LECs_candidates, C, A_residual, V_residual)
                index_max_error = jnp.argmax(errors_est)
                
                # convert worst emulated candidate to full space
                t_em = jnp.einsum('ib,b->i', X, C[index_max_error])
                
                # compute exact solution at worst point
                t_ex = solve_func(LECs_candidates[index_max_error], A, V)
                
                # calibrate error using real solution
                errors_cal = jnp.linalg.norm(t_em - t_ex) * errors_est / errors_est[index_max_error]

                errors['min'].append( jnp.min(errors_cal) )
                errors['med'].append( jnp.median(errors_cal) )
                errors['max'].append( jnp.max(errors_cal) )
                errors['p5'].append( jnp.percentile(errors_cal, 5.) )
                errors['p25'].append( jnp.percentile(errors_cal, 25.) )
                errors['p75'].append( jnp.percentile(errors_cal, 75.) )
                errors['p95'].append( jnp.percentile(errors_cal, 95.) )
        
        elif mode == "greedy":
        
            # collect n_init snapshots, leaving room for n_max snapshots
            with timed_section(times['setup']):
                T = jnp.zeros((self.mesh.n_mesh+1, n_max), dtype=jnp.complex128)
                T = T.at[:, :n_init].set(
                    jax.vmap(solve_func, in_axes=(0,None,None))(LECs_candidates[:n_init], A, V).T
                )
            
                LECs_train = jnp.zeros((n_max, self.potential.n_operators), dtype=jnp.complex128)
                LECs_train = LECs_train.at[:n_init].set( LECs_candidates[:n_init] )
                
            for i in range(n_init, n_max):
            
                with timed_section(times['reduce']):
                    X = self.qr(T)
                    X = X.at[:,i:].set(0.0)
                    Y = test_basis_func(X, A, V)
                    A_reduced, V_reduced = self.project(A, V, Y, X)
                
                with timed_section(times['emulate']):
                    C = jax.vmap(emulate_func, in_axes=(0,None,None,None))(LECs_candidates, A_reduced, V_reduced, i)
                    
                    print("\n\n C * ", C.shape)
                    C = jnp.concatenate((C, jnp.zeros((LECs_candidates.shape[0], n_max-i))), axis=1)
                    print("\n\n LECs * ", LECs_candidates.shape)
                    print("\n\n A red * ", A_reduced.shape)
                    print("\n\n V red * ", V_reduced.shape)
                    print("\n\n C * ", C.shape)
                    
                with timed_section(times['solve']):
                    # this is just to compare with emulation time
                    T_ex = jax.vmap(solve_func, in_axes=(0,None,None))(LECs_candidates, A, V)
                    
                with timed_section(times['error']):

                    A_residual, V_residual = residual_basis_func(X, A, V, A_reduced, V_reduced)
                    
                    print("LECs_candidates = ", LECs_candidates.shape)
                    print("C = ", C.shape)
                    print("YAX = ", A_residual.shape)
                    print("YV = ", V_residual.shape)
                    
                    errors_est = self.estimate_error(LECs_candidates, C, A_residual, V_residual)
                    
                    
                    index_max_error = jnp.argmax(errors_est)
                    
                    # convert worst emulated candidate to full space
                    t_em = jnp.einsum('ib,b->i', X, C[index_max_error])

                    # compute exact solution at worst point
                    t_ex = solve_func(LECs_candidates[index_max_error], A, V)
                    
                    # add point
                    T = T.at[:,i].set(t_ex)
                    LECs_train = LECs_train.at[i].set(LECs_candidates[index_max_error])

                    # calibrate error using real solution
                    errors_cal = jnp.linalg.norm(t_em - t_ex) * errors_est / errors_est[index_max_error]
                    
                    
                    
                    errors['min'].append( jnp.min(errors_cal) )
                    errors['med'].append( jnp.median(errors_cal) )
                    errors['max'].append( jnp.max(errors_cal) )
                    errors['p5'].append( jnp.percentile(errors_cal, 5.) )
                    errors['p25'].append( jnp.percentile(errors_cal, 25.) )
                    errors['p75'].append( jnp.percentile(errors_cal, 75.) )
                    errors['p95'].append( jnp.percentile(errors_cal, 95.) )
                    
                if jnp.max(errors_cal) < tol:
                    break
        
        model['reduced A'] = A_reduced
        model['reduced V'] = V_reduced
        model['trial basis'] = X
        model['test basis'] = Y
        model['snapshots'] = T
        model['LECs'] = LECs_train
        
        for dict in [model, errors, times]:
            for key, val in dict.items():
                dict[key] = jnp.array(val)
                
        return model, errors, times
        

    @partial(jax.jit, static_argnames=("self"))
    def qr(self, A):
        Q, _ = jnp.linalg.qr(A)
        return Q
        
        
    @partial(jax.jit, static_argnames=("self"))
    def svd(self, A):
        U, s, _ = jnp.linalg.svd(A, full_matrices=False)
        return U, s


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
import json
import glob

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
        
        # containers for emulators
        def nested_Nones(m, n):
            return [[None for _ in range(n)] for _ in range(m)]
            
        nc = self.channels.n_single
        ncc = self.channels.n_coupled
        nk = self.solver.n_poles
        
        self.onshell_t_ck  = nested_Nones(nc, nk)
        self.onshell_t_cck = nested_Nones(ncc, nk)
        self.t_ck  = nested_Nones(nc, nk)
        self.t_cck = nested_Nones(ncc, nk)
        self.error_ck  = nested_Nones(nc, nk)
        self.error_cck = nested_Nones(ncc, nk)


    def fit(
        self,
        LECs_candidates,
        rom='g',
        mode='pod',
        n_init=2,
        n_max=100,
        tol=1e-5
    ):
    
        if rom == 'g' and mode == 'pod':
        
            def fit_single_channel(LECs, c, k):
                return self.fit_single_channel_pod_g(LECs_candidates, c, k, n_max=n_max, tol=tol)
                
            def fit_coupled_channel(LECs, cc, k):
                return self.fit_coupled_channel_pod_g(LECs_candidates, cc, k, n_max=n_max, tol=tol)
                
        elif rom == 'g' and mode == 'greedy':
            
            def fit_single_channel(LECs, c, k):
                return self.fit_single_channel_greedy_g(LECs_candidates, c, k, n_init=n_init, n_max=n_max, tol=tol)
                
            def fit_coupled_channel(LECs, cc, k):
                return self.fit_coupled_channel_greedy_g(LECs_candidates, cc, k, n_init=n_init, n_max=n_max, tol=tol)
                
        if rom == 'lspg' and mode == 'pod':
        
            def fit_single_channel(LECs, c, k):
                return self.fit_single_channel_pod_lspg(LECs_candidates, c, k, n_max=n_max, tol=tol)
                
            def fit_coupled_channel(LECs, cc, k):
                return self.fit_coupled_channel_pod_lspg(LECs_candidates, cc, k, n_max=n_max, tol=tol)
                
        elif rom == 'lspg' and mode == 'greedy':

            def fit_single_channel(LECs, c, k):
                return self.fit_single_channel_greedy_lspg(LECs_candidates, c, k, n_init=n_init, n_max=n_max, tol=tol)
                
            def fit_coupled_channel(LECs, cc, k):
                return self.fit_coupled_channel_greedy_lspg(LECs_candidates, cc, k, n_init=n_init, n_max=n_max, tol=tol)
            
        # loop through all energies and channels
        for k in range(self.solver.n_poles):
        
            for c in range(self.channels.n_single):
                fit_single_channel(LECs_candidates, c, k)
                
            for cc in range(self.channels.n_coupled):
                fit_coupled_channel(LECs_candidates, cc, k)
                

        # stack jitted emulator functions in a grid
        def make_grid_funcs(funcs_2d):

            m = len(funcs_2d)
            n = len(funcs_2d[0])

            def apply_all(LECs):
                rows = []
                for i in range(m):
                    row = []
                    for j in range(n):
                        row.append(funcs_2d[i][j](LECs))
                    rows.append(jnp.stack(row, axis=0))
                return jnp.stack(rows, axis=0)

            return jax.jit(apply_all)
            
            
        # store emulators for all energies and channels
        self.emulate_t_ck = make_grid_funcs(self.t_ck)
        self.emulate_t_cck = make_grid_funcs(self.t_cck)
        
        self.emulate_onshell_t_ck = make_grid_funcs(self.onshell_t_ck)
        self.emulate_onshell_t_cck = make_grid_funcs(self.onshell_t_cck)
        
        self.estimate_error_ck = make_grid_funcs(self.error_ck)
        self.estimate_error_cck = make_grid_funcs(self.error_cck)
        



    def fit_single_channel_pod_g(
        self,
        LECs_candidates,
        c: int,
        k: int,
        n_max: int = 100,  # maximum number of snapshots to select from candidates
        tol: float = 1e-5
    ):
    
        # select training data and get high-fidelity solutions
        LECs_train = LECs_candidates[:n_max]
        
        T = jax.vmap(
            self.solver.single_channel_t,
            in_axes=(0,None,None)
        )(LECs_train, c, k).T

        # orthogonalize basis and truncate
        X, singular_vals = self.svd(T)
        index = int(jnp.argmax(singular_vals/singular_vals[0] <= tol))
        n_basis = index + 1 if index > 0 else n_max
        X = X[:,:n_basis]
        
        # project operators
        VG, V = self.solver.single_channel_operators(c, k)
        VG_reduced = jnp.einsum('ia,ijo,jb->abo', jnp.conjugate(X), VG, X)
        V_reduced  = jnp.einsum('ia,io->ao', jnp.conjugate(X), V)
        
        Y = self.residual_basis(VG, V, X)
        I_residual = jnp.einsum('ia,ib->ab', jnp.conjugate(Y), X)
        VG_residual = jnp.einsum('ia,ijo,jb->abo', jnp.conjugate(Y), VG, X)
        V_residual  = jnp.einsum('ia,io->ao', jnp.conjugate(Y), V)
        
        
        def reduced_solve(LECs):
            return jnp.linalg.solve(
                jnp.identity(VG_reduced.shape[0]) - VG_reduced @ LECs,
                V_reduced @ LECs
            )
        
        def emulate_t(LECs):
            C = reduced_solve(LECs)
            return X @ C
            
        def emulate_onshell_t(LECs):
            C = reduced_solve(LECs)
            return X[0] @ C
            
        def estimate_residual(LECs):
            C = reduced_solve(LECs)
            return jnp.linalg.norm(
                I_residual @ C
                - (VG_residual @ LECs) @ C
                - V_residual @ LECs
            )
            
        # solve all candidates in reduced space, estimate residuals
        estimated_residuals = jax.vmap(
            estimate_residual,
            in_axes=(0,)
        )(LECs_candidates)
        
        # find calibration ratio
        index_max = jnp.argmax(estimated_residuals)
        t_em = emulate_t(LECs_candidates[index_max])
        t_ex = self.solver.single_channel_t(LECs_candidates[index_max], c, k)
        
        calibration_ratio = jnp.linalg.norm(t_em - t_ex) / estimated_residuals[index_max]
        calibrated_residuals = calibration_ratio * estimated_residuals
        
        def estimate_error(LECs):
            return calibration_ratio * estimate_residual(LECs)
        
        # store emulator
        self.t_ck[c][k] = jax.jit(emulate_t)
        self.onshell_t_ck[c][k] = jax.jit(emulate_onshell_t)
        self.error_ck[c][k] = jax.jit(estimate_error)
            


    def fit_coupled_channel_pod_g(
        self,
        LECs_candidates,
        cc: int,
        k: int,
        n_max: int = 100,  # maximum number of snapshots to select from candidates
        tol: float = 1e-5
    ):
    
        # select training data and get high-fidelity solutions
        LECs_train = LECs_candidates[:n_max]
        
        T = jax.vmap(
            self.solver.coupled_channel_t_flat,
            in_axes=(0,None,None)
        )(LECs_train, cc, k).T

        # orthogonalize basis and truncate
        X, singular_vals = self.svd(T)
        index = int(jnp.argmax(singular_vals/singular_vals[0] <= tol))
        n_basis = index + 1 if index > 0 else n_max
        X = X[:,:n_basis]
        
        # project operators
        VG, V = self.solver.coupled_channel_operators_flat(cc, k)
        VG_reduced = jnp.einsum('ia,ijo,jb->abo', jnp.conjugate(X), VG, X)
        V_reduced  = jnp.einsum('ia,io->ao', jnp.conjugate(X), V)
        
        Y = self.residual_basis(VG, V, X)
        I_residual = jnp.einsum('ia,ib->ab', jnp.conjugate(Y), X)
        VG_residual = jnp.einsum('ia,ijo,jb->abo', jnp.conjugate(Y), VG, X)
        V_residual  = jnp.einsum('ia,io->ao', jnp.conjugate(Y), V)
        
        nq = self.mesh.n_mesh
        self.onshell_elems = jnp.array([0, nq+1, 2*nq+2, 3*nq+3])
        
        def reduced_solve(LECs):
            return jnp.linalg.solve(
                jnp.identity(VG_reduced.shape[0]) - VG_reduced @ LECs,
                V_reduced @ LECs
            )
        
        def emulate_t(LECs):
            C = reduced_solve(LECs)
            return X @ C
            
        def emulate_onshell_t(LECs):
            C = reduced_solve(LECs)
            return X[self.onshell_elems] @ C
            
        def estimate_residual(LECs):
            C = reduced_solve(LECs)
            return jnp.linalg.norm(
                I_residual @ C
                - (VG_residual @ LECs) @ C
                - V_residual @ LECs
            )
            
        # estimate residuals in reduced space
        estimated_residuals = jax.vmap(
            estimate_residual,
            in_axes=(0,)
        )(LECs_candidates)
        
        # find calibration ratio
        index_max = jnp.argmax(estimated_residuals)
        t_em = emulate_t(LECs_candidates[index_max])
        t_ex = self.solver.coupled_channel_t_flat(LECs_candidates[index_max], cc, k)
        
        calibration_ratio = jnp.linalg.norm(t_em - t_ex) / estimated_residuals[index_max]
        calibrated_residuals = calibration_ratio * estimated_residuals
        
        def estimate_error(LECs):
            return calibration_ratio * estimate_residual(LECs)
            
        # estimate onshell error ?
        
        # store emulator
        self.t_cck[cc][k] = jax.jit(emulate_t)
        self.onshell_t_cck[cc][k] = jax.jit(emulate_onshell_t)
        self.error_cck[cc][k] = jax.jit(estimate_error)
        
        

    def fit_single_channel_greedy_g(
        self,
        LECs_candidates,
        c: int,
        k: int,
        n_init: int = 2,
        n_max: int = 100,
        tol: float = 1e-5
    ):

        n_dim = self.mesh.n_mesh + 1
        T = jnp.zeros((n_dim, n_max), dtype=jnp.complex128)
        
        # collect n_init snapshots, leaving room for n_max snapshots
        T = T.at[:, :n_init].set(
            jax.vmap(self.solver.single_channel_t, in_axes=(0, None, None))(
                LECs_candidates[:n_init], c, k
            ).T
        )
        
        LECs_train = jnp.zeros((n_max, self.potential.n_operators), dtype=jnp.complex128)
        LECs_train = LECs_train.at[:n_init].set(LECs_candidates[:n_init])
        
        # prestore operators
        VG, V = self.solver.single_channel_operators(c, k)

        # --- helper to build reduced quantities and emulators ---
        def build_reduced_emulator(X):
            VG_reduced = jnp.einsum('ia,ijo,jb->abo', jnp.conjugate(X), VG, X)
            V_reduced  = jnp.einsum('ia,io->ao', jnp.conjugate(X), V)
            
            Y = self.residual_basis(VG, V, X)
            I_residual = jnp.einsum('ia,ib->ab', jnp.conjugate(Y), X)
            VG_residual = jnp.einsum('ia,ijo,jb->abo', jnp.conjugate(Y), VG, X)
            V_residual  = jnp.einsum('ia,io->ao', jnp.conjugate(Y), V)
            
            def reduced_solve(LECs):
                return jnp.linalg.solve(
                    jnp.identity(VG_reduced.shape[0]) - VG_reduced @ LECs,
                    V_reduced @ LECs
                )
            
            def emulate_t(LECs):
                C = reduced_solve(LECs)
                return X @ C
                
            def emulate_onshell_t(LECs):
                C = reduced_solve(LECs)
                return X[0] @ C
                
            def estimate_residual(LECs):
                C = reduced_solve(LECs)
                return jnp.linalg.norm(
                    I_residual @ C
                    - (VG_residual @ LECs) @ C
                    - V_residual @ LECs
                )
            
            return emulate_t, emulate_onshell_t, estimate_residual

        # --- main greedy loop ---
        for i in range(n_init, n_max):

            # orthogonalize current snapshots
            X = self.qr(T)
            X = X.at[:, i:].set(0.0)

            emulate_t, emulate_onshell_t, estimate_residual = build_reduced_emulator(X)
            
            estimated_residuals = jax.vmap(estimate_residual)(LECs_candidates)
            
            index_max = jnp.argmax(estimated_residuals)
            t_em = emulate_t(LECs_candidates[index_max])
            t_ex = self.solver.single_channel_t(LECs_candidates[index_max], c, k)
            
            calibration_ratio = (
                jnp.linalg.norm(t_em - t_ex) / estimated_residuals[index_max]
            )
            calibrated_residuals = calibration_ratio * estimated_residuals
            
            # add exact solution to basis
            T = T.at[:, i].set(t_ex)
            LECs_train = LECs_train.at[i].set(LECs_candidates[index_max])

            def estimate_error(LECs):
                return calibration_ratio * estimate_residual(LECs)
            
            if jnp.max(calibrated_residuals) < tol:
                # recompute final emulator cleanly
                X = self.qr(T)[:, :i]
                emulate_t, emulate_onshell_t, estimate_residual = build_reduced_emulator(X)
                estimated_residuals = jax.vmap(estimate_residual)(LECs_candidates)

                index_max = jnp.argmax(estimated_residuals)
                t_em = emulate_t(LECs_candidates[index_max])
                t_ex = self.solver.single_channel_t(LECs_candidates[index_max], c, k)
                
                calibration_ratio = (
                    jnp.linalg.norm(t_em - t_ex) / estimated_residuals[index_max]
                )
                def estimate_error(LECs):
                    return calibration_ratio * estimate_residual(LECs)
                
                break

        # store emulator
        self.t_ck[c][k] = jax.jit(emulate_t)
        self.onshell_t_ck[c][k] = jax.jit(emulate_onshell_t)
        self.error_ck[c][k] = jax.jit(estimate_error)

        

    def fit_coupled_channel_greedy_g(
        self,
        LECs_candidates,
        cc: int,
        k: int,
        n_init: int = 2,
        n_max: int = 100,
        tol: float = 1e-5
    ):

        n_dim = (self.mesh.n_mesh + 1) * 4  # 4 coupled channels
        T = jnp.zeros((n_dim, n_max), dtype=jnp.complex128)
        
        # collect n_init snapshots, leaving room for n_max snapshots
        T = T.at[:, :n_init].set(
            jax.vmap(self.solver.coupled_channel_t_flat, in_axes=(0, None, None))(
                LECs_candidates[:n_init], cc, k
            ).T
        )

        LECs_train = jnp.zeros((n_max, self.potential.n_operators), dtype=jnp.complex128)
        LECs_train = LECs_train.at[:n_init].set(LECs_candidates[:n_init])
        
        # prestore operators
        VG, V = self.solver.coupled_channel_operators_flat(cc, k)

        nq = self.mesh.n_mesh
        self.onshell_elems = jnp.array([0, nq+1, 2*nq+2, 3*nq+3])

        # --- helper to build reduced quantities and emulators ---
        def build_reduced_emulator(X):
            VG_reduced = jnp.einsum('ia,ijo,jb->abo', jnp.conjugate(X), VG, X)
            V_reduced  = jnp.einsum('ia,io->ao', jnp.conjugate(X), V)
            
            Y = self.residual_basis(VG, V, X)
            I_residual = jnp.einsum('ia,ib->ab', jnp.conjugate(Y), X)
            VG_residual = jnp.einsum('ia,ijo,jb->abo', jnp.conjugate(Y), VG, X)
            V_residual  = jnp.einsum('ia,io->ao', jnp.conjugate(Y), V)
            
            def reduced_solve(LECs):
                return jnp.linalg.solve(
                    jnp.identity(VG_reduced.shape[0]) - VG_reduced @ LECs,
                    V_reduced @ LECs
                )
            
            def emulate_t(LECs):
                C = reduced_solve(LECs)
                return X @ C
                
            def emulate_onshell_t(LECs):
                C = reduced_solve(LECs)
                return X[self.onshell_elems] @ C
                
            def estimate_residual(LECs):
                C = reduced_solve(LECs)
                return jnp.linalg.norm(
                    I_residual @ C
                    - (VG_residual @ LECs) @ C
                    - V_residual @ LECs
                )
            
            return emulate_t, emulate_onshell_t, estimate_residual

        # --- main greedy loop ---
        for i in range(n_init, n_max):

            # orthogonalize current snapshots
            X = self.qr(T)
            X = X.at[:, i:].set(0.0)

            emulate_t, emulate_onshell_t, estimate_residual = build_reduced_emulator(X)
            
            estimated_residuals = jax.vmap(estimate_residual)(LECs_candidates)
            
            index_max = jnp.argmax(estimated_residuals)
            t_em = emulate_t(LECs_candidates[index_max])
            t_ex = self.solver.coupled_channel_t_flat(LECs_candidates[index_max], cc, k)
            
            calibration_ratio = (
                jnp.linalg.norm(t_em - t_ex) / estimated_residuals[index_max]
            )
            calibrated_residuals = calibration_ratio * estimated_residuals
            
            # add exact solution to basis
            T = T.at[:, i].set(t_ex)
            LECs_train = LECs_train.at[i].set(LECs_candidates[index_max])

            def estimate_error(LECs):
                return calibration_ratio * estimate_residual(LECs)

            
            if jnp.max(calibrated_residuals) < tol:
                # recompute final emulator cleanly
                X = self.qr(T)[:, :i]
                emulate_t, emulate_onshell_t, estimate_residual = build_reduced_emulator(X)
                estimated_residuals = jax.vmap(estimate_residual)(LECs_candidates)

                index_max = jnp.argmax(estimated_residuals)
                t_em = emulate_t(LECs_candidates[index_max])
                t_ex = self.solver.coupled_channel_t_flat(LECs_candidates[index_max], cc, k)
                
                calibration_ratio = (
                    jnp.linalg.norm(t_em - t_ex) / estimated_residuals[index_max]
                )
                def estimate_error(LECs):
                    return calibration_ratio * estimate_residual(LECs)
                
                break

        # store emulator
        self.t_cck[cc][k] = jax.jit(emulate_t)
        self.onshell_t_cck[cc][k] = jax.jit(emulate_onshell_t)
        self.error_cck[cc][k] = jax.jit(estimate_error)


    def fit_coupled_channel_greedy_lspg(
        self,
        LECs_candidates,
        cc: int,
        k: int,
        n_init: int = 2,
        n_max: int = 100,
        tol: float = 1e-5
    ):
        """
        Greedy LSPG (Petrov–Galerkin) reduced-order model for coupled channels.

        Uses a trial basis X and a test basis Y (from residual_basis),
        solving the least-squares problem (A C ≈ b) with jnp.linalg.lstsq.
        """

        n_dim = (self.mesh.n_mesh + 1) * 4
        T = jnp.zeros((n_dim, n_max), dtype=jnp.complex128)

        # collect n_init snapshots
        T = T.at[:, :n_init].set(
            jax.vmap(self.solver.coupled_channel_t_flat, in_axes=(0, None, None))(LECs_candidates[:n_init], cc, k).T
        )

        LECs_train = jnp.zeros((n_max, self.potential.n_operators), dtype=jnp.complex128)
        LECs_train = LECs_train.at[:n_init].set(LECs_candidates[:n_init])

        # prestore operators
        VG, V = self.solver.coupled_channel_operators_flat(cc, k)

        nq = self.mesh.n_mesh
        self.onshell_elems = jnp.array([0, nq + 1, 2 * nq + 2, 3 * nq + 3])

        for i in range(n_init, n_max):
        
            # orthogonalize current snapshots
            X = self.qr(T)
            X = X.at[:, i:].set(0.0)

            # compute test basis (Petrov-Galerkin)
            Y = self.residual_basis(VG, V, X)

            # project operators
            I_reduced = jnp.einsum("ia,ib->ab", jnp.conjugate(Y), X)
            VG_reduced = jnp.einsum("ia,ijo,jb->abo", jnp.conjugate(Y), VG, X)
            V_reduced = jnp.einsum("ia,io->ao", jnp.conjugate(Y), V)

            def reduced_solve(LECs):
                # A = I_reduced - VG_reduced @ LECs
                A = I_reduced - jnp.einsum("abo,o->ab", VG_reduced, LECs)
                b = jnp.einsum("ao,o->a", V_reduced, LECs)
                # least squares solve (A C ≈ b)
                C, *_ = jnp.linalg.lstsq(A, b, rcond=None)
                return C

            def emulate_t(LECs):
                C = reduced_solve(LECs)
                return X @ C

            def emulate_onshell_t(LECs):
                C = reduced_solve(LECs)
                return X[self.onshell_elems] @ C

            def estimate_residual(LECs):
                C = reduced_solve(LECs)
                return jnp.linalg.norm(
                    I_reduced @ C
                    - (VG_reduced @ LECs) @ C
                    - V_reduced @ LECs
                )

            # estimate residuals for all candidates
            estimated_residuals = jax.vmap(estimate_residual, in_axes=(0,))(LECs_candidates)

            # find point of max residual
            index_max = jnp.argmax(estimated_residuals)
            t_em = emulate_t(LECs_candidates[index_max])
            t_ex = self.solver.coupled_channel_t_flat(LECs_candidates[index_max], cc, k)

            # calibration ratio
            calibration_ratio = jnp.linalg.norm(t_em - t_ex) / estimated_residuals[index_max]
            calibrated_residuals = calibration_ratio * estimated_residuals

            # add new exact solution
            T = T.at[:, i].set(t_ex)
            LECs_train = LECs_train.at[i].set(LECs_candidates[index_max])

            def estimate_error(LECs):
                return calibration_ratio * estimate_residual(LECs)

            if jnp.max(calibrated_residuals) < tol:

                # recompute X, Y, and reduced operators
                X = self.qr(T)[:, :i]
                Y = self.residual_basis(VG, V, X)

                I_reduced = jnp.einsum("ia,ib->ab", jnp.conjugate(Y), X)
                VG_reduced = jnp.einsum("ia,ijo,jb->abo", jnp.conjugate(Y), VG, X)
                V_reduced = jnp.einsum("ia,io->ao", jnp.conjugate(Y), V)

                def reduced_solve(LECs):
                    A = I_reduced - jnp.einsum("abo,o->ab", VG_reduced, LECs)
                    b = jnp.einsum("ao,o->a", V_reduced, LECs)
                    C, *_ = jnp.linalg.lstsq(A, b, rcond=None)
                    return C

                def emulate_t(LECs):
                    C = reduced_solve(LECs)
                    return X @ C

                def emulate_onshell_t(LECs):
                    C = reduced_solve(LECs)
                    return X[self.onshell_elems] @ C

                def estimate_residual(LECs):
                    C = reduced_solve(LECs)
                    return jnp.linalg.norm(
                        I_reduced @ C
                        - (VG_reduced @ LECs) @ C
                        - V_reduced @ LECs
                    )

                estimated_residuals = jax.vmap(estimate_residual, in_axes=(0,))(LECs_candidates)
                index_max = jnp.argmax(estimated_residuals)
                t_em = emulate_t(LECs_candidates[index_max])
                t_ex = self.solver.coupled_channel_t_flat(LECs_candidates[index_max], cc, k)

                calibration_ratio = jnp.linalg.norm(t_em - t_ex) / estimated_residuals[index_max]
                calibrated_residuals = calibration_ratio * estimated_residuals

                def estimate_error(LECs):
                    return calibration_ratio * estimate_residual(LECs)

                break

        # store emulator
        self.t_cck[cc][k] = jax.jit(emulate_t)
        self.onshell_t_cck[cc][k] = jax.jit(emulate_onshell_t)
        self.error_cck[cc][k] = jax.jit(estimate_error)



    def fit_single_channel_greedy_lspg(
        self,
        LECs_candidates,
        c: int,
        k: int,
        n_init: int = 2,
        n_max: int = 100,
        tol: float = 1e-5
    ):
        """
        Greedy LSPG (Petrov–Galerkin) for single channel using jnp.linalg.lstsq.
        """

        n_dim = self.mesh.n_mesh + 1
        T = jnp.zeros((n_dim, n_max), dtype=jnp.complex128)

        # collect n_init snapshots
        T = T.at[:, :n_init].set(
            jax.vmap(self.solver.single_channel_t, in_axes=(0, None, None))(LECs_candidates[:n_init], c, k).T
        )

        LECs_train = jnp.zeros((n_max, self.potential.n_operators), dtype=jnp.complex128)
        LECs_train = LECs_train.at[:n_init].set(LECs_candidates[:n_init])

        # prestore operators
        VG, V = self.solver.single_channel_operators(c, k)

        # main greedy loop
        for i in range(n_init, n_max):

            # orthogonalize current snapshots
            X = self.qr(T)
            X = X.at[:, i:].set(0.0)

            # test basis (residual basis)
            Y = self.residual_basis(VG, V, X)

            # reduced/projected operators (Petrov-Galerkin)
            I_reduced = jnp.einsum('ia,ib->ab', jnp.conjugate(Y), X)           # n_test x n_basis
            VG_reduced = jnp.einsum('ia,ijo,jb->abo', jnp.conjugate(Y), VG, X)
            V_reduced  = jnp.einsum('ia,io->ao', jnp.conjugate(Y), V)

            def reduced_solve(LECs):
                A = I_reduced - jnp.einsum('abo,o->ab', VG_reduced, LECs)   # n_test x n_basis
                b = jnp.einsum('ao,o->a', V_reduced, LECs)                 # n_test
                C, *_ = jnp.linalg.lstsq(A, b, rcond=None)
                return C

            def emulate_t(LECs):
                C = reduced_solve(LECs)
                return X @ C

            def emulate_onshell_t(LECs):
                C = reduced_solve(LECs)
                return X[0] @ C

            def estimate_residual(LECs):
                C = reduced_solve(LECs)
                # evaluate residual in reduced (Petrov) form like before
                return jnp.linalg.norm(
                    I_reduced @ C
                    - (VG_reduced @ LECs) @ C
                    - V_reduced @ LECs
                )

            # estimate residuals for all candidates
            estimated_residuals = jax.vmap(estimate_residual, in_axes=(0,))(LECs_candidates)

            # greedy selection
            index_max = jnp.argmax(estimated_residuals)
            t_em = emulate_t(LECs_candidates[index_max])
            t_ex = self.solver.single_channel_t(LECs_candidates[index_max], c, k)

            calibration_ratio = jnp.linalg.norm(t_em - t_ex) / estimated_residuals[index_max]
            calibrated_residuals = calibration_ratio * estimated_residuals

            # add exact solution to basis
            T = T.at[:, i].set(t_ex)
            LECs_train = LECs_train.at[i].set(LECs_candidates[index_max])

            def estimate_error(LECs):
                return calibration_ratio * estimate_residual(LECs)


            if jnp.max(calibrated_residuals) < tol:
                # recompute final X,Y and reduced ops using basis up to i
                X = self.qr(T)[:, :i]
                Y = self.residual_basis(VG, V, X)

                I_reduced = jnp.einsum('ia,ib->ab', jnp.conjugate(Y), X)
                VG_reduced = jnp.einsum('ia,ijo,jb->abo', jnp.conjugate(Y), VG, X)
                V_reduced  = jnp.einsum('ia,io->ao', jnp.conjugate(Y), V)

                def reduced_solve(LECs):
                    A = I_reduced - jnp.einsum('abo,o->ab', VG_reduced, LECs)
                    b = jnp.einsum('ao,o->a', V_reduced, LECs)
                    C, *_ = jnp.linalg.lstsq(A, b, rcond=None)
                    return C

                def emulate_t(LECs):
                    C = reduced_solve(LECs)
                    return X @ C

                def emulate_onshell_t(LECs):
                    C = reduced_solve(LECs)
                    return X[0] @ C

                def estimate_residual(LECs):
                    C = reduced_solve(LECs)
                    return jnp.linalg.norm(
                        I_reduced @ C
                        - (VG_reduced @ LECs) @ C
                        - V_reduced @ LECs
                    )

                estimated_residuals = jax.vmap(estimate_residual, in_axes=(0,))(LECs_candidates)
                index_max = jnp.argmax(estimated_residuals)
                t_em = emulate_t(LECs_candidates[index_max])
                t_ex = self.solver.single_channel_t(LECs_candidates[index_max], c, k)

                calibration_ratio = jnp.linalg.norm(t_em - t_ex) / estimated_residuals[index_max]

                def estimate_error(LECs):
                    return calibration_ratio * estimate_residual(LECs)

                break

        # store emulator
        self.t_ck[c][k] = jax.jit(emulate_t)
        self.onshell_t_ck[c][k] = jax.jit(emulate_onshell_t)
        self.error_ck[c][k] = jax.jit(estimate_error)



    def fit_single_channel_pod_lspg(
        self,
        LECs_candidates,
        c: int,
        k: int,
        n_max: int = 100,
        tol: float = 1e-5
    ):
        """
        POD + LSPG for single channel using jnp.linalg.lstsq.
        """

        # select training data and get high-fidelity solutions
        LECs_train = LECs_candidates[:n_max]

        T = jax.vmap(self.solver.single_channel_t, in_axes=(0, None, None))(LECs_train, c, k).T

        # orthogonalize basis and truncate (SVD)
        X, singular_vals = self.svd(T)
        index = int(jnp.argmax(singular_vals / singular_vals[0] <= tol))
        n_basis = index + 1 if index > 0 else n_max
        X = X[:, :n_basis]

        # project operators (full)
        VG, V = self.solver.single_channel_operators(c, k)

        # test basis (Petrov-Galerkin)
        Y = self.residual_basis(VG, V, X)

        # reduced/projected operators
        I_reduced = jnp.einsum('ia,ib->ab', jnp.conjugate(Y), X)
        VG_reduced = jnp.einsum('ia,ijo,jb->abo', jnp.conjugate(Y), VG, X)
        V_reduced  = jnp.einsum('ia,io->ao', jnp.conjugate(Y), V)

        def reduced_solve(LECs):
            A = I_reduced - jnp.einsum('abo,o->ab', VG_reduced, LECs)
            b = jnp.einsum('ao,o->a', V_reduced, LECs)
            C, *_ = jnp.linalg.lstsq(A, b, rcond=None)
            return C

        def emulate_t(LECs):
            C = reduced_solve(LECs)
            return X @ C

        def emulate_onshell_t(LECs):
            C = reduced_solve(LECs)
            return X[0] @ C

        def estimate_residual(LECs):
            C = reduced_solve(LECs)
            return jnp.linalg.norm(
                (jnp.identity(VG.shape[0]) - jnp.einsum('ijo,o->ij', VG, LECs)) @ (X @ C)
                - jnp.einsum('io,o->i', V, LECs)
            )

        # estimate residuals in reduced/test space
        estimated_residuals = jax.vmap(estimate_residual, in_axes=(0,))(LECs_candidates)

        index_max = jnp.argmax(estimated_residuals)
        t_em = emulate_t(LECs_candidates[index_max])
        t_ex = self.solver.single_channel_t(LECs_candidates[index_max], c, k)

        calibration_ratio = jnp.linalg.norm(t_em - t_ex) / estimated_residuals[index_max]

        def estimate_error(LECs):
            return calibration_ratio * estimate_residual(LECs)

        # store emulator
        self.t_ck[c][k] = jax.jit(emulate_t)
        self.onshell_t_ck[c][k] = jax.jit(emulate_onshell_t)
        self.error_ck[c][k] = jax.jit(estimate_error)



    def fit_coupled_channel_pod_lspg(
        self,
        LECs_candidates,
        cc: int,
        k: int,
        n_max: int = 40,
        tol: float = 1e-5
    ):
        """
        POD + LSPG for coupled channels using jnp.linalg.lstsq.
        """

        # select training data and get high-fidelity solutions
        LECs_train = LECs_candidates[:n_max]

        T = jax.vmap(self.solver.coupled_channel_t_flat, in_axes=(0, None, None))(LECs_train, cc, k).T

        # orthogonalize basis and truncate (SVD)
        X, singular_vals = self.svd(T)
        index = int(jnp.argmax(singular_vals / singular_vals[0] <= tol))
        n_basis = index + 1 if index > 0 else n_max
        X = X[:, :n_basis]

        # full operators
        VG, V = self.solver.coupled_channel_operators_flat(cc, k)

        # test basis
        Y = self.residual_basis(VG, V, X)

        # reduced/projected operators
        I_reduced = jnp.einsum('ia,ib->ab', jnp.conjugate(Y), X)
        VG_reduced = jnp.einsum('ia,ijo,jb->abo', jnp.conjugate(Y), VG, X)
        V_reduced  = jnp.einsum('ia,io->ao', jnp.conjugate(Y), V)

        nq = self.mesh.n_mesh
        self.onshell_elems = jnp.array([0, nq+1, 2*nq+2, 3*nq+3])

        def reduced_solve(LECs):
            A = I_reduced - jnp.einsum('abo,o->ab', VG_reduced, LECs)
            b = jnp.einsum('ao,o->a', V_reduced, LECs)
            C, *_ = jnp.linalg.lstsq(A, b, rcond=None)
            return C

        def emulate_t(LECs):
            C = reduced_solve(LECs)
            return X @ C

        def emulate_onshell_t(LECs):
            C = reduced_solve(LECs)
            return X[self.onshell_elems] @ C

        def estimate_residual(LECs):
            C = reduced_solve(LECs)
            return jnp.linalg.norm(
                (jnp.identity(VG.shape[0]) - jnp.einsum('ijo,o->ij', VG, LECs)) @ (X @ C)
                - jnp.einsum('io,o->i', V, LECs)
            )

        # estimate residuals
        estimated_residuals = jax.vmap(estimate_residual, in_axes=(0,))(LECs_candidates)

        index_max = jnp.argmax(estimated_residuals)
        t_em = emulate_t(LECs_candidates[index_max])
        t_ex = self.solver.coupled_channel_t_flat(LECs_candidates[index_max], cc, k)

        calibration_ratio = jnp.linalg.norm(t_em - t_ex) / estimated_residuals[index_max]

        def estimate_error(LECs):
            return calibration_ratio * estimate_residual(LECs)

        # store emulator
        self.t_cck[cc][k] = jax.jit(emulate_t)
        self.onshell_t_cck[cc][k] = jax.jit(emulate_onshell_t)
        self.error_cck[cc][k] = jax.jit(estimate_error)


    @partial(jax.jit, static_argnames=("self"))
    def t(
        self,
        LECs: jnp.ndarray,
    ):
        """
        this function assumes emulators have been trained for all channels and energies
        """
        
        ncc = self.channels.n_coupled
        nk = self.solver.n_poles
        nq = self.mesh.n_mesh
        
        Tckq = self.emulate_t_ck(LECs)
        Tcckq = self.emulate_t_cck(LECs).reshape(ncc, nk, 2, 2, nq+1)

        return Tckq, Tcckq
        
        
    @partial(jax.jit, static_argnames=("self"))
    def onshell_t(
        self,
        LECs: jnp.ndarray,
    ):
        """
        this function assumes emulators have been trained for all channels and energies
        """
        
        ncc = self.channels.n_coupled
        nk = self.solver.n_poles
        
        Tck = self.emulate_onshell_t_ck(LECs)
        Tcck = self.emulate_onshell_t_cck(LECs)
        Tcck = Tcck.reshape(ncc, nk, 2, 2)

        return Tck, Tcck
        
        
    @partial(jax.jit, static_argnames=("self"))
    def onshell_t_and_err(
        self,
        LECs: jnp.ndarray,
    ):
        
        ncc = self.channels.n_coupled
        nk = self.solver.n_poles
        
        Tck = self.emulate_onshell_t_ck(LECs)
        Tcck = self.emulate_onshell_t_cck(LECs)
        Tcck = Tcck.reshape(ncc, nk, 2, 2)
        
        err_Tck = self.estimate_error_ck(LECs)
        err_Tcck = self.estimate_error_cck(LECs)
        #err_Tcck = err_Tcck.reshape(ncc, nk) # redundant 

        return (Tck, Tcck), (err_Tck, err_Tcck)



    @partial(jax.jit, static_argnames=("self"))
    def qr(self, A):
        Q, _ = jnp.linalg.qr(A)
        return Q
                
    @partial(jax.jit, static_argnames=("self"))
    def svd(self, A):
        U, s, _ = jnp.linalg.svd(A, full_matrices=False)
        return U, s
        
    @partial(jax.jit, static_argnames=("self"))
    def residual_basis(
        self,
        VG: jnp.ndarray,
        V: jnp.ndarray,
        X: jnp.ndarray,
    ):
    
        # apply AX for all operators
        VGX = jnp.einsum('ijo,jb->iob', VG, X)
        
        # construct residual subspace (also test basis in LSPG)
        # inactive columns are at the end
        Y = jnp.concatenate((
            V,
            jnp.reshape(VGX, (VG.shape[0], -1), order='F'),
            ), axis=1
        )
        
        # orthogonalize residual subspace
        return self.qr(Y)



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


    @partial(jax.jit, static_argnames=("self"))
    def petrov_galerkin_basis(
        self,
        VG: jnp.ndarray,
        V: jnp.ndarray,
        X: jnp.ndarray,
    ):
    
        # apply AX for all operators
        VGX = jnp.einsum('ijo,jb->iob', VG, X)
        
        # construct residual subspace (also test basis in LSPG)
        # inactive columns are at the end
        Y = jnp.concatenate((
            V,
            jnp.reshape(VGX, (VG.shape[0], -1), order='F'),
            ), axis=1
        )
        
        # orthogonalize residual subspace
        Y = self.qr(Y)

        return Y



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
        
        
        def nested_Nones(m, n):
            return [[None for _ in range(n)] for _ in range(m)]
            
        nc = self.channels.n_single
        ncc = self.channels.n_coupled
        nk = self.solver.n_poles
    
        #self.single_channel_reduced_solve     = nested_Nones(nc, nk)
        self.single_channel_emulate_t         = nested_Nones(nc, nk)
        self.single_channel_estimate_residual = nested_Nones(nc, nk)
        self.single_channel_calibration_ratio = nested_Nones(nc, nk)
        self.single_channel_basis_size        = nested_Nones(nc, nk)
        
        #self.coupled_channel_reduced_solve     = nested_Nones(ncc, nk)
        self.coupled_channel_emulate_t         = nested_Nones(ncc, nk)
        self.coupled_channel_estimate_residual = nested_Nones(ncc, nk)
        self.coupled_channel_calibration_ratio = nested_Nones(ncc, nk)
        self.coupled_channel_basis_size        = nested_Nones(ncc, nk)
        
        

    #@partial(jax.jit, static_argnames=("self"))
    def qr(self, A):
        Q, _ = jnp.linalg.qr(A)
        return Q
                
    #@partial(jax.jit, static_argnames=("self"))
    def svd(self, A):
        U, s, _ = jnp.linalg.svd(A, full_matrices=False)
        return U, s
        
    #@partial(jax.jit, static_argnames=("self"))
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
        
        Tckq = self.single_channel_emulate_t_all(LECs)
        Tcckq = self.coupled_channel_emulate_t_all(LECs).reshape(ncc, nk, 2, 2, nq+1)
        

        return Tckq, Tcckq

        

    def fit(
        self,
        LECs_candidates,
        rom='g',
        mode='pod',
        n_init=2,
        n_max=25,
        tol=1e-5
    ):
        for k in range(self.solver.n_poles):
            for c in range(self.channels.n_single):
                self.fit_single_channel_pod_galerkin(LECs_candidates, c, k, n_max=n_max, tol=tol)
            for cc in range(self.channels.n_coupled):
                self.fit_coupled_channel_pod_galerkin(LECs_candidates, cc, k, n_max=n_max, tol=tol)
                

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
            
        #self.single_channel_reduced_solve_all = make_grid_funcs(self.single_channel_reduced_solve)
        self.single_channel_emulate_t_all = make_grid_funcs(self.single_channel_emulate_t)
        self.single_channel_estimate_residual_all = make_grid_funcs(self.single_channel_estimate_residual)

        #self.coupled_channel_reduced_solve_all = make_grid_funcs(self.coupled_channel_reduced_solve)
        self.coupled_channel_emulate_t_all = make_grid_funcs(self.coupled_channel_emulate_t)
        self.coupled_channel_estimate_residual_all = make_grid_funcs(self.coupled_channel_estimate_residual)
        

    

    def fit_single_channel_pod_galerkin(
        self,
        LECs_candidates,
        c: int,
        k: int,
        n_max: int = 25,  # maximum number of snapshots to select from candidates
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
            
        def estimate_residual(LECs):
            C = reduced_solve(LECs)
            return jnp.linalg.norm(
                I_residual @ C
                - (VG_residual @ LECs) @ C
                - V_residual @ LECs
            )
            
        # solve all candidates in reduced space, estimate residuals
        C_candidates = jax.vmap(
            reduced_solve,
            in_axes=(0,)
        )(LECs_candidates)

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
        
        # store emulator
        #self.single_channel_reduced_solve[c][k]     = jax.jit(reduced_solve)
        self.single_channel_emulate_t[c][k]         = jax.jit(emulate_t)
        self.single_channel_estimate_residual[c][k] = jax.jit(estimate_residual)
        self.single_channel_calibration_ratio[c][k] = calibration_ratio
        self.single_channel_basis_size[c][k]        = n_basis
    
        

    def fit_coupled_channel_pod_galerkin(
        self,
        LECs_candidates,
        cc: int,
        k: int,
        n_max: int = 25,  # maximum number of snapshots to select from candidates
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
        
        
        def reduced_solve(LECs):
            return jnp.linalg.solve(
                jnp.identity(VG_reduced.shape[0]) - VG_reduced @ LECs,
                V_reduced @ LECs
            )
        
        def emulate_t(LECs):
            C = reduced_solve(LECs)
            return X @ C
            
        def estimate_residual(LECs):
            C = reduced_solve(LECs)
            return jnp.linalg.norm(
                I_residual @ C
                - (VG_residual @ LECs) @ C
                - V_residual @ LECs
            )
            
        # solve all candidates in reduced space, estimate residuals
        C_candidates = jax.vmap(
            reduced_solve,
            in_axes=(0,)
        )(LECs_candidates)

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
        
        # store emulator
        #self.coupled_channel_reduced_solve[cc][k]     = jax.jit(reduced_solve)
        self.coupled_channel_emulate_t[cc][k]         = jax.jit(emulate_t)
        self.coupled_channel_estimate_residual[cc][k] = jax.jit(estimate_residual)
        self.coupled_channel_calibration_ratio[cc][k] = calibration_ratio
        self.coupled_channel_basis_size[cc][k]        = n_basis





'''

    def load(
        self,
        tol: float = 1e-5,
        rom = "g", # or "lspg"
        mode = "greedy", # or "pod"
    ):
        nk = self.solver.n_poles
        nc = self.channels.n_single
        ncc = self.channels.n_coupled
        self.single_emulators = [[None] * nk for _ in range(nc)]
        self.coupled_emulators = [[None] * nk for _ in range(ncc)]
        
        for k in range(nk):
            for c in range(nc):
                self.single_emulators[c][k] = self.load_single_channel(
                    c, k,
                    tol=tol,
                    rom=rom,
                    mode=mode
                )
                
            for cc in range(ncc):
                self.coupled_emulators[cc][k] = self.load_coupled_channel(
                    cc, k,
                    tol=tol,
                    rom=rom,
                    mode=mode
                )


    def fit(
        self,
        LECs_candidates,
        n_init: int = 2,  # number of initial points for greedy algo
        n_max: int = 25,  # maximum number of snapshots to select from candidates
        tol: float = 1e-5,
        rom = "g", # or "lspg"
        mode = "greedy", # or "pod"
    ):
        nk = self.solver.n_poles
        nc = self.channels.n_single
        ncc = self.channels.n_coupled
        self.single_emulators = [[None] * nk for _ in range(nc)]
        self.coupled_emulators = [[None] * nk for _ in range(ncc)]
        
        for k in range(nk):
            for c in range(nc):
                self.single_emulators[c][k] = self.fit_single_channel(
                    LECs_candidates, c, k,
                    n_init=n_init,
                    n_max=n_max,
                    tol=tol,
                    rom=rom,
                    mode=mode
                )
                
            for cc in range(ncc):
                self.coupled_emulators[cc][k] = self.fit_coupled_channel(
                    LECs_candidates, cc, k,
                    n_init=n_init,
                    n_max=n_max,
                    tol=tol,
                    rom=rom,
                    mode=mode
                )
    
        
    def fit_single_channel(
        self,
        LECs_candidates,
        c: int,     # channel index
        k: int,     # pole index
        n_init: int = 2,  # number of initial points for greedy algo
        n_max: int = 25,  # maximum number of snapshots to select from candidates
        tol: float = 1e-5,
        rom = "g", # or "lspg"
        mode = "greedy", # or "pod"
    ):
    

        n_dim = self.mesh.n_mesh+1
        setup = self.solver.setup_single_channel
        solve = self.solver.solve_single_channel
            
            
        # choose appropriate functions for galerkin or petrov-galerkin rom
        if rom == "g":
            select_test_basis = lambda X, Y: X
            reduced_solve = self.batch_solve_galerkin
            project_residual = lambda X, Y, VG, V, I_reduced, VG_reduced, V_reduced: self.project(X, Y, VG, V)
        elif rom == "lspg":
            select_test_basis = lambda X, Y: Y
            reduced_solve = self.batch_solve_petrov_galerkin
            project_residual = lambda X, Y, VG, V, I_reduced, VG_reduced, V_reduced: (I_reduced, VG_reduced, V_reduced)
        else:
            print("Unknown reduced-order model {rom}.")
            

        # load trained emulator if it exists
        emulator = self.load_emulator(
            channel, Elab, n_init, n_max, tol, rom, mode
        )
        
        if emulator is not None:
            return emulator # model, errors, times, meta
            

        # empty containers for emulator and training analysis
        model = {
            'reduced I': None,
            'reduced VG': None,
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
            'reduced solve': [],
            'full solve': [],
            'error': [],
        }
        
        meta = {
            'Elab': Elab,
            'channel': channel,
            'n_init': n_init,
            'n_max': n_max,
            'tol': tol,
            'rom': rom,
            'mode': mode,
        }
        
        # setup the linear system regardless of mode
        with timed_section(times['setup']):
            if linear_system is None:
                VG, V = setup(channel, Elab)
            else:
                VG, V = linear_system
                
        # execute training
        if mode == "pod":
        
            # collect snapshots of exact solutions
            with timed_section(times['setup']):
                LECs_train = LECs_candidates[:n_max]
                T = jax.vmap(solve, in_axes=(0,None,None))(LECs_train, VG, V).T
                
            # orthogonalize basis, truncate, and project
            with timed_section(times['reduce']):
                X, singular_vals = self.svd(T)
                index = int(jnp.argmax(singular_vals/singular_vals[0] <= tol))
                n_basis = index + 1 if index > 0 else n_max
                X = X[:,:n_basis]  # rhs
                Y = self.petrov_galerkin_basis(VG, V, X)
                I_reduced, VG_reduced, V_reduced = self.project(X, select_test_basis(X, Y), VG, V)
                
            with timed_section(times['reduced solve']):
                C = reduced_solve(LECs_candidates, I_reduced, VG_reduced, V_reduced)
                
            with timed_section(times['full solve']):
                # this is just to directly compare with reduced solve time
                _ = jax.vmap(solve, in_axes=(0,None,None))(LECs_candidates, VG, V)
                
            with timed_section(times['error']):
            
                I_residual, VG_residual, V_residual = project_residual(X, Y, VG, V, I_reduced, VG_reduced, V_reduced)

                errors_est = self.batch_estimate_error(LECs_candidates, C, I_residual, VG_residual, V_residual)
                index_max_error = jnp.argmax(errors_est)
                
                # convert worst emulated candidate to full space
                #t_em = jnp.einsum('ib,b->i', X, C[index_max_error])
                t_em = self.reconstruct(X, C[index_max_error])
                
                # compute exact solution at worst point
                t_ex = solve(LECs_candidates[index_max_error], VG, V)
                
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
        
            print("\n* in greedy")
        
            # collect n_init snapshots, leaving room for n_max snapshots
            with timed_section(times['setup']):
                T = jnp.zeros((n_dim, n_max), dtype=jnp.complex128)
                T = T.at[:, :n_init].set(
                    jax.vmap(solve, in_axes=(0,None,None))(LECs_candidates[:n_init], VG, V).T
                )
                LECs_train = jnp.zeros((n_max, self.potential.n_operators), dtype=jnp.complex128)
                LECs_train = LECs_train.at[:n_init].set( LECs_candidates[:n_init] )
                
            for i in range(n_init, n_max):
            
                print("\n* i = ", i)
            
                with timed_section(times['reduce']):
                    X = self.qr(T)
                    X = X.at[:,i:].set(0.0)
                    Y = self.petrov_galerkin_basis(VG, V, X)
                    I_reduced, VG_reduced, V_reduced = self.project(X, select_test_basis(X, Y), VG, V)
                    
                

                with timed_section(times['reduced solve']):
                    C = reduced_solve(LECs_candidates, I_reduced, VG_reduced, V_reduced)
    
                with timed_section(times['full solve']):
                    # this is just to compare with emulation time
                    _ = jax.vmap(solve, in_axes=(0,None,None))(LECs_candidates, VG, V)
                    
                print("speedup = ", times['full solve'][-1] / times['reduced solve'][-1])
                    
                with timed_section(times['error']):
        
                    I_residual, VG_residual, V_residual = project_residual(X, Y, VG, V, I_reduced, VG_reduced, V_reduced)
                    errors_est = self.batch_estimate_error(LECs_candidates, C, I_residual, VG_residual, V_residual)
                    index_max_error = jnp.argmax(errors_est)
                    errors_est.block_until_ready()
                    
                    # convert worst emulated candidate to full space
                    #t_em = jnp.einsum('ib,b->i', X, C[index_max_error])
                    t_em = self.reconstruct(X, C[index_max_error])
                    #t_em.block_until_ready()

                    # compute exact solution at worst point
                    t_ex = solve(LECs_candidates[index_max_error], VG, V)
                    #t_ex.block_until_ready()
                    
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
                    
                    print("error = ", jnp.max(errors_cal))

                
                    
                if jnp.max(errors_cal) < tol:
                    n_basis = i
                    break
                    
        
        model['reduced VG'] = VG_reduced
        model['reduced V'] = V_reduced
        model['reduced I'] = I_reduced
        model['trial basis'] = X
        model['test basis'] = Y
        model['snapshots'] = T
        model['LECs'] = LECs_train
        #model['reduced solve'] = reduced_solve
        
        for dict in [model, errors, times]:
            for key, val in dict.items():
                if isinstance(val, list):
                    dict[key] = jnp.array(val)
                #print(key, dict[key])
                
                
        self.save_emulator(channel, Elab, model, errors, times, meta, rom, mode, tol)
                
        return model, errors, times, meta
        


        
    def total_cross_section(
        self,
        models,
        metas,
        LECs
    ):
        sigma_tot = 0.0
    
        for model, meta in zip(models, metas):
        
            sigma_tot += self.scattering_params(model, meta, LECs)[2]
            
        return sigma_tot.squeeze()
            
            

    def scattering_params(
        self,
        model,
        meta,
        LECs
    ):
        return self.solver.scattering_params(
            channel=meta["channel"],
            Elab=meta["Elab"],
            Tq=self.t(model, meta, LECs)
        )
        

    def t(
        self,
        model,
        meta,
        LECs: jnp.ndarray = None
    ) -> jnp.ndarray:
    
        if meta["rom"] == "g":
            reduced_solve = self.solve_galerkin
        else:
            reduced_solve = self.solve_petrov_galerkin
            
        C = reduced_solve(LECs, model["reduced I"], model["reduced VG"], model["reduced V"])
        T = self.reconstruct(model["trial basis"], C.squeeze())
        
        if meta["channel"] in self.channels.coupled:
            T = jnp.reshape(T, (2,2,-1))
        
        return T
        
        
    @partial(jax.jit, static_argnames=("self"))
    def reconstruct(
        self,
        X,
        C
    ):
        return jnp.dot(X, C)
    

    @partial(jax.jit, static_argnames=("self"))
    def solve_galerkin(
        self,
        LECs,
        I_reduced,
        VG_reduced,
        V_reduced,
    ):
        
        return jnp.linalg.solve(
            jnp.identity(VG_reduced.shape[0]) - jnp.einsum('abo,o->ab', VG_reduced, LECs),
            jnp.einsum('bo,o->b', V_reduced, LECs)
        )
        
    @partial(jax.jit, static_argnames=("self"))
    def batch_solve_galerkin(
        self,
        LECs_batch,
        I_reduced,
        VG_reduced,
        V_reduced
    ):
        LECs_batch = jnp.reshape(LECs_batch, (-1, self.potential.n_operators))
        return jax.vmap(self.solve_galerkin, in_axes=(0,None,None,None))(
            LECs_batch, I_reduced, VG_reduced, V_reduced
        )
        
    @partial(jax.jit, static_argnames=("self"))
    def solve_petrov_galerkin(
        self,
        LECs,
        I_reduced,
        VG_reduced,
        V_reduced,
    ):
        #i = i or VG_reduced.shape[0]
        #j = self.potential.n_operators * (i + 1)
        
        return jnp.linalg.lstsq(
            I_reduced - jnp.einsum('abo,o->ab', VG_reduced, LECs),
            jnp.einsum('bo,o->b', V_reduced, LECs)
        )[0]
        
    @partial(jax.jit, static_argnames=("self"))
    def batch_solve_petrov_galerkin(
        self,
        LECs_batch,
        I_reduced,
        VG_reduced,
        V_reduced
    ):
        LECs_batch = jnp.reshape(LECs_batch, (-1, self.potential.n_operators))
        return jax.vmap(self.solve_petrov_galerkin, in_axes=(0,None,None,None))(
            LECs_batch, I_reduced, VG_reduced, V_reduced
        )
        
        
    def load_emulator(
        self,
        channel: str = "1S0",
        Elab: float = 10.0,
        n_init: int = 2,  # number of initial points for greedy algo
        n_max: int = 25,  # maximum number of snapshots to select from candidates
        tol: float = 1e-5,
        rom = "g", # or "lspg"
        mode = "greedy", # or "pod"
    ):
        print("")
        
        def load_arrays(filename):
        
            if glob.glob(filename):
                print(f"Loading file: {filename}")
                t = time.perf_counter()
                arrays = jnp.load(filename)
                t = time.perf_counter() - t
                print(f"Done in {t:.5f} sec.")
            else:
                arrays = None
                print(f"File not found: {filename}")
                
            return arrays
            
        def load_dict(filename):
        
            if glob.glob(filename):
                print(f"Loading file: {filename}")
                t = time.perf_counter()
                with open(filename) as f:
                    dict = json.load(f)
                t = time.perf_counter() - t
                print(f"Done in {t:.5f} sec.")
            else:
                dict = None
                print(f"File not found: {filename}")
                
            return dict
            
            
        filebase = f"saved_emulators/{self.potential.name}_Nq{self.mesh.n_mesh}_{rom}_{mode}_{tol:.1e}_{channel}_{Elab:.3f}MeV"
        
        meta = load_dict(filebase + "_meta.json")
        model = load_arrays(filebase + "_model.npz")
        errors = load_arrays(filebase + "_errors.npz")
        times = load_arrays(filebase + "_times.npz")
        
        if meta is not None and model is not None and errors is not None and times is not None:
            return model, errors, times, meta
        else:
            return None
        
        
    def save_emulator(
        self,
        channel: str,
        Elab: float,
        model: dict,
        errors: dict,
        times: dict,
        meta: dict,
        rom: str,
        mode: str,
        tol: float,
    ):

        filebase = f"saved_emulators/{self.potential.name}_Nq{self.mesh.n_mesh}_{rom}_{mode}_{tol:.1e}_{channel}_{Elab:.3f}MeV"

        def save_arrays(filename, arrays):
            print(f"Writing file: {filename}")
            t = time.perf_counter()
            jnp.savez(filename, **arrays)
            t = time.perf_counter() - t
            print(f"Done in {t:.5f} sec.")

        def save_dict(filename, dct):
            print(f"Writing file: {filename}")
            t = time.perf_counter()
            with open(filename, "w") as f:
                json.dump(dct, f, indent=2)
            t = time.perf_counter() - t
            print(f"Done in {t:.5f} sec.")

        # actually save
        save_dict(filebase + "_meta.json", meta)
        save_arrays(filebase + "_model.npz", model)
        save_arrays(filebase + "_errors.npz", errors)
        save_arrays(filebase + "_times.npz", times)
 
        
    def fit(
        self,
        LECs_candidates,
        channel: str = "1S0",
        Elab: float = 10.0,
        n_init: int = 2,  # number of initial points for greedy algo
        n_max: int = 25,  # maximum number of snapshots to select from candidates
        tol: float = 1e-5,
        rom = "g", # or "lspg"
        mode = "greedy", # or "pod"
        linear_system: Tuple[jnp.ndarray, jnp.ndarray] = None, # remove?
    ):
        
        # choose the appropriate functions for single/coupled channels
        if channel in self.channels.single.keys():
            n_dim = self.mesh.n_mesh+1
            setup = self.solver.setup_single_channel
            solve = self.solver.solve_single_channel
            
        elif channel in self.channels.coupled.keys():
            n_dim = 4*self.mesh.n_mesh+4
            setup = self.solver.setup_coupled_channel_flat
            solve = self.solver.solve_coupled_channel_flat
        
        else:
            print("Channel {channel} not found in list.")
            
        assert n_max <= n_dim, "Maximum basis size is too large."
            
        # choose appropriate functions for galerkin or petrov-galerkin rom
        if rom == "g":
            select_test_basis = lambda X, Y: X
            reduced_solve = self.batch_solve_galerkin
            project_residual = lambda X, Y, VG, V, I_reduced, VG_reduced, V_reduced: self.project(X, Y, VG, V)
        elif rom == "lspg":
            select_test_basis = lambda X, Y: Y
            reduced_solve = self.batch_solve_petrov_galerkin
            project_residual = lambda X, Y, VG, V, I_reduced, VG_reduced, V_reduced: (I_reduced, VG_reduced, V_reduced)
        else:
            print("Unknown reduced-order model {rom}.")
            

        # load trained emulator if it exists
        emulator = self.load_emulator(
            channel, Elab, n_init, n_max, tol, rom, mode
        )
        
        if emulator is not None:
            return emulator # model, errors, times, meta
            

        # empty containers for emulator and training analysis
        model = {
            'reduced I': None,
            'reduced VG': None,
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
            'reduced solve': [],
            'full solve': [],
            'error': [],
        }
        
        meta = {
            'Elab': Elab,
            'channel': channel,
            'n_init': n_init,
            'n_max': n_max,
            'tol': tol,
            'rom': rom,
            'mode': mode,
        }
        
        # setup the linear system regardless of mode
        with timed_section(times['setup']):
            if linear_system is None:
                VG, V = setup(channel, Elab)
            else:
                VG, V = linear_system
                
        # execute training
        if mode == "pod":
        
            # collect snapshots of exact solutions
            with timed_section(times['setup']):
                LECs_train = LECs_candidates[:n_max]
                T = jax.vmap(solve, in_axes=(0,None,None))(LECs_train, VG, V).T
                
            # orthogonalize basis, truncate, and project
            with timed_section(times['reduce']):
                X, singular_vals = self.svd(T)
                index = int(jnp.argmax(singular_vals/singular_vals[0] <= tol))
                n_basis = index + 1 if index > 0 else n_max
                X = X[:,:n_basis]  # rhs
                Y = self.petrov_galerkin_basis(VG, V, X)
                I_reduced, VG_reduced, V_reduced = self.project(X, select_test_basis(X, Y), VG, V)
                
            with timed_section(times['reduced solve']):
                C = reduced_solve(LECs_candidates, I_reduced, VG_reduced, V_reduced)
                
            with timed_section(times['full solve']):
                # this is just to directly compare with reduced solve time
                _ = jax.vmap(solve, in_axes=(0,None,None))(LECs_candidates, VG, V)
                
            with timed_section(times['error']):
            
                I_residual, VG_residual, V_residual = project_residual(X, Y, VG, V, I_reduced, VG_reduced, V_reduced)

                errors_est = self.batch_estimate_error(LECs_candidates, C, I_residual, VG_residual, V_residual)
                index_max_error = jnp.argmax(errors_est)
                
                # convert worst emulated candidate to full space
                #t_em = jnp.einsum('ib,b->i', X, C[index_max_error])
                t_em = self.reconstruct(X, C[index_max_error])
                
                # compute exact solution at worst point
                t_ex = solve(LECs_candidates[index_max_error], VG, V)
                
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
        
            print("\n* in greedy")
        
            # collect n_init snapshots, leaving room for n_max snapshots
            with timed_section(times['setup']):
                T = jnp.zeros((n_dim, n_max), dtype=jnp.complex128)
                T = T.at[:, :n_init].set(
                    jax.vmap(solve, in_axes=(0,None,None))(LECs_candidates[:n_init], VG, V).T
                )
                LECs_train = jnp.zeros((n_max, self.potential.n_operators), dtype=jnp.complex128)
                LECs_train = LECs_train.at[:n_init].set( LECs_candidates[:n_init] )
                
            for i in range(n_init, n_max):
            
                print("\n* i = ", i)
            
                with timed_section(times['reduce']):
                    X = self.qr(T)
                    X = X.at[:,i:].set(0.0)
                    Y = self.petrov_galerkin_basis(VG, V, X)
                    I_reduced, VG_reduced, V_reduced = self.project(X, select_test_basis(X, Y), VG, V)
                    
                

                with timed_section(times['reduced solve']):
                    C = reduced_solve(LECs_candidates, I_reduced, VG_reduced, V_reduced)
    
                with timed_section(times['full solve']):
                    # this is just to compare with emulation time
                    _ = jax.vmap(solve, in_axes=(0,None,None))(LECs_candidates, VG, V)
                    
                print("speedup = ", times['full solve'][-1] / times['reduced solve'][-1])
                    
                with timed_section(times['error']):
        
                    I_residual, VG_residual, V_residual = project_residual(X, Y, VG, V, I_reduced, VG_reduced, V_reduced)
                    errors_est = self.batch_estimate_error(LECs_candidates, C, I_residual, VG_residual, V_residual)
                    index_max_error = jnp.argmax(errors_est)
                    errors_est.block_until_ready()
                    
                    # convert worst emulated candidate to full space
                    #t_em = jnp.einsum('ib,b->i', X, C[index_max_error])
                    t_em = self.reconstruct(X, C[index_max_error])
                    #t_em.block_until_ready()

                    # compute exact solution at worst point
                    t_ex = solve(LECs_candidates[index_max_error], VG, V)
                    #t_ex.block_until_ready()
                    
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
                    
                    print("error = ", jnp.max(errors_cal))

                
                    
                if jnp.max(errors_cal) < tol:
                    n_basis = i
                    break
                    
        
        model['reduced VG'] = VG_reduced
        model['reduced V'] = V_reduced
        model['reduced I'] = I_reduced
        model['trial basis'] = X
        model['test basis'] = Y
        model['snapshots'] = T
        model['LECs'] = LECs_train
        #model['reduced solve'] = reduced_solve
        
        for dict in [model, errors, times]:
            for key, val in dict.items():
                if isinstance(val, list):
                    dict[key] = jnp.array(val)
                #print(key, dict[key])
                
                
        self.save_emulator(channel, Elab, model, errors, times, meta, rom, mode, tol)
                
        return model, errors, times, meta
        
    def project(
        self,
        trial_basis: jnp.ndarray,
        test_basis: jnp.ndarray,
        VG: jnp.ndarray,
        V: jnp.ndarray,
    ):
    
        I_reduced = jnp.einsum('ia,ib->ab', jnp.conjugate(test_basis), trial_basis)
        VG_reduced = jnp.einsum('ia,ijo,jb->abo', jnp.conjugate(test_basis), VG, trial_basis)
        V_reduced  = jnp.einsum('ia,io->ao', jnp.conjugate(test_basis), V)

        return I_reduced, VG_reduced, V_reduced
        
        
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
        
        
    @partial(jax.jit, static_argnames=("self"))
    def estimate_error(
        self,
        LECs: jnp.ndarray,
        C: jnp.ndarray,
        I_residual: jnp.ndarray,
        VG_residual: jnp.ndarray,
        V_residual: jnp.ndarray,
    ):
        A_residual = I_residual - jnp.einsum('abo,o->ab', VG_residual, LECs)
        V_residual = jnp.einsum('ao,o->a', V_residual, LECs)

        return jnp.linalg.norm(
            jnp.einsum('ab,b->a', A_residual, C) - V_residual
        )
        

    @partial(jax.jit, static_argnames=("self"))
    def batch_estimate_error(
        self,
        LECs_batch: jnp.ndarray,
        C_batch: jnp.ndarray,
        I_residual: jnp.ndarray,
        VG_residual: jnp.ndarray,
        V_residual: jnp.ndarray,
    ):

        #LECs_batch = (s,o)
        #C_batch = (s,b)
        #I_residual = (a,b)
        #VG_residual = (a,b,o)
        #V_residual = (a,o)
        
        #s >> a > b > o
   
        
        IC = jnp.einsum('ab,sb->sa', I_residual, C_batch)
        VGC = jnp.einsum('abo,sb->sao', VG_residual, C_batch)
        VGC = jnp.einsum('sao,so->sa', VGC, LECs_batch)
        V = jnp.einsum('ao,so->sa', V_residual, LECs_batch)
        
        return jnp.linalg.norm(IC-VGC-V, axis=1)
    
        
        


'''

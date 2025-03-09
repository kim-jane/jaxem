import jax.numpy as jnp
from jax import config
import jax
config.update("jax_enable_x64", True)
from jax import random, grad, jit, vmap
from scipy.interpolate import griddata, RBFInterpolator
from functools import partial
import matplotlib.pyplot as plt
from .Tools import *
from .Potential import Potential
from .Map import Map
import matplotlib.pyplot as plt
import time
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from collections import defaultdict


import logging
import time

'''
indices:
    o = operators
    b = basis
    s = samples
    c = single channels
    cc = coupled channels
    k = energies / poles
    q = momenta
'''

class TMatrix:

    def __init__(self, config):
        
        self.hbarc = config.hbarc
        self.m = config.m / self.hbarc # fm^-1
        self.output = config.output
        self.factor = config.factor
        self.config = config

        self.pot = Potential(config)
        self.map = Map(config)
        self.chan = self.pot.chan
        
        # compute all poles k
        self.Elab = jnp.sort(config.Elab) # MeV
        self.k = jnp.array([jnp.sqrt(0.5 * self.m * Elab / self.hbarc) for Elab in self.Elab])
        self.Nk = self.k.shape[0]
        
        # make momentum grid q
        self.Nq = config.Nq
        self.q, self.wq = self.map.grid() # fm^-1
        
        # setup logger for time stamps
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(f"output/{self.output}.txt", mode="a")
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)  # Set log level as needed
        self.logger.info(f"logger initialized and writing to file: output/{self.output}.txt")
        
        # setup TMatrix object
        self.log_block_start("SETUP")
        self.precalc_single_pot()
        self.precalc_coupled_pot()
        self.precalc_prop()
        self.log_block_end()
        
        
    @partial(jax.jit, static_argnames=('self'))
    def system_single(self, LECs, c_index, k_index):
        Vqq = jnp.einsum('o,oij->ij', LECs, self.Vockqq[:,c_index,k_index,:,:])
        iq = jnp.arange(self.Nq+1)
        Aqq = -jnp.einsum('ij,j->ij', Vqq, self.Gkq[k_index,:])
        Aqq = Aqq.at[iq,iq].add(1.0)
        return (Aqq, Vqq[:,0])
        
    @partial(jax.jit, static_argnames=('self'))
    def system_coupled(self, LECs, cc_index, k_index):
        Vqq = jnp.einsum('o,oij->ij', LECs, self.Vocckqq[:,cc_index,k_index,:,:])
        iq = jnp.arange(2*self.Nq+2)
        Aqq = -jnp.einsum('ij,j->ij', Vqq, jnp.tile(self.Gkq[k_index], (2,)))
        Aqq = Aqq.at[iq,iq].add(1.0)
        return (Aqq, Vqq[:,[0, self.Nq + 1]])
        
    
    @partial(jax.jit, static_argnames=('self'))
    def solve_single(self, LECs, c_index, k_index):
        Aqq, Vq = self.system_single(LECs, c_index, k_index)
        Tq = jnp.linalg.solve(Aqq, Vq)
        return Tq
        
    @partial(jax.jit, static_argnames=('self'))
    def solve_coupled(self, LECs, cc_index, k_index):
        Aqq, Vq = self.system_coupled(LECs, cc_index, k_index)
        Tq = jnp.linalg.solve(Aqq, Vq)
        Tq = jnp.reshape(Tq, (2, self.Nq+1, 2))
        Tq = jnp.transpose(Tq, (0,2,1))
        return Tq
        
    def solve(self, LECs):
    
        Tsckq = [[] for c in range(self.chan.Nsingle)]
        
        for c in range(self.chan.Nsingle):
            for k in range(self.Nk):
                Tsckq[c].append(self.solve_single(LECs, c, k))
                
                
        Tscckq = [[] for cc in range(self.chan.Ncoupled)]
        
        for cc in range(self.chan.Ncoupled):
            for k in range(self.Nk):
                Tscckq[cc].append(self.solve_coupled(LECs, cc, k))
                
        Tsckq = jnp.array(Tsckq)[None,...]
        Tscckq = jnp.array(Tscckq)[None,...]
    
        
        print("Tsckq = ", Tsckq.shape)
        print("Tscckq = ", Tscckq.shape)
        return Tsckq, Tscckq
        
        
    def GROM_single(self, LECs_train, c_index, k_index, truncate=False):
    
        # solve high fidelity model
        Tsq = jax.vmap(self.solve_single, in_axes=(0,None,None))(LECs_train, c_index, k_index)
        
        # orthogonalize
        Uqs, Ss, _ = jnp.linalg.svd(Tsq.T, full_matrices=False)
        
        
        # truncate basis
        if truncate:
            index = int(jnp.argmax(Ss/Ss[0] <= self.config.svd_tol))
            Nbasis = index + 1 if index > 0 else LECs_train.shape[0]
        else:
            Nbasis = LECs_train.shape[0]
            
        Xqb = Uqs[:,:Nbasis]
        
        # project
        Voqq = self.Vockqq[:,c_index,k_index,:,:]
        VGobb = jnp.einsum('ia,oij,j,jb->oab', jnp.conjugate(Xqb), Voqq, self.Gkq[k_index,:], Xqb)
        Vob = jnp.einsum('ib,oi->ob', jnp.conjugate(Xqb), Voqq[:,:,0])
        
        return (VGobb, Vob, Xqb)
        
        
    def GROM_coupled(self, LECs_train, cc_index, k_index, truncate=False):
    
        # solve high fidelity model
        Tsq = jax.vmap(self.solve_coupled, in_axes=(0,None,None))(LECs_train, cc_index, k_index)
        Tsq = jnp.concatenate((Tsq[:,0,0,:], Tsq[:,1,0,:], Tsq[:,0,1,:], Tsq[:,1,1,:]), axis=1)
        
        # orthogonalize
        Uqs, Ss, _ = jnp.linalg.svd(Tsq.T, full_matrices=False)
        
        # truncate basis
        if truncate:
            index = int(jnp.argmax(Ss/Ss[0] <= self.config.svd_tol))
            Nbasis = index + 1 if index > 0 else LECs_train.shape[0]
        else:
            Nbasis = LECs_train.shape[0]
            
        Xqb = Uqs[:,:Nbasis]
        
        def make_block_diag(VGqq):
            return jax.scipy.linalg.block_diag(VGqq, VGqq)
        
        # project
        Voqq = self.Vocckqq[:,cc_index,k_index,:,:]
        VGoqq = jnp.einsum('oij,j->oij', Voqq, jnp.tile(self.Gkq[k_index,:], (2,)))
        VGoqq = jax.vmap(make_block_diag, in_axes=(0))(VGoqq)
        VGobb = jnp.einsum('ia,oij,jb->oab', jnp.conjugate(Xqb), VGoqq, Xqb)
        
        Voq = jnp.concatenate((Voqq[:,:,0], Voqq[:,:,self.Nq+1]), axis=1)
        Vob = jnp.einsum('ib,oi->ob', jnp.conjugate(Xqb), Voq)
        
        return (VGobb, Vob, Xqb)
        
        
        
    #@partial(jax.jit, static_argnames=('self'))
    def emulate_single(self, LECs, emulator):
    
        VGobb, Vob, Xqb = emulator
        VGbb = jnp.einsum('o,oab->ab', LECs, VGobb)
        Vb = jnp.einsum('o,ob->b', LECs, Vob)
        ib = jnp.arange(Xqb.shape[1])
        Abb = (-VGbb).at[ib,ib].add(1.0)
        Cb = jnp.linalg.solve(Abb, Vb)
        Tq = jnp.einsum('b,ib->i', Cb, Xqb)
        return Tq
        
        
    #@partial(jax.jit, static_argnames=('self'))
    def emulate_coupled(self, LECs, emulator):
    
        VGobb, Vob, Xqb = emulator
        VGbb = jnp.einsum('o,oab->ab', LECs, VGobb)
        Vb = jnp.einsum('o,ob->b', LECs, Vob)
        ib = jnp.arange(Xqb.shape[1])
        Abb = (-VGbb).at[ib,ib].add(1.0)
        Cb = jnp.linalg.solve(Abb, Vb)
        Tq = jnp.einsum('b,ib->i', Cb, Xqb)
        Tq = jnp.reshape(Tq, (2, 2, -1))
        Tq = jnp.transpose(Tq, (1,0,2))
        return Tq
        
    
    def estimate_error_single(self, LECs, c_index, k_index, Tq_emulated):
        Aqq, Vq = self.system_single(LECs, c_index, k_index)
        return jnp.linalg.norm(jnp.dot(Aqq, Tq_emulated) - Vq)
        
        
        
    def estimate_error_coupled(self, LECs, cc_index, k_index, Tq_emulated):
        Aqq, Vq = self.system_coupled(LECs, cc_index, k_index)
        Tq_emulated = jnp.transpose(Tq_emulated, (0,2,1))
        Tq_emulated = jnp.reshape(Tq_emulated, (2*(self.Nq+1), 2))
        
        return jnp.linalg.norm(jnp.dot(Aqq, Tq_emulated) - Vq)
                    
                    
    def train_POD_GROM(self):
    
        key = jax.random.PRNGKey(self.config.seed)
    
        # sample training set
        key, key_in = jax.random.split(key)
        LECs_train = sample_LECs(key_in,
                                 self.config.Ntrain,
                                 self.pot.LECs,
                                 self.config.scale_min,
                                 self.config.scale_max,
                                 static_indices=self.config.static_LECs)
                                 
        # sample testing set (add best fit LECs)
        key, key_in = jax.random.split(key)
        LECs_test = sample_LECs(key_in,
                                self.config.Ntest,
                                self.pot.LECs,
                                self.config.scale_min,
                                self.config.scale_max,
                                static_indices=self.config.static_LECs)
        LECs_test = jnp.concatenate((self.pot.LECs[None,:], LECs_test))
        
        data = self.empty_training_data()

        
        if self.pot.compute_single:
        
            for c, chan_label in enumerate(self.chan.single_spect_not):
                for k in range(self.Nk):
                    
                    emulator = self.GROM_single(LECs_train, c, k, truncate=True)
                    
                    Tsq_em = jax.vmap(self.emulate_single, in_axes=(0,None))(LECs_test, emulator)
                    
                    errors_est = jax.vmap(self.estimate_error_single, in_axes=(0,None,None,0))(LECs_test, c, k, Tsq_em)
                    
                    # calibrate error
                    s = jnp.argmax(errors_est)
                    Tq_ex = self.solve_single(LECs_test[s], c, k)
                    error_ex = jnp.linalg.norm(Tq_ex - Tsq_em[s])
                    errors_cal = errors_est * error_ex / errors_est[s]
                    max_cal_error = jnp.max(errors_cal)
                    
                    print(chan_label, k, emulator[2].shape[1], max_cal_error)
                    
                    # store
                    data[chan_label][k]['emulator'] = emulator
                    data[chan_label][k]['Tq at best LECs'].append(Tsq_em[0])
                    data[chan_label][k]['max cal error'].append(max_cal_error)
                    
                
       
        if self.pot.compute_coupled:
        
            for cc, chan_label in enumerate(self.chan.coupled_spect_not):
                for k in range(self.Nk):
                    
                    emulator = self.GROM_coupled(LECs_train, cc, k, truncate=True)
                    
                    Tsq_em = jax.vmap(self.emulate_coupled, in_axes=(0,None))(LECs_test, emulator)
                                        
                    errors_est = jax.vmap(self.estimate_error_coupled, in_axes=(0,None,None,0))(LECs_test, cc, k, Tsq_em)
                    
                    # calibrate error
                    s = jnp.argmax(errors_est)
                    Tq_ex = self.solve_coupled(LECs_test[s], cc, k)
                    error_ex = jnp.linalg.norm(Tq_ex - Tsq_em[s])
                    errors_cal = errors_est * error_ex / errors_est[s]
                    max_cal_error = jnp.max(errors_cal)
                    
                    print(chan_label, k, emulator[2].shape[1], max_cal_error)
                    
                    # store
                    data[chan_label][k]['emulator'] = emulator
                    data[chan_label][k]['Tq at best LECs'].append(Tsq_em[0])
                    data[chan_label][k]['max cal error'].append(max_cal_error)

        return data
        

    def train_greedy_GROM(self):
    
        key = jax.random.PRNGKey(self.config.seed)
    
        # sample candidate points
        key, key_in = jax.random.split(key)
        LECs_cand = sample_LECs(key_in,
                                self.config.Ncand,
                                self.pot.LECs,
                                self.config.scale_min,
                                self.config.scale_max,
                                static_indices=self.config.static_LECs)
                                
                                 
        # testing set is the same as candidate points plus best fit LECs
        LECs_test = jnp.concatenate((self.pot.LECs[None,:], LECs_cand))
                
        # place holders
        data = self.empty_training_data()

        if self.pot.compute_single:
        
            for c, chan_label in enumerate(self.chan.single_spect_not):
                for k in range(self.Nk):
                
                    # select initial subset
                    LECs_train = LECs_cand[:self.config.Ninit]
                    
                    for i in range(self.config.Nmax):
                    
                        
                        emulator = self.GROM_single(LECs_train, c, k)
                        
                        Tsq_em = jax.vmap(self.emulate_single, in_axes=(0,None))(LECs_test, emulator)
                        
                        errors_est = jax.vmap(self.estimate_error_single, in_axes=(0,None,None,0))(LECs_test, c, k, Tsq_em)
                        
                        # calibrate error for all points except best fit one
                        s = jnp.argmax(errors_est[1:])+1
                        Tq_ex = self.solve_single(LECs_test[s], c, k)
                        error_ex = jnp.linalg.norm(Tq_ex - Tsq_em[s])
                        errors_cal = errors_est * error_ex / errors_est[s]
                        max_cal_error = jnp.max(errors_cal)
                        
                        # store
                        data[chan_label][k]['Tq at best LECs'].append(Tsq_em[0])
                        data[chan_label][k]['max cal error'].append(max_cal_error)
                        
                        
                        
                        # check
                        if max_cal_error < self.config.err_tol:
                        
                            data[chan_label][k]['emulator'] = emulator
                            print(chan_label, k, i, max_cal_error)
                            break
                            
                        else:
                            
                            # add training point
                            LECs_train = jnp.concatenate((LECs_train, LECs_test[s][None,:]))
                    
                
       
        if self.pot.compute_coupled:
        
            for cc, chan_label in enumerate(self.chan.coupled_spect_not):
                for k in range(self.Nk):
                
                    # select initial subset
                    LECs_train = LECs_cand[:self.config.Ninit]
                    
                    for i in range(self.config.Nmax):
                        
                        emulator = self.GROM_coupled(LECs_train, cc, k)
                        
                        Tsq_em = jax.vmap(self.emulate_coupled, in_axes=(0,None))(LECs_test, emulator)
                        
                        errors_est = jax.vmap(self.estimate_error_coupled, in_axes=(0,None,None,0))(LECs_test, cc, k, Tsq_em)
                        
                        # calibrate error
                        s = jnp.argmax(errors_est[1:])+1
                        Tq_ex = self.solve_coupled(LECs_test[s], cc, k)
                        error_ex = jnp.linalg.norm(Tq_ex - Tsq_em[s])
                        errors_cal = errors_est * error_ex / errors_est[s]
                        max_cal_error = jnp.max(errors_cal)
                        
                        # store
                        data[chan_label][k]['Tq at best LECs'].append(Tsq_em[0])
                        data[chan_label][k]['max cal error'].append(max_cal_error)
                        
                        
                        
                        # check
                        if max_cal_error < self.config.err_tol:
                        
                            data[chan_label][k]['emulator'] = emulator
                            print(chan_label, k, i, max_cal_error)
                            break
                            
                        else:
                            
                            # add training point
                            LECs_train = jnp.concatenate((LECs_train, LECs_test[s][None,:]))

        return data
        
        
    def observables_single(self, T_onshell, c_index, k_index):
    
        S = 1 - 1j * jnp.pi * self.factor * self.m * self.k[k_index] * T_onshell
        delta = 0.5 * jnp.arctan2(S.imag, S.real)
        eta = jnp.abs(S)
        
        j = self.chan.single_channels[0,c_index]
        sigma = 0.5 * jnp.pi * (2*j+1) * (1-S).real / (self.k[k_index]**2)
        
        return (delta, eta, sigma)
        
            
            
        
    def observables_coupled(self, T_onshell, cc_index, k_index):
    
        I = jnp.identity(2)
        S = I - 1j * jnp.pi * self.factor * self.m * self.k[k_index] * T_onshell
        
        Z = 0.5 * ( S[0,1] + S[1,0] ) / jnp.sqrt(S[0,0] * S[1,1])
        epsilon = -0.25 * 1j * jnp.log( (1 + Z) / (1 - Z) )
        epsilon = epsilon.real
        
        if cc_index == 0:
            epsilon = jnp.where(epsilon < -1e-8, jnp.abs(epsilon), epsilon)
        
        S_minus = S[0,0] / jnp.cos(2 * epsilon)
        S_plus = S[1,1] / jnp.cos(2 * epsilon)
        
        delta_minus = 0.5 * jnp.arctan2(S_minus.imag, S_minus.real)
        delta_plus = 0.5 * jnp.arctan2(S_plus.imag, S_plus.real)
        delta_minus = jnp.where(delta_minus < -1e-8, jnp.pi + delta_minus, delta_minus)
        
        eta_minus = jnp.abs(S_minus)
        eta_plus = jnp.abs(S_plus)
        
        j = self.chan.coupled_channels[0,cc_index]
        sigma = 0.5 * jnp.pi * (2*j+1) * (2 - S_minus - S_plus).real / (self.k[k_index]**2) # use S_minus and S_plus?
        
        

            
        
        return (delta_minus, delta_plus, epsilon), (eta_minus, eta_plus), sigma
    

    
    def observables(self, training_data):
            
        data = self.empty_observable_data()
        
        if self.pot.compute_single:
            for c, chan_label in enumerate(self.chan.single_spect_not):
                for k in range(self.Nk):
                    T_onshell = training_data[chan_label][k]['Tq at best LECs'][-1][0]
                    delta, eta, sigma = self.observables_single(T_onshell, c, k)
                    data[chan_label]['phase shift'].append(delta)
                    data[chan_label]['unitarity'].append(eta)
                    data[chan_label]['cross section'].append(sigma)
                    
                data[chan_label]['phase shift'] = jnp.array(data[chan_label]['phase shift'])
                data[chan_label]['unitarity'] = jnp.array(data[chan_label]['unitarity'])
                data[chan_label]['cross section'] = jnp.array(data[chan_label]['cross section'])
                
       
        if self.pot.compute_coupled:
            for cc, chan_label in enumerate(self.chan.coupled_spect_not):
                for k in range(self.Nk):
                    T_onshell = training_data[chan_label][k]['Tq at best LECs'][-1][:,:,0]
                    print('T_onshell = ', T_onshell.shape)
                    deltas, etas, sigma = self.observables_coupled(T_onshell, cc, k)
                    data[chan_label]['phase shift'].append(deltas)
                    data[chan_label]['unitarity'].append(etas)
                    data[chan_label]['cross section'].append(sigma)
                    
                data[chan_label]['phase shift'] = jnp.array(data[chan_label]['phase shift'])
                data[chan_label]['unitarity'] = jnp.array(data[chan_label]['unitarity'])
                data[chan_label]['cross section'] = jnp.array(data[chan_label]['cross section'])
                    
        return data
    
    

    def train_greedy_GROM_single(self, LECs_lbd, LECs_ubd):
   
        if self.pot.compute_single:
        
            self.log_block_start("GREEDY GROM")

            # sample candidate training points
            self.log_category = "single"
            self.log_stage = "train greedy GROM"
            self.log_start(f"sampling {self.config.Ncand} candidate points")
            key = random.PRNGKey(self.config.seed)
            key, key_in = random.split(key)
            LECs_cand = latin_hypercube(key_in, self.config.Ncand, self.pot.No, minvals=LECs_lbd, maxvals=LECs_ubd)
            self.log_end()
            self.log_message(f"LECs_cand shape = {LECs_cand.shape}")
            
            # choose a small subset to start
            self.log_start(f"selecting {self.config.Ninit} initial training points")
            LECs_init = LECs_cand[:self.config.Ninit]
            self.log_end()
            self.log_message(f"LECs_init shape = {LECs_init.shape}")
            
            # the candidate LECs are the validation points by default
            LECs_val = LECs_cand
            
            # solve high fidelity model at all training points
            Tsckq_init = self.solve_single(LECs_init)

            # train single channel emulator
            for c, chan_label in enumerate(self.chan.single_spect_not):
                for k, Elab in enumerate(self.Elab):
                    
                    self.log_stage = "train greedy GROM"
                    self.log_message(f"training {chan_label} channel at Elab = {Elab} MeV...")
                    self.log_start(f"orthogonalizing {self.config.Ninit} initial training points")
                    Tqb_train = jnp.transpose(Tsckq_init[:,c,k,:])
                    LECs_train = LECs_init
                    Ntrain = self.config.Ninit
                    Xqb, _ = jnp.linalg.qr(Tqb_train)
                    self.log_end()

                    for iter in range(self.config.Nmax):
                    
                        # project
                        self.log_stage = "train greedy GROM"
                        self.log_message(f"greedy iteration = {iter}")
                        self.log_start("projecting")
                        XVGXobb = jnp.einsum('ia,oij,j,jb->oab',
                                              jnp.conjugate(Xqb), self.Vockqq[:,c,k,:,:], self.Gkq[k,:], Xqb)
                        XVob = jnp.einsum('ib,oi->ob', jnp.conjugate(Xqb), self.Vockqq[:,c,k,:,0])
                        self.log_end()
                        
                        # sample validation points and add the training points
                        if self.config.resample_val:
                            self.log_stage = "val greedy GROM"
                            self.log_start(f"sampling {self.config.Nval} new validation points")
                            key, key_in = random.split(key)
                            LECs_val = latin_hypercube(key_in, self.config.Nval, self.pot.No, minvals=LECs_lbd, maxvals=LECs_ubd)
                            LECs_val = jnp.concatenate((LECs_val, LECs_train))
                            self.log_end()
    
                        # solve low fidelity model at validation points
                        self.log_start(f"emulating at validation points")
                        XVGXsbb = jnp.einsum('so,oab->sab', LECs_val, XVGXobb)
                        XVsb = jnp.einsum('so,ob->sb', LECs_val, XVob)
                        ib = jnp.arange(Ntrain)
                        XAXsbb = (-XVGXsbb).at[:,ib,ib].add(1.0)
                        Csb = jnp.linalg.solve(XAXsbb, XVsb[...,jnp.newaxis])[...,0]
                        emulated_Tsq = jnp.einsum('sb,ib->si', Csb, Xqb)
                        self.log_end()
                        
                        # estimate error at validation points
                        self.log_start(f"estimating error at validation points")
                        VGsqq = jnp.einsum('so,oij,j->sij', LECs_val, self.Vockqq[:,c,k,:,:], self.Gkq[k,:])
                        Vsq = jnp.einsum('so,oi->si', LECs_val, self.Vockqq[:,c,k,:,0])
                        iq = jnp.arange(self.Nq+1)
                        Asqq = (-VGsqq).at[:,iq,iq].add(1.0)
                        estimated_Es = jnp.linalg.norm(jnp.einsum('sij,sj->si', Asqq, emulated_Tsq) - Vsq, axis=1)
                        self.log_end()
                        
                        # select validation point with largest estimated error
                        s_maxE = jnp.argmax(estimated_Es)
                        LECs_add = LECs_val[s_maxE]
                        
                        # calculate exact solution at validation point with largest error
                        self.log_start(f"solving high-fidelity model at validation point with largest error")
                        exact_Tq = jnp.linalg.solve(Asqq[s_maxE], Vsq[s_maxE])
                        self.log_end()
                        
                        # calibrate estimated error
                        self.log_start(f"calibrating error at validation points")
                        exact_E = jnp.linalg.norm(exact_Tq - emulated_Tsq[s_maxE])
                        calibration_ratio = exact_E / estimated_Es[s_maxE]
                        calibrated_Es = estimated_Es * calibration_ratio
                        self.log_end()
                        self.log_message(f"calibration ratio = {calibration_ratio:.8f}")
                        self.log_message(f"avg calibrated testing error = {jnp.mean(calibrated_Es):.8e}")
                        self.log_message(f"std calibrated testing error = {jnp.std(calibrated_Es):.8e}")
                        self.log_message(f"max calibrated testing error = {jnp.max(calibrated_Es):.8e}")
                        
                        self.log_start(f"evaluating calibrated error")
                            
                        # store emulator for this channel and energy once converged
                        if jnp.max(calibrated_Es) < self.config.err_tol:
                        
                            iters = jnp.arange(iter + 1)
                            self.log_end()
                            self.log_message(f"greedy algorithm done in {iter} iterations, used {Ntrain} training points")
                            
                            break
                            
                        # add validation point with largest error
                        Ntrain += 1
                        LECs_train = jnp.concatenate((LECs_train, LECs_add[jnp.newaxis,:]), axis=0)
                        #Xqb = gram_schmidt_insert(Xqb, exact_Tq)
                        #Xqb = householder_insert(Xqb, exact_Tq)
                        #Xqb = modified_gram_schmidt_insert(Xqb, exact_Tq)
                        
                        Tqb_train = jnp.append(Tqb_train, exact_Tq[:,jnp.newaxis], axis=1)
                        Xqb, _ = jnp.linalg.qr(Tqb_train)
                        
                        self.log_end()
                        self.log_message(f"added validation point with max error to training points, Tqb_train shape = {Tqb_train.shape}")
                        


            #self.emulator['greedy GROM'] = {'Xqb': Xqb_stored, 'XVGXobb': XVGXobb_stored, 'XVob': XVob_stored}
            #print("\nGreedy GROM emulator stored with label 'greedy GROM'.")
            
            
        self.log_block_end()
    
        



    def phase_shifts_old(self, T_single=None, T_coupled=None):
    
        if T_single is not None:
        
            # extract on-shell elements
            Tsckq = jnp.reshape(T_single, (-1, self.chan.Nsingle, self.Nk, self.Nq+1))
            Tsck = Tsckq[:,:,:,0]
            
            # convert to S matrix
            S_sck = 1 - 1j * self.factor * jnp.pi * self.m * self.k[jnp.newaxis,jnp.newaxis,:] * Tsck
            
            # compute single-channel phase shifts
            delta_sck = 0.5 * jnp.arctan2(S_sck.imag, S_sck.real)
            
            # compute eta
            eta_sck = jnp.abs(S_sck)
            
            # package output
            single_output = (delta_sck, eta_sck)
            
            
        else:
            single_output = None
        
        if T_coupled is not None:
        
            print("* ", T_coupled.shape)
        
            # extract on-shell elements
            Tscckq = jnp.reshape(T_coupled, (-1, self.chan.Ncoupled, self.Nk, 2, 2, self.Nq+1))
            Tscck = Tscckq[:,:,:,:,:,0]
            
            print("* ",Tscckq.shape)
            print("* ",Tscck.shape)

            # convert to S matrix
            S_scck = - 1j * self.factor * jnp.pi * self.m * self.k[jnp.newaxis,jnp.newaxis,:,jnp.newaxis,jnp.newaxis] * Tscck
            i = jnp.arange(2)
            S_scck = S_scck.at[:,:,:,i,i].add(1.0) # adding 1 to -- and ++
            
            # compute mixing angle epsilon
            z = 0.5 * ( S_scck[...,0,1] + S_scck[...,1,0] ) / jnp.sqrt(S_scck[...,0,0] * S_scck[...,1,1])
            epsilon_scck = -0.25 * 1j * jnp.log( (1 + z) / (1 - z) )
            epsilon_scck = epsilon_scck.real
            #epsilon_scck = jnp.where(epsilon_scck[:,0,:][:,None,:] < -1e-8, jnp.abs(epsilon_scck), epsilon_scck)
            epsilon_scck = epsilon_scck.at[:,0,:].set(jnp.where(epsilon_scck[:,0,:] < -1e-8, abs(epsilon_scck[:,0,:]), epsilon_scck[:,0,:]))
            
            # compute phase shifts
            S_minus_scck = S_scck[...,0,0] / jnp.cos(2 * epsilon_scck)
            S_plus_scck = S_scck[...,1,1] / jnp.cos(2 * epsilon_scck)

            delta_minus_scck = 0.5 * jnp.arctan2(S_minus_scck.imag, S_minus_scck.real)
            delta_plus_scck = 0.5 * jnp.arctan2(S_plus_scck.imag, S_plus_scck.real)
            
            # fix branch cuts
            delta_minus_scck = jnp.where(delta_minus_scck < -1e-8, jnp.pi + delta_minus_scck, delta_minus_scck)
            
            # compute etas
            eta_minus_scck = jnp.abs(S_minus_scck)
            eta_plus_scck = jnp.abs(S_plus_scck)
            
            # package output
            coupled_output = (delta_minus_scck, delta_plus_scck, epsilon_scck, eta_minus_scck, eta_plus_scck)
            
        else:
            coupled_output = None
            
        return single_output, coupled_output
        
    def log_start(self, message):
        self.logger.info(f"- {self.log_category} - {self.log_stage} - start - {message}")
        self.start_time = time.time()
        self.message = message

    def log_end(self):
        elapsed_time = time.time() - self.start_time
        self.logger.info(f"- {self.log_category} - {self.log_stage} - end - {self.message} (elapsed time: {elapsed_time:.5f} sec)")
        
    def log_message(self, message):
        self.logger.info(f"- {self.log_category} - {self.log_stage} - {message}")

    def log_block_start(self, message):
        self.logger.info(f">>> {message}")
        self.block_start_time = time.time()
        self.block_message = message
        
    def log_block_end(self):
        elapsed_time = time.time() - self.block_start_time
        self.logger.info(f"<<< {self.block_message} (total elapsed time: {elapsed_time:.5f} sec)")
    

    def precalc_single_pot(self):
    
        self.log_category = "single"
        self.log_stage = "setup"
        filename = 'saved_potentials/'+self.config.pot_file+'_c'  # c = single channel
    
        if self.config.load_pot:
        
            # load potential operators from file
            self.log_start("loading potential operators from file")
            with open(filename+'.pot', 'rb') as f:
                self.Vockqq = jnp.load(f)
            self.log_end()
            
    
        elif self.pot.compute_single:
            
            # call Potential object
            self.log_start("precalculating potential operators qq")
            Vocqq = self.pot.Voc(self.q)
            self.log_end()
            
            self.log_start("precalculating potential operators kq")
            Vockq = self.pot.Voc(self.k, self.q)
            self.log_end()
            
            self.log_start("precalculating potential operators k")
            Vock = self.pot.Voc(self.k, diag=True)
            self.log_end()
            
            # fill big potential matrix
            self.log_start("filling potential operators")
            self.Vockqq = jnp.zeros((self.pot.No, self.chan.Nsingle, self.Nk, self.Nq+1, self.Nq+1), dtype=jnp.complex128)
            self.Vockqq = self.Vockqq.at[:,:,:,1:,1:].set( jnp.tile(Vocqq[:,:,jnp.newaxis,:,:], (1,1,self.Nk,1,1)) )
            self.Vockqq = self.Vockqq.at[:,:,:,1:,0].set( Vockq )
            self.Vockqq = self.Vockqq.at[:,:,:,0,1:].set( Vockq )
            self.Vockqq = self.Vockqq.at[:,:,:,0,0].set( Vock )
            self.log_end()
            
            # write to file
            self.config.write_info(filename+'.info')
            with open(filename+'.pot', 'wb') as f:
                jnp.save(f, self.Vockqq, allow_pickle=False)
            
            
            # plot for debugging
            '''
            k = 1
            for o in range(self.pot.No):
                for c in range(self.chan.Nsingle):
                
                    fig, ax = plt.subplots(1, 1, figsize=(4,4))
                    ax.imshow(self.Vockqq[o,c,k].real)
                    plt.suptitle(f"Operator {o}, {self.chan.single_spect_not[c]} Channel")
                    plt.show()
                    plt.close()
            '''
            
            
    def precalc_coupled_pot(self):
    

        self.log_category = "coupled"
        self.log_stage = "setup"
        filename = 'saved_potentials/'+self.config.pot_file+'_cc'  # cc = coupled channel
    
        if self.config.load_pot:
        
            # load potential operators from file
            self.log_start("loading potential operators from file")
            with open(filename+'.pot', 'rb') as f:
                self.Vocckqq = jnp.load(f)
            self.log_end()
        
        elif self.pot.compute_coupled:
        
            # precalculate potential operators for all channels and energies
            self.log_start("precalculating potential operators qq")
            Voccqq = self.pot.Vocc(self.q)
            self.log_end()
            
            self.log_start("precalculating potential operators kq")
            Vocckq = self.pot.Vocc(self.k, self.q)
            self.log_end()

            self.log_start("precalculating potential operators k")
            Vocck = self.pot.Vocc(self.k, diag=True)
            self.log_end()
        
            # unpack
            Voccqq_mm, Voccqq_pp, Voccqq_mp, Voccqq_pm = Voccqq
            Vocckq_mm, Vocckq_pp, Vocckq_mp, Vocckq_pm = Vocckq
            Vocck_mm, Vocck_pp, Vocck_mp, Vocck_pm = Vocck
        
            self.log_start("filling potential operators")
            self.Vocckqq = jnp.zeros((self.pot.No, self.chan.Ncoupled, self.Nk, 2, 2, self.Nq+1, self.Nq+1), dtype=jnp.complex128)
            
            self.Vocckqq = self.Vocckqq.at[:,:,:,0,0,1:,1:].set( jnp.tile(Voccqq_mm[:,:,jnp.newaxis,:,:], (1,1,self.Nk,1,1)) )
            self.Vocckqq = self.Vocckqq.at[:,:,:,1,1,1:,1:].set( jnp.tile(Voccqq_pp[:,:,jnp.newaxis,:,:], (1,1,self.Nk,1,1)) )
            self.Vocckqq = self.Vocckqq.at[:,:,:,0,1,1:,1:].set( jnp.tile(Voccqq_mp[:,:,jnp.newaxis,:,:], (1,1,self.Nk,1,1)) )
            self.Vocckqq = self.Vocckqq.at[:,:,:,1,0,1:,1:].set( jnp.tile(Voccqq_pm[:,:,jnp.newaxis,:,:], (1,1,self.Nk,1,1)) )
            
            self.Vocckqq = self.Vocckqq.at[:,:,:,0,0,1:,0].set( Vocckq_mm )
            self.Vocckqq = self.Vocckqq.at[:,:,:,1,1,1:,0].set( Vocckq_pp )
            self.Vocckqq = self.Vocckqq.at[:,:,:,0,1,1:,0].set( Vocckq_pm )
            self.Vocckqq = self.Vocckqq.at[:,:,:,1,0,0,1:].set( Vocckq_pm )
            
            self.Vocckqq = self.Vocckqq.at[:,:,:,0,0,0,1:].set( Vocckq_mm )
            self.Vocckqq = self.Vocckqq.at[:,:,:,1,1,0,1:].set( Vocckq_pp )
            self.Vocckqq = self.Vocckqq.at[:,:,:,0,1,0,1:].set( Vocckq_mp )
            self.Vocckqq = self.Vocckqq.at[:,:,:,1,0,1:,0].set( Vocckq_mp )

            
            self.Vocckqq = self.Vocckqq.at[:,:,:,0,0,0,0].set( Vocck_mm )
            self.Vocckqq = self.Vocckqq.at[:,:,:,1,1,0,0].set( Vocck_pp )
            self.Vocckqq = self.Vocckqq.at[:,:,:,0,1,0,0].set( Vocck_mp )
            self.Vocckqq = self.Vocckqq.at[:,:,:,1,0,0,0].set( Vocck_pm )
            
            
            # plot for debugging
            '''
            k = 0
            for o in range(self.pot.No):
                for cc in range(self.chan.Ncoupled):
                    fig, ax = plt.subplots(2, 2, figsize=(8,8))
                    ax[0,0].imshow(self.Vocckqq[o,cc,k,0,0].real)
                    ax[1,1].imshow(self.Vocckqq[o,cc,k,1,1].real)
                    ax[0,1].imshow(self.Vocckqq[o,cc,k,0,1].real)
                    ax[1,0].imshow(self.Vocckqq[o,cc,k,1,0].real)
                    plt.suptitle(f"Operator {o}, {self.chan.coupled_spect_not[cc]} Channel")
                    plt.show()
                    plt.close()
            '''
            
            self.Vocckqq = jnp.transpose(self.Vocckqq, (0,1,2,3,5,4,6))
            self.Vocckqq = jnp.reshape(self.Vocckqq, (self.pot.No, self.chan.Ncoupled, self.Nk, 2*self.Nq+2, 2*self.Nq+2))
            self.log_end()
            
            # write to file
            self.config.write_info(filename+'.info')
            with open(filename+'.pot', 'wb') as f:
                jnp.save(f, self.Vocckqq, allow_pickle=False)
                
                
        
    def precalc_prop(self):
        """ Precalculates the propagator for all energies k and momenta on the grid q using the subtraction method. 
            The stored array Gkq has shape (Nk, Nq+1). """
    
        # precalculate propagator for all energies
        # units: MeV^2
        self.log_category = "common"
        self.log_stage = "setup"
        self.log_start("precalculating propagators")
        
        if self.map.inf:
            Ck = jnp.zeros(self.Nk)
        else:
            qmax = jnp.max(self.q)
            Ck = jnp.log((qmax + self.k) / (qmax - self.k)) # correction for finite map
        Ck = Ck.astype(jnp.complex128) - 1j * jnp.pi
        Bkq = self.wq / ((self.q**2)[jnp.newaxis,:] - (self.k**2)[:,jnp.newaxis]) # fm
        Bk = jnp.sum(Bkq, axis=1)
        
        self.Gkq = jnp.zeros((self.Nk, self.Nq+1), dtype=jnp.complex128)
        self.Gkq = self.Gkq.at[:,0].set( self.m * self.k * (0.5 * Ck + self.k * Bk) ) # fm^-2
        self.Gkq = self.Gkq.at[:,1:].set( - self.m * Bkq * (self.q**2)[jnp.newaxis,:] ) # fm^-2
        self.Gkq *= self.factor # different normalization conventions
        self.log_end()
        

    def empty_training_data(self):

        data = {
            chan_label: {
                k: {
                    'emulator': None,
                    'Tq at best LECs': [],
                    'max cal error': []
                }
                for k in range(self.Nk)
            }
            for chan_label in (self.chan.single_spect_not + self.chan.coupled_spect_not)
        }
        
        return data


    def empty_observable_data(self):

        data = {
            chan_label: {
                'phase shift': [],
                'unitarity': [],
                'cross section': [],
            }
            for chan_label in (self.chan.single_spect_not + self.chan.coupled_spect_not)
        }
        
        return data

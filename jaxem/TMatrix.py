import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
from jax import random, grad, jit, vmap
from scipy.interpolate import griddata, RBFInterpolator
from functools import partial
import matplotlib.pyplot as plt
from .Tools import latin_hypercube
from .Potential import Potential
from .Map import Map
import matplotlib.pyplot as plt
import time
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

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
        self.m = config.m
        self.output = config.output
        self.tol = config.tol
        self.seed = config.seed
        self.factor = config.factor
        
        self.pot = Potential(config)
        self.map = Map(config)
        self.chan = self.pot.chan
        
        # compute all poles k
        self.Elab = jnp.sort(config.Elab)
        self.k = jnp.array([jnp.sqrt(0.5 * self.m * Elab) for Elab in self.Elab]) # MeV
        self.Nk = self.k.shape[0]
        
        # make momentum grid q
        self.Nq = config.Nq
        self.q, self.wq = self.map.grid() # MeV
        
        # precalculate propagator for all energies
        # units: MeV^2
        print("Precalculating propagators...", end="", flush=True)
        ti = time.time()
        
        if self.map.inf:
            Ck = jnp.zeros(self.Nk)
        else:
            qmax = jnp.max(self.q)
            Ck = jnp.log((qmax + self.k) / (qmax - self.k)) # correction for finite map
        Ck = Ck.astype(jnp.complex128) - 1j * jnp.pi
        Bkq = self.wq / ((self.q**2)[jnp.newaxis,:] - (self.k**2)[:,jnp.newaxis]) # MeV^-1
        Bk = jnp.sum(Bkq, axis=1)
        
        self.Gkq = jnp.zeros((self.Nk, self.Nq+1), dtype=jnp.complex128)
        self.Gkq = self.Gkq.at[:,0].set( self.m * self.k * (0.5 * Ck + self.k * Bk) ) # MeV^2
        self.Gkq = self.Gkq.at[:,1:].set( - self.m * Bkq * (self.q**2)[jnp.newaxis,:] )
        self.Gkq *= config.factor # different normalization conventions
        tf = time.time()
        print(f"Done in {tf-ti:.3f} sec.")
        
        # precalculate potential operators for all channels and energies
        print("Precalculating potential operators...", end="", flush=True)
        ti = time.time()
        
        #Vocqq, Voccqq = self.pot.Voc(self.q)
        Vocqq, Voccqq = self.pot.Voc(self.q, self.q)
        Vockq, Vocckq = self.pot.Voc(self.k, self.q)
        Vock, Vocck = self.pot.Voc(self.k, diag=True)
                
        if self.pot.compute_single:
        
            self.Vockqq = jnp.zeros((self.pot.No, self.chan.Nsingle, self.Nk, self.Nq+1, self.Nq+1), dtype=jnp.complex128)
            self.Vockqq = self.Vockqq.at[:,:,:,1:,1:].set( jnp.tile(Vocqq[:,:,jnp.newaxis,:,:], (1,1,self.Nk,1,1)) )
            self.Vockqq = self.Vockqq.at[:,:,:,1:,0].set( Vockq )
            self.Vockqq = self.Vockqq.at[:,:,:,0,1:].set( Vockq )
            self.Vockqq = self.Vockqq.at[:,:,:,0,0].set( Vock )

        if self.pot.compute_coupled:
        
            Voccqq_mm, Voccqq_pp, Voccqq_mp, Voccqq_pm = Voccqq
            Vocckq_mm, Vocckq_pp, Vocckq_mp, Vocckq_pm = Vocckq
            Vocck_mm, Vocck_pp, Vocck_mp, Vocck_pm = Vocck
        
            self.Vocckqq = jnp.zeros((self.pot.No, self.chan.Ncoupled, self.Nk, 2, 2, self.Nq+1, self.Nq+1), dtype=jnp.complex128)
            
            self.Vocckqq = self.Vocckqq.at[:,:,:,0,0,1:,1:].set( jnp.tile(Voccqq_mm[:,:,jnp.newaxis,:,:], (1,1,self.Nk,1,1)) )
            self.Vocckqq = self.Vocckqq.at[:,:,:,1,1,1:,1:].set( jnp.tile(Voccqq_pp[:,:,jnp.newaxis,:,:], (1,1,self.Nk,1,1)) )
            self.Vocckqq = self.Vocckqq.at[:,:,:,0,1,1:,1:].set( jnp.tile(Voccqq_mp[:,:,jnp.newaxis,:,:], (1,1,self.Nk,1,1)) )
            self.Vocckqq = self.Vocckqq.at[:,:,:,1,0,1:,1:].set( jnp.tile(Voccqq_pm[:,:,jnp.newaxis,:,:], (1,1,self.Nk,1,1)) )
            
            self.Vocckqq = self.Vocckqq.at[:,:,:,0,0,1:,0].set( Vocckq_mm )
            self.Vocckqq = self.Vocckqq.at[:,:,:,1,1,1:,0].set( Vocckq_pp )
            self.Vocckqq = self.Vocckqq.at[:,:,:,0,1,1:,0].set( Vocckq_pm ) # crucial
            self.Vocckqq = self.Vocckqq.at[:,:,:,1,0,0,1:].set( Vocckq_pm ) # crucial
            
            self.Vocckqq = self.Vocckqq.at[:,:,:,0,0,0,1:].set( Vocckq_mm )
            self.Vocckqq = self.Vocckqq.at[:,:,:,1,1,0,1:].set( Vocckq_pp )
            self.Vocckqq = self.Vocckqq.at[:,:,:,0,1,0,1:].set( Vocckq_mp ) # crucial
            self.Vocckqq = self.Vocckqq.at[:,:,:,1,0,1:,0].set( Vocckq_mp ) # crucial

            
            self.Vocckqq = self.Vocckqq.at[:,:,:,0,0,0,0].set( Vocck_mm )
            self.Vocckqq = self.Vocckqq.at[:,:,:,1,1,0,0].set( Vocck_pp )
            self.Vocckqq = self.Vocckqq.at[:,:,:,0,1,0,0].set( Vocck_mp )
            self.Vocckqq = self.Vocckqq.at[:,:,:,1,0,0,0].set( Vocck_pm )
            
            self.Vocckqq = jnp.transpose(self.Vocckqq, (0,1,2,3,5,4,6))
            self.Vocckqq = jnp.reshape(self.Vocckqq, (self.pot.No, self.chan.Ncoupled, self.Nk, 2*self.Nq+2, 2*self.Nq+2))
            
            '''
            self.Vocckqq = jnp.block([ [self.Vocckqq[:, :, :, 0, 0, :, :], self.Vocckqq[:, :, :, 0, 1, :, :]],
                                       [self.Vocckqq[:, :, :, 1, 0, :, :], self.Vocckqq[:, :, :, 1, 1, :, :]] ])
            '''
            print("*** ", self.Vocckqq.shape)
        
        tf = time.time()
        print(f"Done in {tf-ti:.3f} sec.")
        
        
        # check map for each channel and operator
        k = 10 # index of energy / pole
        
        
        if self.pot.compute_single:
        
            for o in range(self.pot.No):
            
                fig, ax = plt.subplots(1, self.chan.Nsingle, figsize=(self.chan.Nsingle * 4, 4))
                
                if self.chan.Nsingle == 1:
                    ax = [ax]
                
                for c in range(self.chan.Nsingle):
                    vmin = jnp.min(self.Vockqq[o,c,k])
                    vmax = jnp.max(self.Vockqq[o,c,k])
                    vmax = max(jnp.abs(vmin), jnp.abs(vmax))
                    ax[c].imshow(self.Vockqq[o,c,k].real, vmin=-vmax, vmax=vmax, cmap='bwr')
                    ax[c].set_title(f"{self.chan.single_spect_not[c]} Channel")
                    ax[c].axhline(-0.5, color='k', linestyle='dashed', linewidth=0.2)
                    ax[c].axhline(0.5, color='k', linestyle='dashed', linewidth=0.2)
                    ax[c].axvline(-0.5, color='k', linestyle='dashed', linewidth=0.2)
                    ax[c].axvline(0.5, color='k', linestyle='dashed', linewidth=0.2)

                fig.suptitle(f"Operator {o}")
                plt.show()
                plt.close()
        
            
        if self.pot.compute_coupled:
            
            for o in range(self.pot.No):
            
                fig, ax = plt.subplots(1, self.chan.Ncoupled, figsize=(self.chan.Ncoupled * 6, 6))
                
                if self.chan.Ncoupled == 1:
                    ax = [ax]
                
                for cc in range(self.chan.Ncoupled):
                    vmin = jnp.min(self.Vocckqq[o,cc,k])
                    vmax = jnp.max(self.Vocckqq[o,cc,k])
                    vmax = max(jnp.abs(vmin), jnp.abs(vmax))
                    #vmax = 0.00001 * vmax
                    ax[cc].imshow(self.Vocckqq[o,cc,k].real, vmin=-vmax, vmax=vmax, cmap='bwr')
                    ax[cc].set_title(f"{self.chan.coupled_spect_not[cc]} Channel")
                    ax[cc].axhline(-0.5, color='k', linestyle='dashed', linewidth=0.2)
                    ax[cc].axhline(0.5, color='k', linestyle='dashed', linewidth=0.2)
                    ax[cc].axhline(self.Nq+0.5, color='k', linestyle='dashed', linewidth=0.2)
                    ax[cc].axhline(self.Nq+1.5, color='k', linestyle='dashed', linewidth=0.2)
                    ax[cc].axvline(-0.5, color='k', linestyle='dashed', linewidth=0.2)
                    ax[cc].axvline(0.5, color='k', linestyle='dashed', linewidth=0.2)
                    ax[cc].axvline(self.Nq+0.5, color='k', linestyle='dashed', linewidth=0.2)
                    ax[cc].axvline(self.Nq+1.5, color='k', linestyle='dashed', linewidth=0.2)
                
                fig.suptitle(f"Operator {o}")
                plt.show()
                plt.close()
        
        
        
        # pytrees for storing emulators
        self.emulator = {'greedy': {}, 'POD': {}}
            
            
    def solve(self, LECs_samples):
    
        LECs_samples = jnp.reshape(LECs_samples, (-1, self.pot.No))
        Nsamples = LECs_samples.shape[0]
        
        print(f"Solving the high-fidelity model for {Nsamples} samples...", flush=True)
        ti = time.time()
    
        if self.pot.compute_single:
    
            # compute potential matrices for all LEC combinations, single channels, and energies
            Vsckqq = jnp.einsum('so,ockij->sckij', LECs_samples, self.Vockqq)
            
            # set up linear system
            iq = jnp.arange(self.Nq+1)
            Asckqq = -jnp.einsum('sckij,kj->sckij', Vsckqq, self.Gkq)
            Asckqq = Asckqq.at[:,:,:,iq,iq].add(1.0)
            
            # solve for half-shell T matrix for all LECs, single channels, and energies
            Tsckq = jnp.linalg.solve(Asckqq, Vsckqq[:,:,:,:,0][..., None])[..., 0]
            
        else:
        
            Tsckq = None
            
        tf = time.time()
        print(f"Done with single channels in {tf-ti:.3f} sec.")
        ti = time.time()
            
        if self.pot.compute_coupled:
    
            # compute potential matrices for all LEC combinations, coupled channels, and energies
            # units: MeV^-2
            Vscckqq = jnp.einsum('so,ockij->sckij', LECs_samples, self.Vocckqq)
  
            # set up linear system
            iq = jnp.arange(2*self.Nq+2)
            Ascckqq = -jnp.einsum('sckij,kj->sckij', Vscckqq, jnp.tile(self.Gkq, (1,2)))
            Ascckqq = Ascckqq.at[:,:,:,iq,iq].add(1.0)

            
            # solve for half-shell T matrix
            Tscckq = jnp.linalg.solve(Ascckqq, Vscckqq[:,:,:,:,[0, self.Nq + 1]])
            
            print("**** T", Tscckq.shape)
            
            Tscckq = jnp.reshape(Tscckq, (Nsamples, self.chan.Ncoupled, self.Nk, 2, self.Nq+1, 2))
            Tscckq = jnp.transpose(Tscckq, (0,1,2,3,5,4))

            print("**** T", Tscckq.shape)
        
        else:
        
            Tscckq = None
            
        tf = time.time()
        print(f"Done with coupled channels in {tf-ti:.3f} sec.")
        
        return Tsckq, Tscckq
        
        
            
        
    
    def phase_shifts(self, T_single=None, T_coupled=None):
    
        # use stapp parameterization with epsilon = 0 for single
        
    
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
        
            # extract on-shell elements
            Tscckq = jnp.reshape(T_coupled, (-1, self.chan.Ncoupled, self.Nk, 2, 2, self.Nq+1))
            Tscck = Tscckq[:,:,:,:,:,0]
            print("")
            
            
            # convert to S matrix
            S_scck = - 1j * self.factor * jnp.pi * self.m * self.k[jnp.newaxis,jnp.newaxis,:,jnp.newaxis,jnp.newaxis] * Tscck
            i = jnp.arange(2)
            S_scck = S_scck.at[:,:,:,i,i].add(1.0) # adding 1 to -- and ++
            
            # compute mixing angle epsilon
            z = - 0.5 * 1j * ( S_scck[...,0,1] + S_scck[...,1,0] ) / jnp.sqrt(S_scck[...,0,0] * S_scck[...,1,1])
            epsilon_scck = - 0.25 * 1j * jnp.log( (1 + 1j * z) / (1 - 1j * z) )
            print(epsilon_scck)
            epsilon_scck = epsilon_scck.real
            #epsilon_scck = jnp.zeros_like(epsilon_scck)
            
            # compute phase shifts
            S_minus_scck = S_scck[...,0,0] / jnp.cos(2 * epsilon_scck)
            S_plus_scck = S_scck[...,1,1] / jnp.cos(2 * epsilon_scck)

            delta_minus_scck = 0.5 * jnp.arctan2(S_minus_scck.imag, S_minus_scck.real)
            delta_plus_scck = 0.5 * jnp.arctan2(S_plus_scck.imag, S_plus_scck.real)
            
            # compute etas
            eta_minus_scck = jnp.abs(S_minus_scck)
            eta_plus_scck = jnp.abs(S_plus_scck)
            print(eta_minus_scck)
            print(eta_plus_scck)
            
            # package output
            coupled_output = (delta_minus_scck, delta_plus_scck, epsilon_scck, eta_minus_scck, eta_plus_scck)
            
        else:
            coupled_output = None
            
        return single_output, coupled_output
        
    
            
        
            
    def train_POD_GROM(self, LECs_lbd, LECs_ubd, Ntrain=20, Ntest=100):
    
        # sample training LECs
        key = random.PRNGKey(self.seed)
        key, key_in = random.split(key)
        LECs_train = latin_hypercube(key_in, Ntrain, self.pot.No, minvals=LECs_lbd, maxvals=LECs_ubd)
        print("LECs_train = ", LECs_train.shape)
        
        # solve high fidelity model at all training points
        Tsckq_train, Tscckq_train = self.solve(LECs_train)
        
        # train single channel emulator
        if self.pot.compute_single:
            for c, chan_label in enumerate(self.chan.single_spect_not):
                for k, Elab in enumerate(self.Elab):
                    
                    print(f"\nTraining POD GROM emulator for {chan_label} channel at Elab = {Elab} MeV...")
                    
                    # select training data
                    Tqs = jnp.transpose(Tsckq_train[:,c,k,:])
                    
                    # orthogonalize
                    Uqs, Ss, _ = jnp.linalg.svd(Tqs, full_matrices=False)
                    
                    # truncate basis
                    index = int(jnp.argmax(Ss/Ss[0] <= self.tol))
                    Nbasis = index + 1 if index > 0 else Ntrain
                    Xqb = Uqs[:,:Nbasis]
                    
                    # project
                    XVGXobb = jnp.einsum('ia,oij,j,jb->oab',
                                          jnp.conjugate(Xqb), self.Vockqq[:,c,k,:,:], self.Gkq[k,:], Xqb)
                    XVob = jnp.einsum('ib,oi->ob', jnp.conjugate(Xqb), self.Vockqq[:,c,k,:,0])
                    
                    # sample testing points
                    key, key_in = random.split(key)
                    LECs_test = latin_hypercube(key_in, Ntest, self.pot.No, minvals=LECs_lbd, maxvals=LECs_ubd)
                    print("LECs_test = ", LECs_test.shape)
                    
                    # solve low fidelity model at testing points
                    XVGXsbb = jnp.einsum('so,oab->sab', LECs_test, XVGXobb)
                    XVsb = jnp.einsum('so,ob->sb', LECs_test, XVob)
                    ib = jnp.arange(Nbasis)
                    XAXsbb = (-XVGXsbb).at[:,ib,ib].add(1.0)
                    Csb = jnp.linalg.solve(XAXsbb, XVsb[...,jnp.newaxis])[...,0]
                    print("Csb = ", Csb.shape)
                    
                    # reconstruct emulated solution
                    emulated_Tsq = jnp.einsum('sb,ib->si', Csb, Xqb)
                    print("emulated Tsq = ", emulated_Tsq.shape)
                    
                    # estimate error at testing points
                    VGsqq = jnp.einsum('so,oij,j->sij', LECs_test, self.Vockqq[:,c,k,:,:], self.Gkq[k,:])
                    Vsq = jnp.einsum('so,oi->si', LECs_test, self.Vockqq[:,c,k,:,0])
                    iq = jnp.arange(self.Nq+1)
                    Asqq = (-VGsqq).at[:,iq,iq].add(1.0)
                    Es = jnp.linalg.norm(jnp.einsum('sij,sj->si', Asqq, emulated_Tsq) - Vsq, axis=1)
                    
                    # also try exact solution for now
                    exact_Tsq = jnp.linalg.solve(Asqq, Vsq[...,jnp.newaxis])[...,0]
                    
                    # plot
                    o1, o2 = 0, 1
                    Ngrid = 100
                    x = jnp.linspace(LECs_lbd[o1], LECs_ubd[o1], Ngrid)
                    y = jnp.linspace(LECs_lbd[o2], LECs_ubd[o2], Ngrid)
                    grid_x, grid_y = jnp.meshgrid(x, y)
                    grid_z = griddata(LECs_test[:,[o1, o2]], Es, (grid_x, grid_y), method='cubic')
                    
                    plt.contourf(grid_x, grid_y, grid_z)
                    plt.scatter(LECs_train[:,o1], LECs_train[:,o2], color='r')
                    plt.show()
                    
                    for s in range(10):
                        plt.plot(exact_Tsq[s,:].real)
                        plt.plot(emulated_Tsq[s,:].real, linestyle='dashed')
                        plt.grid()
                        plt.show()
                        plt.close()

    def train_greedy_GROM(self, LECs_lbd, LECs_ubd, Ninit=1, Nval=1000, Nmax=20, o1=0, o2=1, Ngrid=100):
    
        # initial key
        key = random.PRNGKey(self.seed)
    
        '''
        # initial training points are the corners of the boundary
        io = jnp.arange(self.pot.No)
        LECs_init = jnp.tile(LECs_lbd[jnp.newaxis,:], (self.pot.No, 1))
        LECs_init = LECs_init.at[io,io].set(LECs_ubd)
        print("LECs_init = ", LECs_init.shape)
        '''
        key, key_in = random.split(key)
        LECs_init = latin_hypercube(key_in, Ninit, self.pot.No, minvals=LECs_lbd, maxvals=LECs_ubd)
        
        # solve high fidelity model at all training points
        Tsckq_init, Tscckq_init = self.solve(LECs_init)
        
        # store this emulator as a pytree
        Xqb_stored = {c: {str(Elab): [] for Elab in self.Elab} for c in self.chan.all_spect_not}
        XVGXobb_stored = {c: {str(Elab): [] for Elab in self.Elab} for c in self.chan.all_spect_not}
        XVob_stored = {c: {str(Elab): [] for Elab in self.Elab} for c in self.chan.all_spect_not}
        
        
        
        # train single channel emulator
        if self.pot.compute_single:
            for c, chan_label in enumerate(self.chan.single_spect_not):
                for k, Elab in enumerate(self.Elab):
                    
                    print("\n" + "-"*80)
                    print(f"Training greedy GROM emulator for {chan_label} channel at Elab = {Elab} MeV... ")
                    
                    # select initial training data
                    Tqs_train = jnp.transpose(Tsckq_init[:,c,k,:])
                    #print("avg T = ", jnp.mean(Tqs_train))
                    
                    # select corresponding LECs
                    LECs_train = LECs_init
                    
                    # store errors during training
                    avgE = []
                    stdE = []
                    maxE = []
                    
                    
                    for iter in range(Nmax):
                    
                        print(f"\nIteration = {iter}")
                        
                        print("Orthogonalizing, truncating, and projecting... ", end="")
                        ti = time.time()
                        
                        # orthogonalize
                        Uqs, Ss, Vss = jnp.linalg.svd(Tqs_train, full_matrices=False)
                        
                        # truncate basis
                        index = int(jnp.argmax(Ss/Ss[0] <= self.tol))
                        Nbasis = index + 1 if index > 0 else Ss.shape[0]
                        Xqb = Uqs[:,:Nbasis]
                        #print("index = ", index)
                        #print("Nbasis = ", Nbasis)
                        #print("S/S0 = ", Ss[:Nbasis]/Ss[0])
                        
                        # project
                        XVGXobb = jnp.einsum('ia,oij,j,jb->oab',
                                              jnp.conjugate(Xqb), self.Vockqq[:,c,k,:,:], self.Gkq[k,:], Xqb)
                        XVob = jnp.einsum('ib,oi->ob', jnp.conjugate(Xqb), self.Vockqq[:,c,k,:,0])
                        
                        tf = time.time()
                        print(f"Done in {tf-ti:.3f} sec.")
                        
                        print("Validating... ", end="")
                        ti = time.time()
                        
                        # sample validation points and add the training points
                        key, key_in = random.split(key)
                        LECs_val = latin_hypercube(key_in, Nval, self.pot.No, minvals=LECs_lbd, maxvals=LECs_ubd)
                        LECs_val = jnp.concatenate((LECs_val, LECs_train))
    
                        # solve low fidelity model at validation points
                        XVGXsbb = jnp.einsum('so,oab->sab', LECs_val, XVGXobb)
                        XVsb = jnp.einsum('so,ob->sb', LECs_val, XVob)
                        ib = jnp.arange(Nbasis)
                        XAXsbb = (-XVGXsbb).at[:,ib,ib].add(1.0)
                        Csb = jnp.linalg.solve(XAXsbb, XVsb[...,jnp.newaxis])[...,0]
                        
                        # reconstruct emulated solution
                        emulated_Tsq = jnp.einsum('sb,ib->si', Csb, Xqb)
                        
                        # estimate error at validation points
                        VGsqq = jnp.einsum('so,oij,j->sij', LECs_val, self.Vockqq[:,c,k,:,:], self.Gkq[k,:])
                        Vsq = jnp.einsum('so,oi->si', LECs_val, self.Vockqq[:,c,k,:,0])
                        iq = jnp.arange(self.Nq+1)
                        Asqq = (-VGsqq).at[:,iq,iq].add(1.0)
                        estimated_Es = jnp.linalg.norm(jnp.einsum('sij,sj->si', Asqq, emulated_Tsq) - Vsq, axis=1)
                        
                        tf = time.time()
                        print(f"Done in {tf-ti:.3f} sec.")
                        
                        print("Selecting new training point and calibrating error... ", end="")
                        ti = time.time()
                        
                        # select validation point with largest estimated error
                        s_maxE = jnp.argmax(estimated_Es)
                        LECs_add = LECs_val[s_maxE]
                        
                        # calculate exact solution at validation point with largest error
                        exact_Tq = jnp.linalg.solve(Asqq[s_maxE], Vsq[s_maxE])
                        
                        # calibrate estimated error
                        exact_E = jnp.linalg.norm(exact_Tq - emulated_Tsq[s_maxE])
                        calibration_ratio = exact_E / estimated_Es[s_maxE]
                        calibrated_Es = estimated_Es * calibration_ratio
                        
                        # store calibrated error statistics
                        avgE.append(jnp.mean(calibrated_Es))
                        stdE.append(jnp.std(calibrated_Es))
                        maxE.append(jnp.max(calibrated_Es))
                        
                        tf = time.time()
                        print(f"Done in {tf-ti:.3f} sec.")
                        print(f"Calibration ratio = ", calibration_ratio)
                        print("Avg calibrated validation error = ", avgE[-1])
                        print("Std calibrated validation error = ", stdE[-1])
                        print("Max calibrated validation error = ", maxE[-1])
                        
                        '''
                        # exact error
                        exact_Tsq = jnp.linalg.solve(Asqq, Vsq[...,jnp.newaxis])[...,0]
                        exact_Es = jnp.linalg.norm(exact_Tsq-emulated_Tsq, axis=1)
                        print("Max exact validation error = ", jnp.max(exact_Es))
                        print("Min exact validation error = ", jnp.min(exact_Es))
                        
                        # ratio of errors
                        ratio_Es = estimated_Es / exact_Es
                        print("avg ratio of errors = ", jnp.mean(ratio_Es))
                        print("std ratio of errors = ", jnp.std(ratio_Es))
                        '''

                        # plot so that white = tol, blue = less than tol, red = more than tol
                        x = jnp.linspace(LECs_lbd[o1], LECs_ubd[o1], Ngrid)
                        y = jnp.linspace(LECs_lbd[o2], LECs_ubd[o2], Ngrid)
                        grid_x, grid_y = jnp.meshgrid(x, y)
                        grid_z = griddata(LECs_val[:,[o1, o2]], calibrated_Es, (grid_x, grid_y), method='nearest', rescale=True)
                        
                        exponent = int(jnp.log10(self.tol))
                        diff = 7
                        levels = jnp.logspace(exponent-diff, exponent+diff, num=100)
                        grid_z = jnp.minimum(grid_z, 10**(exponent+diff))
                        grid_z = jnp.maximum(grid_z, 10**(exponent-diff))
                        grid_z = gaussian_filter(grid_z, sigma=0.5)
                        
                        plt.figure(figsize=(6, 6), dpi=300)
                        plt.contourf(grid_x, grid_y, grid_z, cmap='bwr', levels=levels, norm=mcolors.LogNorm(vmin=10**(exponent-diff), vmax=10**(exponent+diff)))
                        plt.scatter(LECs_train[:, o1], LECs_train[:, o2], color='limegreen', s=100, edgecolors='k', linewidth=0.3)
                        plt.scatter(LECs_add[o1], LECs_add[o2], color='cyan', s=100, edgecolors='k', linewidth=0.3)
                        plt.scatter(LECs_val[:Nval, o1], LECs_val[:Nval, o2], color='white', s=5, edgecolors='k', linewidth=0.3)
                        plt.savefig(f"figures/{self.output}_{chan_label}_{Elab}_err_o{o1}_o{o2}_{iter}.png", format="png")
                        plt.close()
                        
                        fig, ax = plt.subplots(2, 1, figsize=(6, 9), dpi=300)
                        ax[0].plot(self.q / self.hbarc, exact_Tq.real[1:] * self.hbarc**3 )
                        ax[0].plot(self.q / self.hbarc, emulated_Tsq[s_maxE,1:].real * self.hbarc**3 , linestyle='dashed')
                        ax[1].plot(self.q / self.hbarc, exact_Tq.imag[1:] * self.hbarc**3)
                        ax[1].plot(self.q / self.hbarc, emulated_Tsq[s_maxE,1:].imag * self.hbarc**3, linestyle='dashed')
                        ax[0].grid()
                        ax[1].grid()
                        ax[0].set_xlim(0, 15)
                        ax[0].set_xlim(0, 15)
                        plt.savefig(f"figures/{self.output}_{chan_label}_{Elab}_T_{iter}.png", format="png")
                        plt.close()
                            
                            
                        # store emulator for this channel and energy once converged
                        if jnp.max(calibrated_Es) < self.tol:
                        
                            Elab_label = str(Elab)
                        
                            Xqb_stored[chan_label][Elab_label] = Xqb
                            XVGXobb_stored[chan_label][Elab_label] = XVGXobb
                            XVob_stored[chan_label][Elab_label] = XVob
                            
                            iters = jnp.arange(iter + 1)
                            avgE = jnp.array(avgE)
                            stdE = jnp.array(stdE)
                            maxE = jnp.array(maxE)
                            
                            plt.figure(figsize=(6,4), dpi=300)
                            plt.plot(iters, avgE, marker='o')
                            plt.fill_between(iters, jnp.maximum(avgE-stdE, 10**(-20)), avgE+stdE, alpha=0.3)
                            plt.plot(maxE, marker='o')
                            plt.axhline(self.tol, color='k', linestyle='dashed')
                            plt.yscale('log')
                            plt.grid(alpha=0.5)
                            plt.savefig(f"figures/{self.output}_{chan_label}_{Elab}_training.png", format="png")
                            plt.close()

                            break
                            
                        # add validation point with largest error
                        LECs_train = jnp.concatenate((LECs_train, LECs_add[jnp.newaxis,:]), axis=0)
                        Tqs_train = jnp.concatenate((Tqs_train, jnp.transpose(exact_Tq)[:,jnp.newaxis]), axis=1)

            self.emulator['greedy GROM'] = {'Xqb': Xqb_stored, 'XVGXobb': XVGXobb_stored, 'XVob': XVob_stored}
            print("\nGreedy GROM emulator stored with label 'greedy GROM'.")
    
    def train_greedy_LSPG(self, LECs_lbd, LECs_ubd, Ninit=1, Nval=1000, Nmax=20, o1=0, o2=1, Ngrid=100):
    
        # initial key
        key = random.PRNGKey(self.seed)
    
        # initial training points are the corners of the boundary
        if Ninit is None:
            io = jnp.arange(self.pot.No)
            LECs_init = jnp.tile(LECs_lbd[jnp.newaxis,:], (self.pot.No, 1))
            LECs_init = LECs_init.at[io,io].set(LECs_ubd)
            
        # or initial training points are randomly chosen
        else:
            key, key_in = random.split(key)
            LECs_init = latin_hypercube(key_in, Ninit, self.pot.No, minvals=LECs_lbd, maxvals=LECs_ubd)

        print("LECs_init = ", LECs_init.shape)
        
        # solve high fidelity model at all training points
        Tsckq_init, Tscckq_init = self.solve(LECs_init)
        
        # store this emulator as a pytree
        Xqb_stored = {c: {str(Elab): [] for Elab in self.Elab} for c in self.chan.all_spect_not}
        XVGXobb_stored = {c: {str(Elab): [] for Elab in self.Elab} for c in self.chan.all_spect_not}
        XVob_stored = {c: {str(Elab): [] for Elab in self.Elab} for c in self.chan.all_spect_not}
        
        # train single channel emulator
        if self.pot.compute_single:
            for c, chan_label in enumerate(self.chan.single_spect_not):
                for k, Elab in enumerate(self.Elab):
                    
                    print("\n" + "-"*80)
                    print(f"Training greedy {ROM} ROM for {chan_label} channel at Elab = {Elab} MeV... ")
                    
                    # select initial training data
                    Tqs_train = jnp.transpose(Tsckq_init[:,c,k,:])
                    #print("avg T = ", jnp.mean(Tqs_train))
                    
                    # select corresponding LECs
                    LECs_train = LECs_init
                    
                    # store errors during training
                    avgE = []
                    stdE = []
                    maxE = []
                    
                    
                    for iter in range(Nmax):
                    
                        print(f"\nIteration = {iter}")
                        
                        print("Orthogonalizing, truncating, and projecting... ", end="")
                        ti = time.time()
                        
                        # matrix to orthogonalize
                        X
                        AXoqs = jnp.einsum('oij,', self.Vockqq[:,c,k,:,:], )
                        XVGXobb = jnp.einsum('ia,oij,j,jb->oab',
                                          jnp.conjugate(Xqb), self.Vockqq[:,c,k,:,:], self.Gkq[k,:], Xqb)
                        
                        # orthogonalize
                        Uqs, Ss, Vss = jnp.linalg.svd(Xqs, full_matrices=False)
                        
                        # truncate basis
                        index = int(jnp.argmax(Ss/Ss[0] <= self.tol))
                        Nbasis = index + 1 if index > 0 else Ss.shape[0]
                        Xqb = Uqs[:,:Nbasis]
                        #print("index = ", index)
                        #print("Nbasis = ", Nbasis)
                        #print("S/S0 = ", Ss[:Nbasis]/Ss[0])
                        
                        # project
                        XVGXobb = jnp.einsum('ia,oij,j,jb->oab',
                                              jnp.conjugate(Xqb), self.Vockqq[:,c,k,:,:], self.Gkq[k,:], Xqb)
                        XVob = jnp.einsum('ib,oi->ob', jnp.conjugate(Xqb), self.Vockqq[:,c,k,:,0])
                        
                        tf = time.time()
                        print(f"Done in {tf-ti:.3f} sec.")
                        
                        print("Validating... ", end="")
                        ti = time.time()
                        
                        # sample validation points and add the training points
                        key, key_in = random.split(key)
                        LECs_val = latin_hypercube(key_in, Nval, self.pot.No, minvals=LECs_lbd, maxvals=LECs_ubd)
                        LECs_val = jnp.concatenate((LECs_val, LECs_train))
    
                        # solve low fidelity model at validation points
                        XVGXsbb = jnp.einsum('so,oab->sab', LECs_val, XVGXobb)
                        XVsb = jnp.einsum('so,ob->sb', LECs_val, XVob)
                        ib = jnp.arange(Nbasis)
                        XAXsbb = (-XVGXsbb).at[:,ib,ib].add(1.0)
                        Csb = jnp.linalg.solve(XAXsbb, XVsb[...,jnp.newaxis])[...,0]
                        
                        # reconstruct emulated solution
                        emulated_Tsq = jnp.einsum('sb,ib->si', Csb, Xqb)
                        
                        # estimate error at validation points
                        VGsqq = jnp.einsum('so,oij,j->sij', LECs_val, self.Vockqq[:,c,k,:,:], self.Gkq[k,:])
                        Vsq = jnp.einsum('so,oi->si', LECs_val, self.Vockqq[:,c,k,:,0])
                        iq = jnp.arange(self.Nq+1)
                        Asqq = (-VGsqq).at[:,iq,iq].add(1.0)
                        estimated_Es = jnp.linalg.norm(jnp.einsum('sij,sj->si', Asqq, emulated_Tsq) - Vsq, axis=1)
                        
                        tf = time.time()
                        print(f"Done in {tf-ti:.3f} sec.")
                        
                        print("Selecting new training point and calibrating error... ", end="")
                        ti = time.time()
                        
                        # select validation point with largest estimated error
                        s_maxE = jnp.argmax(estimated_Es)
                        LECs_add = LECs_val[s_maxE]
                        
                        # calculate exact solution at validation point with largest error
                        exact_Tq = jnp.linalg.solve(Asqq[s_maxE], Vsq[s_maxE])
                        
                        # calibrate estimated error
                        exact_E = jnp.linalg.norm(exact_Tq - emulated_Tsq[s_maxE])
                        calibration_ratio = exact_E / estimated_Es[s_maxE]
                        calibrated_Es = estimated_Es * calibration_ratio
                        
                        # store calibrated error statistics
                        avgE.append(jnp.mean(calibrated_Es))
                        stdE.append(jnp.std(calibrated_Es))
                        maxE.append(jnp.max(calibrated_Es))
                        
                        tf = time.time()
                        print(f"Done in {tf-ti:.3f} sec.")
                        print(f"Calibration ratio = ", calibration_ratio)
                        print("Avg calibrated validation error = ", avgE[-1])
                        print("Std calibrated validation error = ", stdE[-1])
                        print("Max calibrated validation error = ", maxE[-1])
                        
                        '''
                        # exact error
                        exact_Tsq = jnp.linalg.solve(Asqq, Vsq[...,jnp.newaxis])[...,0]
                        exact_Es = jnp.linalg.norm(exact_Tsq-emulated_Tsq, axis=1)
                        print("Max exact validation error = ", jnp.max(exact_Es))
                        print("Min exact validation error = ", jnp.min(exact_Es))
                        
                        # ratio of errors
                        ratio_Es = estimated_Es / exact_Es
                        print("avg ratio of errors = ", jnp.mean(ratio_Es))
                        print("std ratio of errors = ", jnp.std(ratio_Es))
                        '''

                        # plot so that white = tol, blue = less than tol, red = more than tol
                        x = jnp.linspace(LECs_lbd[o1], LECs_ubd[o1], Ngrid)
                        y = jnp.linspace(LECs_lbd[o2], LECs_ubd[o2], Ngrid)
                        grid_x, grid_y = jnp.meshgrid(x, y)
                        grid_z = griddata(LECs_val[:,[o1, o2]], calibrated_Es, (grid_x, grid_y), method='nearest', rescale=True)
                        
                        exponent = int(jnp.log10(self.tol))
                        diff = 7
                        levels = jnp.logspace(exponent-diff, exponent+diff, num=100)
                        grid_z = jnp.minimum(grid_z, 10**(exponent+diff))
                        grid_z = jnp.maximum(grid_z, 10**(exponent-diff))
                        grid_z = gaussian_filter(grid_z, sigma=0.5)
                        
                        plt.figure(figsize=(6, 6), dpi=300)
                        plt.contourf(grid_x, grid_y, grid_z, cmap='bwr', levels=levels, norm=mcolors.LogNorm(vmin=10**(exponent-diff), vmax=10**(exponent+diff)))
                        plt.scatter(LECs_train[:, o1], LECs_train[:, o2], color='limegreen', s=100, edgecolors='k', linewidth=0.3)
                        plt.scatter(LECs_add[o1], LECs_add[o2], color='cyan', s=100, edgecolors='k', linewidth=0.3)
                        plt.scatter(LECs_val[:Nval, o1], LECs_val[:Nval, o2], color='white', s=5, edgecolors='k', linewidth=0.3)
                        plt.savefig(f"figures/{self.output}_{chan_label}_{Elab}_err_o{o1}_o{o2}_{iter}.png", format="png")
                        plt.close()
                        
                        fig, ax = plt.subplots(2, 1, figsize=(6, 9), dpi=300)
                        ax[0].plot(self.q / self.hbarc, exact_Tq.real[1:] * self.hbarc**3 )
                        ax[0].plot(self.q / self.hbarc, emulated_Tsq[s_maxE,1:].real * self.hbarc**3 , linestyle='dashed')
                        ax[1].plot(self.q / self.hbarc, exact_Tq.imag[1:] * self.hbarc**3)
                        ax[1].plot(self.q / self.hbarc, emulated_Tsq[s_maxE,1:].imag * self.hbarc**3, linestyle='dashed')
                        ax[0].grid()
                        ax[1].grid()
                        ax[0].set_xlim(0, 15)
                        ax[0].set_xlim(0, 15)
                        plt.savefig(f"figures/{self.output}_{chan_label}_{Elab}_T_{iter}.png", format="png")
                        plt.close()
                            
                            
                        # store emulator for this channel and energy once converged
                        if jnp.max(calibrated_Es) < self.tol:
                        
                            Elab_label = str(Elab)
                        
                            Xqb_stored[chan_label][Elab_label] = Xqb
                            XVGXobb_stored[chan_label][Elab_label] = XVGXobb
                            XVob_stored[chan_label][Elab_label] = XVob
                            
                            iters = jnp.arange(iter + 1)
                            avgE = jnp.array(avgE)
                            stdE = jnp.array(stdE)
                            maxE = jnp.array(maxE)
                            
                            plt.figure(figsize=(6,4), dpi=300)
                            plt.plot(iters, avgE, marker='o')
                            plt.fill_between(iters, jnp.maximum(avgE-stdE, 10**(-20)), avgE+stdE, alpha=0.3)
                            plt.plot(maxE, marker='o')
                            plt.axhline(self.tol, color='k', linestyle='dashed')
                            plt.yscale('log')
                            plt.grid(alpha=0.5)
                            plt.savefig(f"figures/{self.output}_{chan_label}_{Elab}_training.png", format="png")
                            plt.close()

                            break
                            
                        # add validation point with largest error
                        LECs_train = jnp.concatenate((LECs_train, LECs_add[jnp.newaxis,:]), axis=0)
                        Tqs_train = jnp.concatenate((Tqs_train, jnp.transpose(exact_Tq)[:,jnp.newaxis]), axis=1)

            self.emulator['greedy GROM'] = {'Xqb': Xqb_stored, 'XVGXobb': XVGXobb_stored, 'XVob': XVob_stored}
            print("\nGreedy GROM emulator stored with label 'greedy GROM'.")
    
    def emulate(self, emulator_label):
    
        #print(self.emulator[emulator_label])
        pass

            
    '''
    def train_greedy_GROM(self, LECs_lbd, LECs_ubd, Nval=100, Nmax=5, Nmesh=100, seed=213):


        
    
    def train_POD(self, LECs_train):
    
        print("Building POD emulator...")
        

        
        # Compute half-shell T for single channels:
        if self.pot.compute_single:
        
            # set up potential operator matrices
            ti_S = time.time()
            Volkqq = jnp.zeros((self.pot.No, self.pot.Nl, self.Nk, self.Nq+1, self.Nq+1), dtype=jnp.complex128)
            Volkqq = Volkqq.at[:,:,:,1:,1:].set( jnp.tile(Volqq[:,:,jnp.newaxis,:,:], (1,1,self.Nk,1,1)) )
            Volkqq = Volkqq.at[:,:,:,1:,0].set( Volqk )
            Volkqq = Volkqq.at[:,:,:,0,1:].set( Volqk )
            Volkqq = Volkqq.at[:,:,:,0,0].set( Volk )
            
            # set up full linear system
            iq = jnp.arange(self.Nq+1)
            VGolkqq = -jnp.einsum('olkij,kj->olkij', Volkqq, self.Gkq)
            VGslkqq = jnp.einsum('so,olkij->slkij', LECs_train, VGolkqq)
            Aslkqq = VGslkqq.at[:,:,:,iq,iq].add(1.0)
            print("Aslkqq = ", Aslkqq.shape)
            
            # solve for half-shell T matrix for all training params
            Vslkq = jnp.einsum('so,olki->slki', LECs_train, Volkqq[:,:,:,:,0])
            Tslkq = jnp.linalg.solve(Aslkqq, Vslkq[..., None])[..., 0]
            print("Tslkq = ", Tslkq.shape)
            
            # orthogonalize and truncate snapshots using SVD
            U, S, Vh = jnp.linalg.svd(jnp.transpose(Tslkq, (1,2,3,0)), full_matrices=False)
            S_norm = S/S[:,:,0][:,:,jnp.newaxis]
            print("U = ", U.shape)
            print("S = ", S.shape)
            print("Vh = ", Vh.shape)
            
            indices = jnp.argmax(S_norm < self.tol, axis=2)
            self.Ns_single = jnp.max(indices)+1
            print("indices = ", indices.shape)

            # orthogonalize by selecting subset of left singular vectors
            print(S_norm[:,:,:self.Ns_single])
            X = U[:,:,:,:self.Ns_single]
            
            #for i in range(self.Ns_single):
            #    plt.plot(X[0,0,1:,i].real)
            #plt.show()
            
            # I checked that the below is the identity
            #XhX = jnp.einsum('lkqi,lkqj->lkij', jnp.conjugate(X), X)
            #print(XhX)
            
            #for i in range(XhX.shape[0]):
            #    for j in range(XhX.shape[1]):
            #        print(XhX[i,j])
            #        plt.imshow(XhX[i,j].real)
            #        plt.show()
            
            # store projected VG and V
            self.projected_VGolkqq = jnp.einsum('lkqi,olkqr,lkrj->olkij', jnp.conjugate(X), VGolkqq, X)
            self.projected_Volkq = jnp.einsum('lkaq,olka->olkq', jnp.conjugate(X), Volkqq[:,:,:,:,0])
            self.LECs_train = LECs_train
            
            tf_S = time.time()
            print(f"Time building emulator with {self.Ns_single}/{LECs_train.shape[0]} snapshots: {tf_V - ti_V + tf_S - ti_S:3f} s.")
            
        
            
    def test_POD(self, LECs_test):
        
        # solve smaller linear system
        iq = jnp.arange(self.Ns_single)
        VGolkqq = -jnp.einsum('olkij,kj->olkij', Volkqq, self.Gkq)
        VGslkqq = jnp.einsum('so,olkij->slkij', LECs_train, VGolkqq)
        Aslkqq = VGslkqq.at[:,:,:,iq,iq].add(1.0)
        print("Aslkqq = ", Aslkqq.shape)
        
        # solve for half-shell T matrix for all training params
        Vslkq = jnp.einsum('so,olki->slki', LECs_train, Volkqq[:,:,:,:,0])
        Tslkq = jnp.linalg.solve(Aslkqq, Vslkq[..., None])[..., 0]
        print("Tslkq = ", Tslkq.shape)

    '''

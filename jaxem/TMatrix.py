import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
from jax import random, grad, jit, vmap
from scipy.interpolate import griddata, RBFInterpolator
from functools import partial
import matplotlib.pyplot as plt
from .Tools import latin_hypercube, gram_schmidt_insert, householder_insert, modified_gram_schmidt_insert
from .Potential import Potential
from .Map import Map
import matplotlib.pyplot as plt
import time
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

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
        self.tol = config.tol
        self.seed = config.seed
        self.factor = config.factor
        
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
        self.logger.info(f"Logger initialized and writing to file: output/{self.output}.txt")
        
        # setup TMatrix object
        self.log_block_start("SETUP")
        
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
        self.Gkq *= config.factor # different normalization conventions
        self.log_end()

                
        if self.pot.compute_single:
        
            # precalculate potential operators for all channels and energies
            self.log_category = "single"
            self.log_start("precalculating potential operators qq")
            Vocqq = self.pot.Voc(self.q)
            self.log_end()
            
            self.log_start("precalculating potential operators kq")
            Vockq = self.pot.Voc(self.k, self.q)
            self.log_end()
            
            self.log_start("precalculating potential operators k")
            Vock = self.pot.Voc(self.k, diag=True)
            self.log_end()
            
            self.log_start("filling potential operators")
            self.Vockqq = jnp.zeros((self.pot.No, self.chan.Nsingle, self.Nk, self.Nq+1, self.Nq+1), dtype=jnp.complex128)
            self.Vockqq = self.Vockqq.at[:,:,:,1:,1:].set( jnp.tile(Vocqq[:,:,jnp.newaxis,:,:], (1,1,self.Nk,1,1)) )
            self.Vockqq = self.Vockqq.at[:,:,:,1:,0].set( Vockq )
            self.Vockqq = self.Vockqq.at[:,:,:,0,1:].set( Vockq )
            self.Vockqq = self.Vockqq.at[:,:,:,0,0].set( Vock )
            self.log_end()
            
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


        if self.pot.compute_coupled:
        
            # precalculate potential operators for all channels and energies
            self.log_category = "coupled"
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
        
        
        # pytrees for storing emulators
        self.emulator = {'greedy': {}, 'POD': {}}
        
        self.log_block_end()
            
            
    def solve(self, LECs_samples):
    
        LECs_samples = jnp.reshape(LECs_samples, (-1, self.pot.No))
        Nsamples = LECs_samples.shape[0]
        
        if self.pot.compute_single:
        
            self.log_category = "single"
            self.log_start(f"solving high-fidelity model for {Nsamples} samples")
            
            # compute potential matrices for all LEC combinations, single channels, and energies
            Vsckqq = jnp.einsum('so,ockij->sckij', LECs_samples, self.Vockqq) # MeV^2
            
            # set up linear system
            iq = jnp.arange(self.Nq+1)
            Asckqq = -jnp.einsum('sckij,kj->sckij', Vsckqq, self.Gkq) # 1
            Asckqq = Asckqq.at[:,:,:,iq,iq].add(1.0)
            
            # solve for half-shell T matrix for all LECs, single channels, and energies
            Tsckq = jnp.linalg.solve(Asckqq, Vsckqq[:,:,:,:,0][..., None])[..., 0] # MeV^2
            
            self.log_end()
            
        else:
        
            Tsckq = None
            
            
        if self.pot.compute_coupled:
        
            self.log_category = "coupled"
            self.log_start(f"solving high-fidelity model for {Nsamples} samples")
    
            # compute potential matrices for all LEC combinations, coupled channels, and energies
            # units: fm^2
            Vscckqq = jnp.einsum('so,ockij->sckij', LECs_samples, self.Vocckqq)
  
            # set up linear system
            iq = jnp.arange(2*self.Nq+2)
            Ascckqq = -jnp.einsum('sckij,kj->sckij', Vscckqq, jnp.tile(self.Gkq, (1,2)))
            Ascckqq = Ascckqq.at[:,:,:,iq,iq].add(1.0)

            # solve for half-shell T matrix
            Tscckq = jnp.linalg.solve(Ascckqq, Vscckqq[:,:,:,:,[0, self.Nq + 1]])
            Tscckq = jnp.reshape(Tscckq, (Nsamples, self.chan.Ncoupled, self.Nk, 2, self.Nq+1, 2))
            Tscckq = jnp.transpose(Tscckq, (0,1,2,3,5,4))

            self.log_end()
            
        else:
        
            Tscckq = None
        
        return Tsckq, Tscckq
        
        
    
    def train_POD_GROM(self, LECs_lbd, LECs_ubd, Ntrain=40, Ntest=1000):
    
        self.log_block_start("POD GROM")
    
        # sample training LECs
        self.log_category = "common"
        self.log_stage = "train POD GROM"
        self.log_start(f"sampling {Ntrain} training points")
        key = random.PRNGKey(self.seed)
        key, key_in = random.split(key)
        LECs_train = latin_hypercube(key_in, Ntrain, self.pot.No, minvals=LECs_lbd, maxvals=LECs_ubd)
        self.log_end()
        self.log_message(f"LECs_train shape = {LECs_train.shape}")
        
        # solve high fidelity model at all training points
        Tsckq_train, Tscckq_train = self.solve(LECs_train)
        
        # train single channel emulator
        if self.pot.compute_single:
            self.log_category = "single"
            for c, chan_label in enumerate(self.chan.single_spect_not):
                for k, Elab in enumerate(self.Elab):
                    
                    self.log_stage = "train POD GROM"
                    self.log_message(f"training {chan_label} channel at Elab = {Elab} MeV...")
                    self.log_start("orthogonalizing, truncating, and projecting")
                    
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
                    
                    self.log_end()
                    self.log_message(f"Nbasis = {Nbasis}")
                    
                    # sample testing points
                    self.log_stage = "test POD GROM"
                    self.log_start(f"sampling {Ntest} testing points")
                    key, key_in = random.split(key)
                    LECs_test = latin_hypercube(key_in, Ntest, self.pot.No, minvals=LECs_lbd, maxvals=LECs_ubd)
                    self.log_end()
                    self.log_message(f"LECs_test shape = {LECs_test.shape}")
                    
                    # solve low fidelity model at testing points
                    self.log_start("emulating at testing points")
                    XVGXsbb = jnp.einsum('so,oab->sab', LECs_test, XVGXobb)
                    XVsb = jnp.einsum('so,ob->sb', LECs_test, XVob)
                    ib = jnp.arange(Nbasis)
                    XAXsbb = (-XVGXsbb).at[:,ib,ib].add(1.0)
                    Csb = jnp.linalg.solve(XAXsbb, XVsb[...,jnp.newaxis])[...,0]
                    emulated_Tsq = jnp.einsum('sb,ib->si', Csb, Xqb)
                    self.log_end()
                    self.log_message(f"Csb shape = {Csb.shape}")
                    self.log_message(f"emulated Tsq shape = {emulated_Tsq.shape}")
                    
                    # estimate error at testing points
                    self.log_start("estimating error at testing points")
                    VGsqq = jnp.einsum('so,oij,j->sij', LECs_test, self.Vockqq[:,c,k,:,:], self.Gkq[k,:])
                    Vsq = jnp.einsum('so,oi->si', LECs_test, self.Vockqq[:,c,k,:,0])
                    iq = jnp.arange(self.Nq+1)
                    Asqq = (-VGsqq).at[:,iq,iq].add(1.0)
                    estimated_Es = jnp.linalg.norm(jnp.einsum('sij,sj->si', Asqq, emulated_Tsq) - Vsq, axis=1)
                    self.log_end()
                    self.log_message(f"max estimated error = {jnp.max(estimated_Es):.8e}")
                    
                    
                    self.log_start("calibrating error at testing points")

                    # select testing point with largest estimated error
                    s_maxE = jnp.argmax(estimated_Es)
                    LECs_add = LECs_test[s_maxE]
                    
                    # calculate exact solution at testing point with largest error
                    exact_Tq = jnp.linalg.solve(Asqq[s_maxE], Vsq[s_maxE])
                    
                    # calibrate estimated error
                    exact_E = jnp.linalg.norm(exact_Tq - emulated_Tsq[s_maxE])
                    calibration_ratio = exact_E / estimated_Es[s_maxE]
                    calibrated_Es = estimated_Es * calibration_ratio
                    
                    self.log_end()
                    self.log_message(f"calibration ratio = {calibration_ratio:.8f}")
                    self.log_message(f"avg calibrated testing error = {jnp.mean(calibrated_Es):.8e}")
                    self.log_message(f"std calibrated testing error = {jnp.std(calibrated_Es):.8e}")
                    self.log_message(f"max calibrated testing error = {jnp.max(calibrated_Es):.8e}")

        self.log_block_end()




    def train_greedy_GROM(self, LECs_lbd, LECs_ubd, Ninit=2, Nval=1000, Nmax=20, o1=0, o2=1, Ngrid=100):
    
        self.log_block_start("GREEDY GROM")

        # sample initial training points
        self.log_category = "common"
        self.log_stage = "train greedy GROM"
        self.log_start(f"sampling {Ninit} training points")
        key = random.PRNGKey(self.seed)
        key, key_in = random.split(key)
        LECs_init = latin_hypercube(key_in, Ninit, self.pot.No, minvals=LECs_lbd, maxvals=LECs_ubd)
        self.log_end()
        self.log_message(f"LECs_train shape = {LECs_init.shape}")

        # solve high fidelity model at all training points
        Tsckq_init, Tscckq_init = self.solve(LECs_init)
        
        # store this emulator as a pytree
        #Xqb_stored = {c: {str(Elab): [] for Elab in self.Elab} for c in self.chan.all_spect_not}
        #XVGXobb_stored = {c: {str(Elab): [] for Elab in self.Elab} for c in self.chan.all_spect_not}
        #XVob_stored = {c: {str(Elab): [] for Elab in self.Elab} for c in self.chan.all_spect_not}
        

        # train single channel emulator
        if self.pot.compute_single:
            self.log_category = "single"
            for c, chan_label in enumerate(self.chan.single_spect_not):
                for k, Elab in enumerate(self.Elab):
                    
                    self.log_stage = "train greedy GROM"
                    self.log_message(f"training {chan_label} channel at Elab = {Elab} MeV...")
                    self.log_start(f"orthogonalizing {Ninit} initial training points")
                    Tqb_train = jnp.transpose(Tsckq_init[:,c,k,:])
                    LECs_train = LECs_init
                    Ntrain = Ninit
                    Xqb, _ = jnp.linalg.qr(Tqb_train)
                    self.log_end()
                    
                    # store errors during training
                    avgE = []
                    stdE = []
                    maxE = []

                    for iter in range(Nmax):
                    
                        self.log_stage = "train greedy GROM"
                        self.log_message(f"greedy iteration = {iter}")
                    
                        # project
                        self.log_start("projecting")
                        XVGXobb = jnp.einsum('ia,oij,j,jb->oab',
                                              jnp.conjugate(Xqb), self.Vockqq[:,c,k,:,:], self.Gkq[k,:], Xqb)
                        XVob = jnp.einsum('ib,oi->ob', jnp.conjugate(Xqb), self.Vockqq[:,c,k,:,0])
                        self.log_end()
                        
                        # sample validation points and add the training points
                        self.log_stage = "val greedy GROM"
                        self.log_start(f"sampling {Nval} validation points")
                        key, key_in = random.split(key)
                        LECs_val = latin_hypercube(key_in, Nval, self.pot.No, minvals=LECs_lbd, maxvals=LECs_ubd)
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
                        
                        self.log_start(f"calibrating error at validation points")
                        
                        # select validation point with largest estimated error
                        s_maxE = jnp.argmax(estimated_Es)
                        LECs_add = LECs_val[s_maxE]
                        
                        # calculate exact solution at validation point with largest error
                        exact_Tq = jnp.linalg.solve(Asqq[s_maxE], Vsq[s_maxE])
                        
                        # calibrate estimated error
                        exact_E = jnp.linalg.norm(exact_Tq - emulated_Tsq[s_maxE])
                        calibration_ratio = exact_E / estimated_Es[s_maxE]
                        calibrated_Es = estimated_Es * calibration_ratio

                        self.log_end()
                        self.log_message(f"calibration ratio = {calibration_ratio:.8f}")
                        self.log_message(f"avg calibrated testing error = {jnp.mean(calibrated_Es):.8e}")
                        self.log_message(f"std calibrated testing error = {jnp.std(calibrated_Es):.8e}")
                        self.log_message(f"max calibrated testing error = {jnp.max(calibrated_Es):.8e}")
                        
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
                        plt.savefig(f"figures/{self.output}_{chan_label}_{Elab}_err_o{o1}_o{o2}_{iter:02d}.png", format="png")
                        #plt.show()
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
                        '''
                        
                        
                        self.log_start(f"evaluating calibrated error")
                            
                        # store emulator for this channel and energy once converged
                        if jnp.max(calibrated_Es) < self.tol:
                        
                            Elab_label = str(Elab)
                        
                            #Xqb_stored[chan_label][Elab_label] = Xqb
                            #XVGXobb_stored[chan_label][Elab_label] = XVGXobb
                            #XVob_stored[chan_label][Elab_label] = XVob
                            
                            iters = jnp.arange(iter + 1)
                            avgE = jnp.array(avgE)
                            stdE = jnp.array(stdE)
                            maxE = jnp.array(maxE)
                            
                            '''
                            plt.figure(figsize=(6,4), dpi=300)
                            plt.plot(iters, avgE, marker='o')
                            plt.fill_between(iters, jnp.maximum(avgE-stdE, 10**(-20)), avgE+stdE, alpha=0.3)
                            plt.plot(maxE, marker='o')
                            plt.axhline(self.tol, color='k', linestyle='dashed')
                            plt.yscale('log')
                            plt.grid(alpha=0.5)
                            plt.savefig(f"figures/{self.output}_{chan_label}_{Elab}_training.png", format="png")
                            plt.close()
                            '''
                            
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
                        print(Tqb_train.shape)
                        Xqb, _ = jnp.linalg.qr(Tqb_train)
                        
                        self.log_end()
                        self.log_message(f"added validation point with max error to training points")
                        


            #self.emulator['greedy GROM'] = {'Xqb': Xqb_stored, 'XVGXobb': XVGXobb_stored, 'XVob': XVob_stored}
            #print("\nGreedy GROM emulator stored with label 'greedy GROM'.")
            
            
        self.log_block_end()
        

    def phase_shifts(self, T_single=None, T_coupled=None):
    
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
    
    

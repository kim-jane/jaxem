import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import Dict, Tuple, Optional


class Channels:

    def __init__(
        self,
        isospin_channel: str = "np",
        Jmax: int = 1
    ):
        self.isospin_channel = isospin_channel
        self.Jmax = Jmax
                
        # determine twice isospin projection
        if self.isospin_channel == 'nn':
            Tz = -1
            
        elif self.isospin_channel == 'pp':
            Tz = 1
            
        elif self.isospin_channel == 'pn' or self.isospin_channel == 'np':
            Tz = 0
            
        else:
            raise ValueError(f"Isospin channel {isospin_channel} unknown.")
        
        # list of tuples (J, S, T, Tz, L)
        single_channels = []
        
        # corresponding spectroscopic notation (2S+1)(L)(J)
        single_labels = []
        
        # list of tuples (J, S, T, Tz, L, L')
        coupled_channels = []
        
        # corresponding spectroscopic notation (2S+1)(J-1)(J) - (2S+1)(J+1)(J)
        coupled_labels = []
        
        spect_not = ['S', 'P', 'D', 'F', 'G', 'H', 'I', 'K',
                     'L', 'M', 'N', 'O', 'Q', 'R', 'T', 'U']
                       
        # flag all allowed self.isospin_channels
        for J in range(Jmax+1):
            for S in range(2):
                for L in range(J+2):
                    for T in [0, 1] if Tz == 0 else [1]:
                        if (abs(L-S) <= J <= L+S) and (L + S + T) % 2 == 1:
                            single_channels.append((J, S, T, Tz, L))

                            single_labels.append(f"{2*S+1}{spect_not[L]}{J}")
                    

        # extract coupled self.isospin_channels
        indices_to_remove = []
        for i, j in zip(*jnp.triu_indices(len(single_channels), k=1)):
            J1, S1, T1, Tz1, L1 = single_channels[i]
            J2, S2, T2, Tz2, L2 = single_channels[j]
            spect_not1 = single_labels[i]
            spect_not2 = single_labels[j]
            if (J1 == J2) and (S1 == S2) and (T1 == T2) and (abs(L1 - L2) == 2):
                coupled_channels.append((J1, S1, T1, Tz1, L1, L2))
                coupled_labels.append(f"{spect_not1}-{spect_not2}")
                indices_to_remove.extend((i,j))
                
        # remove the coupled self.isospin_channels from the single self.isospin_channel list
        for i in sorted(indices_to_remove, reverse=True):
            del single_channels[i]
            del single_labels[i]
            
            
        self.single_quantum_nums = jnp.array(single_channels)
        self.coupled_quantum_nums = jnp.array(coupled_channels)
        
        self.n_single = self.single_quantum_nums.shape[0]
        self.n_coupled = self.coupled_quantum_nums.shape[0]
        
        self.single_labels = single_labels
        self.coupled_labels = coupled_labels
        
        self.print_single_channels()
        self.print_coupled_channels()


    def print_single_channels(self):
    
        if self.n_single > 0:
            print("Single channels:")
            for i in range(self.n_single):
                J, S, T, Tz, L = self.single_quantum_nums[i]
                label = self.single_labels[i]
                print(f"\t{i} --- {self.isospin_channel}, {label} --- J = {J}, S = {S}, T = {T}, L = {L}")

    
        
    def print_coupled_channels(self):
    
        if self.n_coupled > 0:
            print("Coupled channels:")
            for i in range(self.n_coupled):
                J, S, T, Tz, L1, L2 = self.coupled_quantum_nums[i]
                label = self.coupled_labels[i]
                print(f"\t{i} --- {self.isospin_channel}, {label} --- J = {J}, S = {S}, T = {T}, L = {L1}, L' = {L2}")

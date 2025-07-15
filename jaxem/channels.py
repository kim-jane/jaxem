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
        single_spect_not = []
        
        # list of tuples (J, S, T, Tz, L, L')
        coupled_channels = []
        
        # corresponding spectroscopic notation (2S+1)(J-1)(J) - (2S+1)(J+1)(J)
        coupled_spect_not = []
        
        spect_not = ['S', 'P', 'D', 'F', 'G', 'H', 'I', 'K',
                     'L', 'M', 'N', 'O', 'Q', 'R', 'T', 'U']
                       
        # flag all allowed self.isospin_channels
        for J in range(Jmax+1):
            for S in range(2):
                for L in range(J+2):
                    for T in [0, 1] if Tz == 0 else [1]:
                        if (abs(L-S) <= J <= L+S) and (L + S + T) % 2 == 1:
                            single_channels.append((J, S, T, Tz, L))
                            single_spect_not.append(f"{2*S+1}{spect_not[L]}{J}")
                    

        # extract coupled self.isospin_channels
        indices_to_remove = []
        for i, j in zip(*jnp.triu_indices(len(single_channels), k=1)):
            J1, S1, T1, Tz1, L1 = single_channels[i]
            J2, S2, T2, Tz2, L2 = single_channels[j]
            spect_not1 = single_spect_not[i]
            spect_not2 = single_spect_not[j]
            if (J1 == J2) and (S1 == S2) and (T1 == T2) and (abs(L1 - L2) == 2):
                coupled_channels.append((J1, S1, T1, Tz1, L1, L2))
                coupled_spect_not.append(f"{spect_not1}-{spect_not2}")
                indices_to_remove.extend((i,j))
                
        # remove the coupled self.isospin_channels from the single self.isospin_channel list
        for i in sorted(indices_to_remove, reverse=True):
            del single_channels[i]
            del single_spect_not[i]

        # store in nested dictionaries
        self.single = {
            label: {
                "J": J, "S": S, "T": T, "Tz": Tz, "L": L
            }
            for label, (J, S, T, Tz, L) in zip(single_spect_not, single_channels)
        }
        
        self.coupled = {
            label: {
                "J": J, "S": S, "T": T, "Tz": Tz, "L1": L1, "L2": L2
            }
            for label, (J, S, T, Tz, L1, L2) in zip(coupled_spect_not, coupled_channels)
        }

        # print
        self.print_single_channels()
        self.print_coupled_channels()
        


    def print_single_channels(self):

        print("Single channels:")
        
        for spect_not, quantum_numbers in self.single.items():
            J = quantum_numbers["J"]
            S = quantum_numbers["S"]
            T = quantum_numbers["T"]
            Tz = quantum_numbers["Tz"]
            L = quantum_numbers["L"]
            if Tz == 1:
                print(f"\t{spect_not}, pp -> J = {J}, S = {S}, T = {T}, L = {L}")
            elif Tz == -1:
                print(f"\t{spect_not}, nn -> J = {J}, S = {S}, T = {T}, L = {L}")
            else:
                print(f"\t{spect_not}, np -> J = {J}, S = {S}, T = {T}, L = {L}")
    
        
    def print_coupled_channels(self):

        print("Coupled channels:")
        for spect_not, quantum_numbers in self.coupled.items():
            J = quantum_numbers["J"]
            S = quantum_numbers["S"]
            T = quantum_numbers["T"]
            Tz = quantum_numbers["Tz"]
            L1 = quantum_numbers["L1"]
            L2 = quantum_numbers["L2"]
            
            if Tz == 1:
                print(f"\t{spect_not}, pp -> J = {J}, S = {S}, T = {T}, L = {L1}, L' = {L2}")
            elif Tz == -1:
                print(f"\t{spect_not}, nn -> J = {J}, S = {S}, T = {T}, L = {L1}, L' = {L2}")
            else:
                print(f"\t{spect_not}, np -> J = {J}, S = {S}, T = {T}, L = {L1}, L' = {L2}")


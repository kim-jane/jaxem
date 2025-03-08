import jax.numpy as jnp


class Channel:

    def __init__(self, config):
    
        channel = config.channel
        Jmax = config.Jmax
                
        # determine twice isospin projection
        if channel == 'nn':
            Tz = -1
            
        elif channel == 'pp':
            Tz = 1
            
        elif channel == 'pn' or channel == 'np':
            Tz = 0
            
        else:
            raise ValueError(f"Channel {channel} unknown.")
        
        # list of tuples (J, S, T, Tz, L)
        self.single_channels = []
        
        # corresponding spectroscopic notation (2S+1)(L)(J)
        self.single_spect_not = []
        
        # list of tuples (J, S, T, Tz, L, L')
        self.coupled_channels = []
        
        # corresponding spectroscopic notation (2S+1)(J-1)(J) - (2S+1)(J+1)(J)
        self.coupled_spect_not = []
        
        spect_not = ['S', 'P', 'D', 'F', 'G', 'H', 'I', 'K',
                     'L', 'M', 'N', 'O', 'Q', 'R', 'T', 'U']
                       
        # flag all allowed channels
        for J in range(Jmax+1):
            for S in range(2):
                for L in range(J+2):
                    for T in [0, 1] if Tz == 0 else [1]:
                        if (abs(L-S) <= J <= L+S) and (L + S + T) % 2 == 1:
                            self.single_channels.append((J, S, T, Tz, L))
                            self.single_spect_not.append(f"{2*S+1}{spect_not[L]}{J}")
                    

        # extract coupled channels
        indices_to_remove = []
        for i, j in zip(*jnp.triu_indices(len(self.single_channels), k=1)):
            J1, S1, T1, Tz1, L1 = self.single_channels[i]
            J2, S2, T2, Tz2, L2 = self.single_channels[j]
            spect_not1 = self.single_spect_not[i]
            spect_not2 = self.single_spect_not[j]
            if (J1 == J2) and (S1 == S2) and (T1 == T2) and (abs(L1 - L2) == 2):
                self.coupled_channels.append((J1, S1, T1, Tz1, L1, L2))
                self.coupled_spect_not.append(f"{spect_not1}-{spect_not2}")
                indices_to_remove.extend((i,j))
                
        # remove the coupled channels from the single channel list
        for i in sorted(indices_to_remove, reverse=True):
            del self.single_channels[i]
            del self.single_spect_not[i]
        
        
        self.Nsingle = len(self.single_channels)
        self.Ncoupled = len(self.coupled_channels)

        # store in JAX arrays
        self.single_channels = jnp.array(self.single_channels, dtype=jnp.int16)
        self.coupled_channels = jnp.array(self.coupled_channels, dtype=jnp.int16)

        # transpose for easy unpacking
        self.single_channels = jnp.transpose(self.single_channels)
        self.coupled_channels = jnp.transpose(self.coupled_channels)
        


    def print_single_channels(self):

        print("Single channels:")
        for i in range(self.Nsingle):
            J, S, T, Tz, L = self.single_channels[:,i]
            spect_not = self.single_spect_not[i]
            if Tz == 1:
                print(f"\t{i} -> {spect_not}, pp, J = {J}, S = {S}, T = {T}, L = {L}")
            elif Tz == -1:
                print(f"\t{i} -> {spect_not}, nn, J = {J}, S = {S}, T = {T}, L = {L}")
            else:
                print(f"\t{i} -> {spect_not}, np, J = {J}, S = {S}, T = {T}, L = {L}")
    
        
    def print_coupled_channels(self):

        print("Coupled channels:")
        for i in range(self.Ncoupled):
            J, S, T, Tz, L1, L2 = self.coupled_channels[:,i]
            spect_not = self.coupled_spect_not[i]
            if Tz == 1:
                print(f"\t{i} -> {spect_not}, pp, J = {J}, S = {S}, T = {T}, L = {L1}, L' = {L2}")
            elif Tz == -1:
                print(f"\t{i} -> {spect_not}, nn, J = {J}, S = {S}, T = {T}, L = {L1}, L' = {L2}")
            else:
                print(f"\t{i} -> {spect_not}, np, J = {J}, S = {S}, T = {T}, L = {L1}, L' = {L2}")

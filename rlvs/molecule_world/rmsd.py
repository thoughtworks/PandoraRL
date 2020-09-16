# Calculates Root-mean-square deviation (RMSD) between structure A and B in pdb format
# Then calculates the reward function for that pose
# No rotation or translation done here
# If using within RL code, may skip the reading of pdb (get_co-ordinates)
# if data is already in vector format
## Based on https://github.com/charnley/rmsd

import numpy as np

class RMSD:
    def __init__(self, reference_molecule):
        self.reference_molecule = reference_molecule
        
    def __call__(self, target_molecule):
        if self.reference_molecule.num_atoms != target_molecule.num_atoms:
            raise Exception("error: Structures not same size")
        
        if np.count_nonzero(self.reference_molecule.atoms != target_molecule.atoms):
            raise Exception("error: Atoms are not in the same order")

        r_cent = self.centroid(self.reference_molecule.coords)
        t_cent = self.centroid(target_molecule.coords)

        r_coord = self.reference_molecule.coords - r_cent
        t_coord = target_molecule.coords - t_cent
    
        return self.__rmsd(r_coord, t_coord)

    @staticmethod
    def __rmsd(V, W):
        N = len(V)
        return np.sqrt(((V-W)**2).sum() / N)

    @staticmethod
    def centroid(X):
       # Calculates centroid, i.e. mean position of all the points in
       # all of the coordinate directions, from a vectorset X.
        C = X.mean(axis=0)
        return C

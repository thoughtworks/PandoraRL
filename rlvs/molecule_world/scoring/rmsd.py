# Calculates Root-mean-square deviation (RMSD) between structure A and B in pdb format
# Then calculates the reward function for that pose
# No rotation or translation done here
# If using within RL code, may skip the reading of pdb (get_co-ordinates)
# if data is already in vector format
# Based on https://github.com/charnley/rmsd

import numpy as np


class RMSD:
    def __init__(self, reference_molecule):
        self.reference_molecule = reference_molecule

    def __call__(self, target_molecule):
        if self.reference_molecule.n_atoms != target_molecule.n_atoms:
            raise Exception("error: Structures not same size")

        r_cent = 0  # self.centroid(self.reference_molecule.get_coords())
        t_cent = 0  # self.centroid(target_molecule.get_coords())

        r_coord = self.reference_molecule.get_coords() - r_cent
        t_coord = target_molecule.get_coords() - t_cent

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

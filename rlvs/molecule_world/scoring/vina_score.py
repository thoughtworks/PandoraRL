import numpy as np
from rlvs.constants import H, C, O, N, S, Features, Vina, HydrogenBondPair
from rlvs.agents.utils import filter_by_distance, to_numpy

class VinaScore:
    W1 = -0.0356
    W2 = -0.00516
    W3 = 0.840
    W4 = -0.0351
    W5 = -0.587
    W_ROT = 0.0585

    def __init__(self, protein, ligand):
        self.protein = protein
        self.ligand = ligand

    def gauss1(self, surface_dist):
        o1 = 0
        s1 = 0.5
        return np.exp(-((surface_dist - o1)/s1)**2)

    
    def gauss2(self, surface_dist):
        o1 = 3
        s1 = 2
        
        return np.exp(-((surface_dist - o1)/s1)**2)
    

    def repulsion(self, surface_dist):
        rep = np.zeros(surface_dist.shape)
        rep = np.where(surface_dist < 0, surface_dist ** 2, 0)

        return rep

    def hydrophobic(self, surface_dist, feature_lists):
        p1 = 0.5
        p2 = 1.5

        hyph = np.zeros(surface_dist.shape)

        # TODO: add constant indexes for hydrophobic, and carbon, hydrogen

        hyph[
            (feature_lists[:, 0, C.feature_list_index] == 1) & # carbon in protein
            (feature_lists[:, 1, C.feature_list_index] == 1) & # carbon in ligand
            (feature_lists[:, 0, Features.HYDROPHOBIC]== 1) & # atom is hydrophobic in protein
            (feature_lists[:, 1, Features.HYDROPHOBIC] == 1) &  # atom is hydrophobic in ligand
            (surface_dist < p1)
        ] = 1

        hyph[
            (feature_lists[:, 0, C.feature_list_index] == 1) & # carbon in protein
            (feature_lists[:, 1, C.feature_list_index] == 1) & # carbon in ligand
            (feature_lists[:, 0, Features.HYDROPHOBIC]== 1) & # atom is hydrophobic in protein
            (feature_lists[:, 1, Features.HYDROPHOBIC] == 1) &  # atom is hydrophobic in ligand
            (surface_dist >= p1) & (surface_dist < p2)
        ] = (p2 - surface_dist[
            (feature_lists[:, 0, C.feature_list_index] == 1) & # carbon in protein
            (feature_lists[:, 1, C.feature_list_index] == 1) & # carbon in ligand
            (feature_lists[:, 0, Features.HYDROPHOBIC]== 1) & # atom is hydrophobic in protein
            (feature_lists[:, 1, Features.HYDROPHOBIC] == 1) &  # atom is hydrophobic in ligand
            (surface_dist >= p1) & (surface_dist < p2)
        ])/(p2 - p1)

        return hyph
    

    def hydrogenbond(self, surface_dist, possible_hydrogen_bonds):
        h1 = -0.7

        hybnd = np.zeros(surface_dist.shape)

        valid_hydrogen_bond_idx = [
            h_bond_pair.idx for h_bond_pair in possible_hydrogen_bonds
             for h_bond in h_bond_pair.donor.hydrogens
             if (
                     h_bond.distance < Vina.DONOR_HYDROGEN_DISTANCE
                     and h_bond.saperation(
                         h_bond_pair.acceptor, H
                     ) < Vina.ACCEPTOR_HYDROGEN_DISTANCE
                     and h_bond.angle(
                         atom1=h_bond_pair.acceptor,
                         named_atom2=H,
                         atom3=h_bond_pair.donor
                     ) >= Vina.HYDROGEN_BOND_ANGLE
             )
        ]

        hybnd[valid_hydrogen_bond_idx] = np.where(
            surface_dist[valid_hydrogen_bond_idx] <= h1, 1,
            np.where(
                (
                    surface_dist[valid_hydrogen_bond_idx] > h1
                ) & (
                    surface_dist[valid_hydrogen_bond_idx] <= 0
                ), -surface_dist[valid_hydrogen_bond_idx]/h1,
                0
            )
        )

        return hybnd

    def possible_hydrogen_bonds(self, valid_pairs):
        is_H_bond_functional_group = lambda atom1, atom2: (O == atom1 and (N == atom2 or O == atom2)) or\
            (N == atom1 and (O == atom2 or N == atom2 or S == atom2)) or (S == atom1 and N == atom2)

        valid_pair_objects = [
            [idx, self.ligand.atoms[pair[0]], self.protein.atoms[pair[1]]]
            for idx, pair in enumerate(valid_pairs)
        ]

        return [
            HydrogenBondPair(pair[0], pair[1], pair[2]) for pair in valid_pair_objects
            if (
                len(pair[1].hydrogens) > 0 and pair[2].acceptor and 
                    is_H_bond_functional_group(pair[1], pair[2])
            )
        ] + [
            HydrogenBondPair(pair[0], pair[2], pair[1]) for pair in valid_pair_objects
            if (
                len(pair[2].hydrogens) > 0 and pair[1].acceptor and 
                    is_H_bond_functional_group(pair[1], pair[2])
            )
        ]

    def total_energy(self):
        valid_pairs = filter_by_distance(self.protein, self.ligand)

        if len(valid_pairs) == 0:
            return 0
        
        feature_lists = np.array([
            [to_numpy(self.protein.atoms.features[y]), to_numpy(self.ligand.atoms.features[x])]
            for x, y in valid_pairs
        ])

        distances = np.linalg.norm(feature_lists[
            :, 0, Features.COORD
        ] - feature_lists[
            :, 1, Features.COORD
        ], axis=1)
        surface_dist = distances - (feature_lists[
            :, 0, Features.VDWr
        ] + feature_lists[
            :, 1, Features.VDWr
        ])
        
        possible_hydrogen_bonds = self.possible_hydrogen_bonds(valid_pairs)
        
        gauss1 = self.gauss1(surface_dist)
        gauss2 = self.gauss2(surface_dist)
        repulsion = self.repulsion(surface_dist)
        hydrophobic = self.hydrophobic(surface_dist, feature_lists)
        hydrogen_bonds = self.hydrogenbond(surface_dist, possible_hydrogen_bonds)

        return np.sum(self.W1 * gauss1 +
                        self.W2 * gauss2 +
                        self.W3 * repulsion +
                        self.W4 * hydrophobic +
                        self.W5 * hydrogen_bonds)
        

      
            


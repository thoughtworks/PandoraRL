import numpy as np
from rlvs.constants import Features, Vina
from ..bond import HydrogenBond, HydrophobicBond
from ..types import BondType
from rlvs.agents.utils import filter_by_distance, to_numpy
from ..named_atom import H
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

    def possible_hydrogen_bonds(self, valid_pairs):
        valid_pair_objects = [
            [idx, self.ligand.atoms[pair[0]], self.protein.atoms[pair[1]]]
            for idx, pair in enumerate(valid_pairs)
        ]

        return [
            HydrogenBond(pair[0], pair[1], pair[2]) for pair in valid_pair_objects
            if BondType.is_hydrogen_bond(pair[1], pair[2])
        ] + [
            HydrogenBond(pair[0], pair[2], pair[1]) for pair in valid_pair_objects
            if BondType.is_hydrogen_bond(pair[2], pair[1])
        ]

    def possible_hydrophobic_bonds(self, valid_pairs):
        return [
            HydrophobicBond(
                idx,
                self.protein.atoms[pair[1]],
                self.ligand.atoms[pair[0]]
            ) for idx, pair in enumerate(valid_pairs) if BondType.is_hydrophobic(
                    self.protein.atoms[pair[1]], self.ligand.atoms[pair[0]]
            )
        ]


    def hydrophobic(self, surface_dist, possible_hydrophobic_bonds):
        p1 = 0.5
        p2 = 1.5

        hyph = np.zeros(surface_dist.shape)
        valid_hydrophobic_bond_ids = [h_bond_pair.idx for h_bond_pair in possible_hydrophobic_bonds]

        hyph[valid_hydrophobic_bond_ids] = np.where(
            surface_dist[valid_hydrophobic_bond_ids] < p1, 1,
            np.where(
                (
                    surface_dist[valid_hydrophobic_bond_ids] >= p1
                ) & (
                    surface_dist[valid_hydrophobic_bond_ids] < p2
                ), p2 - surface_dist[valid_hydrophobic_bond_ids]/(p2 -p1),
                0
            )
        )

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

    def total_energy(self):
        valid_pairs = filter_by_distance(self.protein, self.ligand, distance_threshold=8)

        subset_by_distance = lambda var, distance, threshold: var[distance < threshold] 

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

        gauss1 = np.sum(self.gauss1(surface_dist))
        gauss2 = np.sum(self.gauss2(surface_dist))
        repulsion = np.sum(self.repulsion(surface_dist))

        valid_pairs = subset_by_distance(valid_pairs, distances, 8)
        surface_dist = subset_by_distance(surface_dist, distances, 8)

        possible_hydrogen_bonds = self.possible_hydrogen_bonds(valid_pairs)
        possible_hydrophobic_bonds = self.possible_hydrophobic_bonds(valid_pairs)

        hydrogen_bonds = self.hydrogenbond(surface_dist, possible_hydrogen_bonds)
        hydrophobic = self.hydrophobic(surface_dist, possible_hydrophobic_bonds)

        total_energy = self.W1 * gauss1 + self.W2 * gauss2 + self.W3 * repulsion +\
            np.sum(self.W4 * hydrophobic + self.W5 * hydrogen_bonds)

        print(
            f"""Gauss1:{gauss1}
Gauss2: {gauss2}
Repulsion: {repulsion}
Hydrophobic: {np.sum(hydrophobic)}
HydrogenBond: {np.sum(hydrogen_bonds)}
Total Energy : {total_energy}"""
                      )

        return total_energy
        

      
            


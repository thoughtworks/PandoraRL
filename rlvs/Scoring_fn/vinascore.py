import numpy as np

# Assuming the ligand atoms and protein atoms are saved in lists ligand.AllAtoms & receptor.AllAtoms:
# Each atom has descriptors attached to it as:
    # atom.element (type of element: "C", "O", "N" etc) 
    # atom.coordinates = atom.x, atom.y, atom.z (x, y, z coordinates)
    # atom.VDWr (Van der Waals radius, known for each element)
    # atom.hydrophobic_key (1 for yes, 0 for no)
    # atom.hdonor ((1 for yes, 0 for no)
    # atom.hacceptor ((1 for yes, 0 for no)
# If calculating conformation-independent function:
# N_rot = Number of rotatable bonds in ligand (If considering rigid ligand, put to zero)

def distance(self, point1, point2):
    deltax = point1.x - point2.x
    deltay = point1.y - point2.y
    deltaz = point1.z - point2.z
    dist = np.sqrt(deltax**2 + deltay**2 + deltaz**2)
    return dist

def angle_between_points(self, point1, point2):
    dist1 = np.sqrt(point1.x**2 + point1.y**2 + point1.z**2)
    dist2 = np.sqrt(point2.x**2 + point2.y**2 + point2.z**2)
    dot_prod = (point1.x*point2.x + point1.y*point2.y + point1.z*point2.z)/dist1/dist2
    if dot_prod > 1.0: dot_prod = 1.0 
    if dot_prod < -1.0: dot_prod = -1.0
    return np.arccos(dot_prod)*180.0/np.pi

def vector_subtraction(self, vector1, vector2): # vector1 - vector2
    return point(vector1.x - vector2.x, vector1.y - vector2.y, vector1.z - vector2.z)

def angle_between_three_points(self, point1, point2, point3): # As in three connected atoms
    vector1 = self.vector_subtraction(point1, point2)
    vector2 = self.vector_subtraction(point3, point2)
    return self.angle_between_points(vector1, vector2)

def surface_distance(self, atom1, atom2):
    dist = self.distance(atom1.coordinates, atom2.coordinates)
    dist_s = dist - atom1.VDWr - atom2.VDWr
    return dist_s

def gauss1(self, atom1, atom2):
    o1 = 0
    s1 = 0.5
    dist = self.surface_distance(atom1, atom2)
    g1 = np.exp(-((dist-o1)/s1)**2)
    return g1

def gauss2(self, atom1, atom2):
    o1 = 3
    s1 = 2
    dist = self.surface_distance(atom1, atom2)
    g2 = np.exp(-((dist-o1)/s1)**2)
    return g2

def repulsion(self, atom1, atom2):
    dist = self.surface_distance(atom1, atom2)
    if (dist < 0):
        repulse = dist**2
    else:
        repulse = 0
    return repulse

def hydrophobic(self, atom1, atom2):
    p1 = 0.5
    p2 = 1.5
    dist = self.surface_distance(atom1, atom2)
    if (dist < p1):
        hydr = 1
    elseif ((dist >= p1) & (dist < p2)):
        hydr = (p2-dist)/(p2-p1)
    else:
        hydr = 0
    return hydr

def hydrogenbond(self, atom1, atom2):
    h1 = -0.7
    dist = self.surface_distance(atom1, atom2)
    if (dist <= h1):
        hbond = 1
    elseif ((dist > h1) & (dist <= 0)):
        hbond = -dist/h1
    else:
        hbond = 0
    return hbond


w1 = -0.0356
w2 = -0.00516
w3 = 0.840
w4 = -0.0351
w5 = -0.587
w_rot = 0.0585

tot_energy = 0
for ligand_atom_index in ligand.AllAtoms:
    for receptor_atom_index in receptor.AllAtoms:
        ligand_atom = ligand.AllAtoms[ligand_atom_index]
        receptor_atom = receptor.AllAtoms[receptor_atom_index]
        gauss1 = 0
        gauss2 = 0
        repulsion = 0
        hydrophobic = 0
        hydrogenbond = 0
        dist = distance(ligand_atom.coordinates, receptor_atom.coordinates)
        if (dist < 8): # According to Vina paper, could be smaller ~ 4 A
            gauss1 = gauss1(ligand_atom, receptor_atom)
            gauss2 = gauss2(ligand_atom, receptor_atom)
            repulsion = repulsion(ligand_atom, receptor_atom)
            # Check if hydrophobic contact, then calculate hydrophobic term:
            if ((ligand_atom.element == "C") & (receptor_atom.element == "C")):
                if (hydrophobic_key == 1):
                    hydrophobic = hydrophobic(ligand_atom, receptor_atom)
            # Check if hydrogen bond possible between these two atoms.
            # Distance cutoff = 4 A, angle cutoff = 60 (This is generous, ideally bondlength should be ~ 1.8+1=2.8 A / 2.2+1=3.2 A)
            if (dist < 4):
                # If atoms are O or N, and either acceptor or donor:
                if ((ligand_atom.element == "O" or ligand_atom.element == "N") & (receptor_atom.element == "O" or receptor_atom.element == "N")):
                    if ((ligand_atom.hdonor == 1 and receptor_atom.hacceptor == 1) or (ligand_atom.hacceptor == 1 and receptor_atom.hdonor == 1)):
                        hbond = 0 ## If information available on which hydrogen is bonded to donor, can directly calculate O-H-N angle
                        # All hydrogens in ligand that are close enough to ligand donor atom and receptor acceptor atom, and bond angle > 120
                        if (ligand_atom.hdonor == 1):
                            for atm_index in ligand.AllAtoms:
                                # pick a hydrogen atom
                                if (ligand.AllAtoms[atm_index].element == "H"): 
                                    hydrogen = ligand.AllAtoms[atm_index]
                                    # O-H distance is 0.96 A, N-H is 1.01 A
                                    if (dist(ligand_atom.coordinates, hydrogen.coordinates) < 1.1): 
                                        # Bond-angle >= |120|
                                        if (np.abs(180-angle_between_three_points(ligand_atom.coordinates, hydrogen.coordinates, receptor_atom.coordinates)) <= 60.0):
                                            # Bond-length ~ 1.8 A / 2.2 A
                                            if (dist(receptor_atom.coordinates, hydrogen.coordinates) < 2.5): 
                                                hbond == 1
                                                break
                        # All hydrogens in receptor that are close enough to receptor donor atom and ligand acceptor atom, and bond angle > 120
                        if (receptor_atom.hdonor == 1):
                            for atm_index in receptor.AllAtoms:
                                # pick a hydrogen atom
                                if receptor.AllAtoms[atm_index].element == "H": 
                                    hydrogen = receptor.AllAtoms[atm_index]
                                    # O-H distance is 0.96 A, N-H is 1.01 A
                                    if (dist(receptor_atom.coordinates, hydrogen.coordinates) < 1.1):
                                        # Bond-angle >= |120|
                                        if (np.abs(180-angle_between_three_points(receptor_atom.coordinates, hydrogen.coordinates, ligand_atom.coordinates)) <= 60.0):
                                            # Bond-length ~ 1.8 A / 2.2 A
                                            if (dist(ligand_atom.coordinates, hydrogen.coordinates) < 2.5): 
                                                hbond == 1
                                                break
                        if (hbond == 1):
                            hydrogenbond = hydrogenbond(ligand_atom, receptor_atom)
            energy = w1*gauss1 + w2*gauss2 + w3*repulsion + w4*hydrophobic + w5*hydrogenbond
            tot_energy = tot_energy + energy

score = tot_energy/(1+w_rot*N_rot)

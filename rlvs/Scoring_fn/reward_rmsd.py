# Calculates Root-mean-square deviation (RMSD) between structure A and B in pdb format
# Then calculates the reward function for that pose
# No rotation or translation done here
# If using within RL code, may skip the reading of pdb (get_co-ordinates)
# if data is already in vector format
## Based on https://github.com/charnley/rmsd

import copy
import re

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# Dictionary of all elements matched with their atomic masses. 
# https://gist.github.com/lukasrichters14/c862644d4cbcf2d67252a484b7c6049c

ELEMENTS_WEIGHTS = {'h' : 1.008,'he' : 4.003, 'li' : 6.941, 'be' : 9.012,
                 'b' : 10.811, 'c' : 12.011, 'n' : 14.007, 'o' : 15.999,
                 'f' : 18.998, 'ne' : 20.180, 'na' : 22.990, 'mg' : 24.305,
                 'al' : 26.982, 'si' : 28.086, 'p' : 30.974, 's' : 32.066,
                 'cl' : 35.453, 'ar' : 39.948, 'k' : 39.098, 'ca' : 40.078,
                 'sc' : 44.956, 'ti' : 47.867, 'v' : 50.942, 'cr' : 51.996,
                 'mn' : 54.938, 'fe' : 55.845, 'co' : 58.933, 'ni' : 58.693,
                 'cu' : 63.546, 'zn' : 65.38, 'ga' : 69.723, 'ge' : 72.631,
                 'as' : 74.922, 'se' : 78.971, 'br' : 79.904, 'kr' : 84.798,
                 'rb' : 84.468, 'sr' : 87.62, 'y' : 88.906, 'zr' : 91.224,
                 'nb' : 92.906, 'mo' : 95.95, 'tc' : 98.907, 'ru' : 101.07,
                 'rh' : 102.906, 'pd' : 106.42, 'ag' : 107.868, 'cd' : 112.414,
                 'in' : 114.818, 'sn' : 118.711, 'sb' : 121.760, 'te' : 126.7,
                 'i' : 126.904, 'xe' : 131.294, 'cs' : 132.905, 'ba' : 137.328,
                 'la' : 138.905, 'ce' : 140.116, 'pr' : 140.908, 'nd' : 144.243,
                 'pm' : 144.913, 'sm' : 150.36, 'eu' : 151.964, 'gd' : 157.25,
                 'tb' : 158.925, 'dy': 162.500, 'ho' : 164.930, 'er' : 167.259,
                 'tm' : 168.934, 'yb' : 173.055, 'lu' : 174.967, 'hf' : 178.49,
                 'ta' : 180.948, 'w' : 183.84, 're' : 186.207, 'os' : 190.23,
                 'ir' : 192.217, 'pt' : 195.085, 'au' : 196.967, 'hg' : 200.592,
                 'tl' : 204.383, 'pb' : 207.2, 'bi' : 208.980, 'po' : 208.982,
                 'at' : 209.987, 'rn' : 222.081, 'fr' : 223.020, 'ra' : 226.025,
                 'ac' : 227.028, 'th' : 232.038, 'pa' : 231.036, 'u' : 238.029,
                 'np' : 237, 'pu' : 244, 'am' : 243, 'cm' : 247, 'bk' : 247,
                 'ct' : 251, 'es' : 252, 'fm' : 257, 'md' : 258, 'no' : 259,
                 'lr' : 262, 'rf' : 261, 'db' : 262, 'sg' : 266, 'bh' : 264,
                 'hs' : 269, 'mt' : 268, 'ds' : 271, 'rg' : 272, 'cn' : 285,
                 'nh' : 284, 'fl' : 289, 'mc' : 288, 'lv' : 292, 'ts' : 294,
                 'og' : 294}


def rmsd(V, W):
   # Calculate rmsd = Root-mean-square deviation from two sets of vectors
   # V and W ((N,D) matrices, where N is points and D is dimension).
    diff = np.array(V) - np.array(W)
    N = len(V)
    return np.sqrt((diff * diff).sum() / N)

def centroid(X):
   # Calculates centroid, i.e. mean position of all the points in
   # all of the coordinate directions, from a vectorset X.
    C = X.mean(axis=0)
    return C

def get_coordinates(filename):
    # Get coordinates from the first chain in a pdb file
    # and return a vectorset with all the coordinates.
    # Input filename : string; Filename to read
    # Output:
    # atoms : list; List of atomic types
    # V : array; (N,3) where N is number of atoms

    # PDB files tend to be messy. The x, y and z coordinates
    # are supposed to be in column 31-38, 39-46 and 47-54, but this is
    # not always the case.
    # Because of this the three first columns containing a decimal is used.
    # Since the format doesn't require a space between columns, we use the
    # above column indices as a fallback.

    x_column = None
    V = list()

    # Same with atoms and atom naming.
    # The most robust way to do this is probably
    # to assume that the atomtype is given in column 3.

    atoms = list()

    openfunc = open
    openarg = 'r'

    with openfunc(filename, openarg) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("TER") or line.startswith("END"):
                break
            if line.startswith("ATOM"):
                tokens = line.split()
                # Try to get the atomtype
                try:
                    atom = tokens[2][0]
                    if atom in ("H", "C", "N", "O", "S", "P"):
                        atoms.append(atom)
                    else:
                        # e.g. 1HD1
                        atom = tokens[2][1]
                        if atom == "H":
                            atoms.append(atom)
                        else:
                            raise Exception
                except:
                    exit("error: Parsing atomtype for the following line: \n{0:s}".format(line))

                if x_column is None:
                    try:
                        # look for x column
                        for i, x in enumerate(tokens):
                            if "." in x and "." in tokens[i + 1] and "." in tokens[i + 2]:
                                x_column = i
                                break
                    except IndexError:
                        exit("error: Parsing coordinates for the following line: \n{0:s}".format(line))
                # Try to read the coordinates
                try:
                    V.append(np.asarray(tokens[x_column:x_column + 3], dtype=float))
                except:
                    # If that doesn't work, use hardcoded indices
                    try:
                        x = line[30:38]
                        y = line[38:46]
                        z = line[46:54]
                        V.append(np.asarray([x, y ,z], dtype=float))
                    except:
                        exit("error: Parsing input for the following line: \n{0:s}".format(line))

    V = np.asarray(V)
    atoms = np.asarray(atoms)
    assert V.shape[0] == atoms.size

    return atoms, V

def return_rmsd(args):
# Reads the pdb files, transforms to vector representation, then calculates rmsd

    p_all_atoms, p_all = get_coordinates(args.structure_a)
    q_all_atoms, q_all = get_coordinates(args.structure_b)
    p_size = p_all.shape[0]
    q_size = q_all.shape[0]
    if not p_size == q_size:
        print("error: Structures not same size")
        quit()
    if np.count_nonzero(p_all_atoms != q_all_atoms):
        print("error: Atoms are not in the same order")
        exit()

    # Set local view
    p_view = None
    q_view = None
    if args.no_hydrogen:
        p_view = np.where(p_all_atoms != 'H')
        q_view = np.where(q_all_atoms != 'H')
    elif args.remove_idx:
        index = range(p_size)
        index = set(index) - set(args.remove_idx)
        index = list(index)
        p_view = index
        q_view = index
    elif args.add_idx:
        p_view = args.add_idx
        q_view = args.add_idx
    if p_view is None:
        p_coord = copy.deepcopy(p_all)
        q_coord = copy.deepcopy(q_all)
        p_atoms = copy.deepcopy(p_all_atoms)
        q_atoms = copy.deepcopy(q_all_atoms)
    else:
        p_coord = copy.deepcopy(p_all[p_view])
        q_coord = copy.deepcopy(q_all[q_view])
        p_atoms = copy.deepcopy(p_all_atoms[p_view])
        q_atoms = copy.deepcopy(q_all_atoms[q_view])

    # Create the centroid of P and Q which is the geometric center of a
    # N-dimensional region and translate P and Q onto that center.
    p_cent = centroid(p_coord)
    q_cent = centroid(q_coord)
    p_coord -= p_cent
    q_coord -= q_cent

    return rmsd(p_coord, q_coord)

def main():

    import argparse
    import sys

    parser = argparse.ArgumentParser(
        usage='rmsd_reward [options] FILE_A FILE_B',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Input structures
    parser.add_argument('structure_a', metavar='FILE_A', type=str, help='structures in .pdb format')
    parser.add_argument('structure_b', metavar='FILE_B', type=str)

    # Filter
    index_group = parser.add_mutually_exclusive_group()
    index_group.add_argument('-nh', '--no-hydrogen', action='store_true', help='ignore hydrogens when calculating RMSD')
    index_group.add_argument('--remove-idx', nargs='+', type=int, help='index list of atoms NOT to consider', metavar='IDX')
    index_group.add_argument('--add-idx', nargs='+', type=int, help='index list of atoms to consider', metavar='IDX')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args() 

    # dist can use rmsd from autodock vina directly instead of using return_rmsd()
    dist = return_rmsd(args)

    # Ref: https://hal.archives-ouvertes.fr/hal-00331752v2/document
    # alpha =  amplitude of bias
    # sigma = guess for bias influence area 
    # delta fixes the level of systematic exploration far away from the goal
    # The parameters alpha, sigma and delta may need tweaking depending on the data
    # delta and alpha should both be << 1 to avoid going round in circles
    alpha = 0.1
    delta = 0.1
    sigma = 2.0
    
    reward = alpha*np.exp(-dist**2/2/sigma**2) + delta
    print(reward)
    return

if __name__ == "__main__":
    main()

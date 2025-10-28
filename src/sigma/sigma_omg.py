# Standard library
from __future__ import division
import re
import os
from io import StringIO
from math import exp
from collections import namedtuple
import timeit
import json
import itertools
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Conda packages
import pandas
import scipy.spatial.distance
import numpy as np
import matplotlib.pyplot as plt


# From https://doi.org/10.1039/b801115j
# Covalent radii in angstrom, used to determine bonding
covalent_radius = {
    'H': 0.31,
    'He': 0.28,
    'Li': 1.28,
    'Be': 0.96,
    'B': 0.84,
    'C': 0.76,  # sp3 hybridization, sp2: 0.73 sp: 0.69
    'N': 0.71,
    'O': 0.66,
    'F': 0.57,
    'Ne': 0.58,
    'Na': 1.66,
    'Mg': 1.41,
    'Al': 1.21,
    'Si': 1.11,
    'P': 1.07,
    'S': 1.05,
    'Cl': 1.02,
    'Ar': 1.06,
    'K': 2.03,
    'Ca': 1.76,
    'Sc': 1.70,
    'Ti': 1.60,
    'V': 1.53,
    'Cr': 1.39,
    'Mn': 1.39,  # l.s.; h.s.: 1.61
    'Fe': 1.32,  # l.s.; h.s.: 1.52
    'Co': 1.26,  # l.s.; h.s.: 1.50
    'Ni': 1.24,
    'Cu': 1.32,
    'Zn': 1.22,
    'Ga': 1.22,
    'Ge': 1.20,
    'As': 1.19,
    'Se': 1.20,
    'Br': 1.20,
    'Kr': 1.16,
    'Rb': 2.20,
    'Sr': 1.95,
    'Y': 1.90,
    'Zr': 1.75,
    'Nb': 1.64,
    'Mo': 1.54,
    'Tc': 1.47,
    'Ru': 1.46,
    'Rh': 1.42,
    'Pd': 1.39,
    'Ag': 1.45,
    'Cd': 1.44,
    'In': 1.42,
    'Sn': 1.39,
    'Sb': 1.39,
    'Te': 1.38,
    'I': 1.39,
    'Xe': 1.40,
    'Cs': 2.44,
    'Ba': 2.15,
    'La': 2.07,
    'Ce': 2.04,
    'Pr': 2.03,
    'Nd': 2.01,
    'Pm': 1.99,
    'Sm': 1.98,
    'Eu': 1.98,
    'Gd': 1.96,
    'Tb': 1.94,
    'Dy': 1.92,
    'Ho': 1.92,
    'Er': 1.89,
    'Tm': 1.90,
    'Yb': 1.87,
    'Lu': 1.87,
    'Hf': 1.75,
    'Ta': 1.70,
    'W': 1.62,
    'Re': 1.51,
    'Os': 1.44,
    'Ir': 1.41,
    'Pt': 1.36,
    'Au': 1.36,
    'Hg': 1.32,
    'Tl': 1.45,
    'Pb': 1.46,
    'Bi': 1.48,
    'Po': 1.40,
    'At': 1.50,
    'Rn': 1.50,
    'Fr': 2.60,
    'Ra': 2.21,
    'Ac': 2.15,
    'Th': 2.06,
    'Pa': 2.00,
    'U': 1.96,
    'Np': 1.90,
    'Pu': 1.87,
    'Am': 1.80,
    'Cm': 1.69
}
# Construct the complete list of possible lengths of bonds. If a distance between atoms is less
# than the sum of covalent radii of the two atoms forming the possible bond, it is considered
# to be covalently bonded.
bond_distances = {
    (k1, k2): covalent_radius[k1] + covalent_radius[k2] for k1, k2 in itertools.combinations(covalent_radius.keys(), 2)
}
# Also put in the backwards pair (O,H instead of H,O)
for keys, d in bond_distances.copy().items():
    bond_distances[tuple(reversed(keys))] = d
# Also put in the atoms with themselves (e.g., C,C)
for key in covalent_radius.keys():
    bond_distances[(key, key)] = 2 * covalent_radius[key]


def get_seg_DataFrame(COSMO_contents):
    """Parse the segment data from COSMO file contents."""
    logger.debug("Starting to parse segment data from COSMO file")
    logger.debug(f"First 200 characters of COSMO contents: {COSMO_contents[:200]}")
    
    try:
        # For ORCA files
        pattern = r'SURFACE POINTS \(A.U.\)[\s(a-zA-Z0-9-)_#]+\n+([\s\S]+)'
        logger.debug(f"Attempting to match pattern: {pattern}")
        
        match = re.search(pattern, COSMO_contents, re.DOTALL)
        if match is None:
            logger.error("Failed to match SURFACE POINTS pattern in COSMO file")
            logger.debug("File content around expected location:")
            if 'SURFACE POINTS' in COSMO_contents:
                idx = COSMO_contents.find('SURFACE POINTS')
                logger.debug(COSMO_contents[max(0, idx-100):min(len(COSMO_contents), idx+100)])
            else:
                logger.error("'SURFACE POINTS' not found in file")
            raise ValueError("Could not find SURFACE POINTS section in COSMO file")
            
        sdata = match.group(1).rstrip()
        logger.debug(f"Successfully extracted segment data. First 100 chars: {sdata[:100]}")
        
        table_assign = ['x / a.u.', 'y / a.u.', 'z / a.u.', 'area / a.u.', 
                       'potential', 'charge / e', 'w_leb', 'Switch_F', 
                       'G_width', 'atom']
        
        return pandas.read_csv(StringIO(sdata), names=table_assign, sep=r'\s+', engine='python')
        
    except Exception as e:
        logger.error(f"Error parsing segment data: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        raise


def get_atom_DataFrame(COSMO_contents):
    """Parse the atom data from COSMO file contents."""
    logger.debug("Starting to parse atom data from COSMO file")
    
    try:
        pattern = r'CARTESIAN COORDINATES \(A.U.\)[\s\w+(.)-]+[#-]+\n+([\s0-9-.]+)+#'
        logger.debug(f"Attempting to match pattern: {pattern}")
        
        match = re.search(pattern, COSMO_contents, re.DOTALL)
        if match is None:
            logger.error("Failed to match CARTESIAN COORDINATES pattern in COSMO file")
            logger.debug("File content around expected location:")
            if 'CARTESIAN COORDINATES' in COSMO_contents:
                idx = COSMO_contents.find('CARTESIAN COORDINATES')
                logger.debug(COSMO_contents[max(0, idx-100):min(len(COSMO_contents), idx+100)])
            else:
                logger.error("'CARTESIAN COORDINATES' not found in file")
            raise ValueError("Could not find CARTESIAN COORDINATES section in COSMO file")
            
        sdata = match.group(1).rstrip()
        logger.debug(f"Successfully extracted atom data. First 100 chars: {sdata[:100]}")
        
        table_assign = ['x / a.u.', 'y / a.u.', 'z / a.u.', 'radii / a.u.']
        return pandas.read_csv(StringIO(sdata), names=table_assign, sep=r'\s+', engine='python')
        
    except Exception as e:
        logger.error(f"Error parsing atom data: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        raise


def get_area_volume(COSMO_contents):
    """Extract area and volume from COSMO file contents."""
    logger.debug("Starting to extract area and volume")
    
    try:
        # Look for volume and area at the top of the file
        pattern = r'\s*([0-9.]+)\s*# Volume\n\s*([0-9.]+)\s*# Area'
        logger.debug(f"Attempting to match pattern: {pattern}")
        
        match = re.search(pattern, COSMO_contents)
        if match is None:
            logger.error("Failed to match area/volume pattern in COSMO file")
            logger.debug("File content around expected location:")
            if 'Volume' in COSMO_contents:
                idx = COSMO_contents.find('Volume')
                logger.debug(COSMO_contents[max(0, idx-100):min(len(COSMO_contents), idx+100)])
            else:
                logger.error("'Volume' not found in file")
            raise ValueError("Could not find area/volume section in COSMO file")
            
        volume = float(match.group(1))  # a.u.
        area = float(match.group(2))  # a.u.
        
        # convert to A
        volume *= 0.52917721067 ** 3
        area *= 0.52917721067 ** 2
        
        logger.debug(f"Extracted area: {area:.6f} A^2, volume: {volume:.6f} A^3")
        return area, volume
        
    except Exception as e:
        logger.error(f"Error extracting area/volume: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        raise


def weightbin_sigmas(sigmavals, sigmas_grid):
    """
    """
    # Regular grid, so every bin is of same width
    bin_width = sigmas_grid[1] - sigmas_grid[0]
    psigmaA = np.zeros_like(sigmas_grid)
    flag = 0  # no out-of-range
    for sigma, area in sigmavals:
        # Check sigma
        if sigma < np.min(sigmas_grid):
            # raise ValueError('Sigma [{0:g}] is less than minimum of grid [{1}]'.format(sigma, np.min(sigmas_grid)))
            flag = 1
            continue
        if sigma > np.max(sigmas_grid):
            # raise ValueError('Sigma [{0:g}] is greater than maximum of grid [{1}]'.format(sigma, np.max(sigmas_grid)))
            flag = 1
            continue
        # The index to the left of the point in sigma
        left = int((sigma - sigmas_grid[0]) / bin_width)
        # Weighted distance from the left edge of the cell to the right side
        w_left = (sigmas_grid[left + 1] - sigma) / bin_width
        # Add the area into the left and right nodes of the bin, each part weighted by the value
        # of sigma, if equal to the left edge of the bin, then the w_left=1, if the right side,
        # w_left = 0
        psigmaA[left] += area * w_left
        psigmaA[left + 1] += area * (1.0 - w_left)
    if flag == 1:
        print(f"Following has sigma is out-of-range! [{np.min(sigmas_grid):.3f}, {np.max(sigmas_grid):.3f}]")
        raise ValueError
    return psigmaA


Dmol3COSMO = namedtuple('Dmol3COSMO', ['sigmas', 'psigmaA_nhb', 'psigmaA_OH', 'psigmaA_OT', 'df_atom', 'meta'])
DispersiveValues = namedtuple('DispersiveValues', ['dispersion_flag', 'dispersive_molecule', "Nbonds", 'has_COOH'])


class ORCACOSMOParser(object):

    def __init__(self, inpath, geometry_path, mol_path=None, num_profiles=1, averaging='Mullins', max_sigma=0.025):
        """Initialize the ORCA COSMO parser."""
        logger.debug(f"Initializing ORCACOSMOParser with {inpath}")
        
        self.mol_path = mol_path
        # Read the geometry file to get elements
        with open(geometry_path, 'r') as f:
            lines = f.readlines()
            try:
                self.num_atoms = int(lines[0])
                logger.debug(f"Number of atoms from first line: {self.num_atoms}")
                
                # Skip the first two lines (number of atoms and energy/comment line)
                self.elements = []
                for line in lines[2:2+self.num_atoms]:
                    parts = line.split()
                    if parts:  # Make sure line isn't empty
                        self.elements.append(parts[0])
                
                logger.debug(f"Read {len(self.elements)} elements from geometry file")
                logger.debug(f"Elements list: {self.elements}")
                
                if len(self.elements) != self.num_atoms:
                    logger.error(f"Mismatch in number of atoms: expected {self.num_atoms}, got {len(self.elements)}")
                    raise ValueError(f"Mismatch in number of atoms in geometry file")
                
            except Exception as e:
                logger.error(f"Error reading geometry file: {str(e)}")
                logger.error(f"File contents (first few lines):")
                for i, line in enumerate(lines[:min(5, len(lines))]):
                    logger.error(f"Line {i}: {line.strip()}")
                raise

        # Open the COSMO file and read in its contents
        # with open(inpath, 'r') as file:
        #     COSMO_contents = file.readlines()
        COSMO_contents = open(inpath).read()

        # Parse the COSMO file and get metadata
        self.df = get_seg_DataFrame(COSMO_contents)
        # change 'area / a.u. ' to 'area / A^2' -> 1 a.u. = 0.52917721067 A  # https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
        self.df['area / A^2'] = self.df['area / a.u.'] * 0.52917721067 ** 2
        # append n
        self.df['n'] = range(1, self.df.shape[0] + 1)

        self.df_atom = get_atom_DataFrame(COSMO_contents)
        # change 'x / a.u' -> x / A (angstrom)
        for field in ['x', 'y', 'z']:
            self.df_atom[field + ' / A'] = self.df_atom[
                                               field + ' / a.u.'] * 0.52917721067  # https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0

        # append atom symbol from the geometry path
        self.df_atom['atom'] = self.elements

        self.area_A2, self.volume_A3 = get_area_volume(COSMO_contents)

        averaging_options = ['Hsieh', 'Mullins']
        if averaging not in averaging_options:
            raise ValueError('averaging[' + averaging + '] not in ' + str(averaging_options))

        self.num_profiles = num_profiles
        self.averaging = averaging
        self.max_sigma = max_sigma

        # Convert coordinates in a.u. (actually, bohr) to Angstroms
        for field in ['x', 'y', 'z']:
            self.df[field + ' / A'] = self.df[
                                          field + ' / a.u.'] * 0.52917721067  # https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
        # Calculate the effective circular radius for this segment patch from its area
        self.df['rn / A'] = (self.df['area / A^2'] / np.pi) ** 0.5
        self.df['rn^2 / A^2'] = self.df['rn / A'] ** 2

        # Recalculate the charge density here because the vaues in COSMO file are unneccssarily truncated
        self.sigma = np.array(self.df['charge / e'] / self.df['area / A^2'])
        self.rn2 = np.array(self.df['rn^2 / A^2'])

        assert (int(self.df.iloc[-1].n) == len(self.rn2))

        # Calculate the distances between each pair of segments in a euclidean sense
        XA = np.c_[self.df['x / A'], self.df['y / A'], self.df['z / A']]
        self.dist_mat_squared = scipy.spatial.distance.cdist(XA, XA, 'euclidean') ** 2

        # Calculate the distances between each atom center in a euclidean sense
        XA = np.c_[self.df_atom['x / A'], self.df_atom['y / A'], self.df_atom['z / A']]
        self.dist_mat_atom = scipy.spatial.distance.cdist(XA, XA, 'euclidean')

        # Set a flag if the molecule is water; water is treated specially in some cases
        self.is_water = self.df_atom.atom.tolist().count('H') == 2 and self.df_atom.atom.tolist().count(
            'O') == 1 and len(self.df_atom) == 3

        # Calculate the dispersive values
        self.disp = self.get_dispersive_values()

        # Tag each atom with its hydrogen bonding class
        self.df_atom['hb_class'] = self.get_HB_classes_per_atom()

        # Store the number of bonds in the DataFrame
        self.df_atom['Nbonds'] = self.disp.Nbonds

        # Average the charge densities on each segment
        self.sigma_averaged = self.average_sigmas(self.sigma)

        # Split up the profiles
        self.sigma_nhb, self.sigma_OH, self.sigma_OT = self.split_profiles(self.sigma_averaged, self.num_profiles)

    def read_mol_file(self, mol_path):
        """Read bond information from a .mol file."""
        logger.debug(f"Reading bond information from {mol_path}")
        
        with open(mol_path, 'r') as f:
            lines = f.readlines()
            
            # Skip header (first 3 lines)
            counts_line = lines[3]
            n_atoms = int(counts_line.split()[0])
            n_bonds = int(counts_line.split()[1])
            
            # Skip atom block
            bond_start = 4 + n_atoms
            
            # Read bonds
            bonds = [[] for _ in range(n_atoms)]
            for i in range(n_bonds):
                line = lines[bond_start + i]
                atom1 = int(line[0:3]) - 1  # MOL files are 1-indexed
                atom2 = int(line[3:6]) - 1
                bond_type = int(line[6:9])  # 1=single, 2=double, 3=triple
                
                # Add bonds both ways
                bonds[atom1].append((atom2, bond_type))
                bonds[atom2].append((atom1, bond_type))
            
            return bonds

    def get_bonds(self, df_atom, elements, mol_path=None):
        """
        Determine bonds between atoms using .mol file if available, 
        otherwise fall back to distance-based detection.
        """
        if mol_path and os.path.exists(mol_path):
            logger.debug("Using .mol file for bond information")
            mol_bonds = self.read_mol_file(mol_path)
            
            # Convert numeric bonds to element bonds
            bonds = [[] for _ in range(len(df_atom))]
            for i in range(len(df_atom)):
                for j, bond_type in mol_bonds[i]:
                    bonds[i].append((j, elements[j]))
            
            return bonds
        
        else:
            logger.warning("No .mol file provided, falling back to distance-based bond detection")
            # Convert coordinates to Angstroms
            coords = df_atom[['x / a.u.', 'y / a.u.', 'z / a.u.']].values * 0.52917721067
            
            # Calculate all pairwise distances
            distances = scipy.spatial.distance.pdist(coords)
            distances = scipy.spatial.distance.squareform(distances)
            
            # Initialize bonds list
            bonds = [[] for _ in range(len(df_atom))]
            
            # Single pass through all atom pairs
            for i in range(len(df_atom)):
                element_i = elements[i]
                r_i = covalent_radius[element_i]
                
                for j in range(i+1, len(df_atom)):  # Only look at each pair once
                    element_j = elements[j]
                    r_j = covalent_radius[element_j]
                    
                    # Use a single, generous distance criterion
                    max_bond_dist = (r_i + r_j) * 1.4  # 40% tolerance
                    
                    if distances[i,j] <= max_bond_dist:
                        # Add bond both ways
                        bonds[i].append((j, element_j))
                        bonds[j].append((i, element_i))
                        logger.debug(f"Bond found: {element_i}{i}-{element_j}{j} "
                                   f"(distance: {distances[i,j]:.3f} Å)")
            
            # Basic validation
            for i, (atom_bonds, element) in enumerate(zip(bonds, elements)):
                if len(atom_bonds) == 0 and len(df_atom) > 1:
                    logger.error(f"Atom {i} ({element}) has no bonds")
                    # Show distances to all other atoms
                    for j in range(len(df_atom)):
                        if i != j:
                            logger.error(f"  Distance to {elements[j]}{j}: {distances[i,j]:.3f} Å")
            
            return bonds

    def get_HB_classes_per_atom(self):
        """Determine the hydrogen bonding class of each atom."""
        logger.debug("Starting hydrogen bond classification")
        
        hydrogen_bonding_atoms = []
        for i in range(len(self.df_atom)):
            atom_name_i = self.df_atom.atom.iloc[i]
            logger.debug(f"Processing atom {i}: {atom_name_i}")
            
            if atom_name_i in ['N', 'F']:
                # Definite hydrogen bonding atom, cannot be in OH class
                hydrogen_bonding_atoms.append('OT')
                continue

            elif atom_name_i in ['O', 'H']:
                # Get bonds for current atom
                bonds = self.get_bonds(self.df_atom, self.elements, self.mol_path)[i]  # Get bonds for atom i
                logger.debug(f"Bonds for atom {i}: {bonds}")
                
                # Extract atom names from bonds
                bonded_atoms = [atom_name for _, atom_name in bonds]
                logger.debug(f"Bonded atoms: {bonded_atoms}")
                
                atom_type = 'NHB'
                
                # Check for OH bonds
                if (atom_name_i == 'H' and 'O' in bonded_atoms) or \
                   (atom_name_i == 'O' and 'H' in bonded_atoms):
                    atom_type = 'OH'
                # Check for oxygen bonded to non-H
                elif atom_name_i == 'O' and bonded_atoms:
                    atom_type = 'OT'
                # Check for H bonded to F or N
                elif atom_name_i == 'H' and ('F' in bonded_atoms or 'N' in bonded_atoms):
                    atom_type = 'OT'
                
                hydrogen_bonding_atoms.append(atom_type)
                logger.debug(f"Assigned type {atom_type} to atom {i}")

            else:
                # Definite non-hydrogen-bonding
                hydrogen_bonding_atoms.append('NHB')
        
        return hydrogen_bonding_atoms

    def get_dispersive_values(self):
        """Calculate the dispersive parameters needed for the COSMO-SAC-dsp model."""
        logger.debug("Starting dispersive value calculation")
        
        dispersive_parameter_lib = {
            'C(sp3)': 115.7023,
            'C(sp2)': 117.4650,
            'C(sp)': 66.0691,
            '-O-': 95.6184,
            '=O': -11.0549,
            'N(sp3)': 15.4901,
            'N(sp2)': 84.6268,
            'N(sp)': 109.6621,
            'F': 52.9318,
            'Cl': 104.2534,
            'H(OH)': 19.3477,
            'H(NH)': 141.1709,
            'H(water)': 58.3301,
            'H(other)': 0
        }

        dispersive_molecule = 0
        invalid_atom = False
        has_COOH = False
        Natom_nonzero = 0
        dispersion_flag = 'NHB'
        Nbonds = []
        
        # Get all bonds at once
        all_bonds = self.get_bonds(self.df_atom, self.elements, self.mol_path)
        
        for i in range(len(self.df_atom)):
            dispersive_parameter = None
            atom_name_i = self.df_atom.atom.iloc[i]
            atom_bonds = all_bonds[i]  # Get bonds for current atom
            
            logger.debug(f"Processing atom {i} ({atom_name_i}) with {len(atom_bonds)} bonds")
            
            if len(atom_bonds) == 0 and len(self.df_atom) > 1:  # Check if it's not a single atom molecule
                raise ValueError(f"Atom {atom_name_i} in a polyatomic molecule has no bonds, this is not good.")
            
            Nbonds.append(len(atom_bonds))
            bonded_atoms = [atom_name for _, atom_name in atom_bonds]
            
            if atom_name_i == 'C':
                if len(atom_bonds) == 4:
                    dispersive_parameter = dispersive_parameter_lib['C(sp3)']
                elif len(atom_bonds) == 3:
                    dispersive_parameter = dispersive_parameter_lib['C(sp2)']
                elif len(atom_bonds) == 2:
                    dispersive_parameter = dispersive_parameter_lib['C(sp)']
                    
                # Check for COOH group
                if len(atom_bonds) == 3 and bonded_atoms.count('O') == 2:
                    for j, atom_name_j in atom_bonds:
                        if atom_name_j == 'O':
                            o_bonds = all_bonds[j]
                            o_bonded_atoms = [name for _, name in o_bonds]
                            if sorted(o_bonded_atoms) == ['C', 'H']:
                                has_COOH = True
                                break
                            
            elif atom_name_i == 'N':
                if len(atom_bonds) == 3:
                    dispersive_parameter = dispersive_parameter_lib['N(sp3)']
                elif len(atom_bonds) == 2:
                    dispersive_parameter = dispersive_parameter_lib['N(sp2)']
                elif len(atom_bonds) == 1:
                    dispersive_parameter = dispersive_parameter_lib['N(sp)']
                else:
                    raise ValueError(f'N with unexpected number of bonds: {len(atom_bonds)}')
                
            elif atom_name_i == 'O':
                if len(atom_bonds) == 2:
                    dispersive_parameter = dispersive_parameter_lib['-O-']
                elif len(atom_bonds) == 1:
                    dispersive_parameter = dispersive_parameter_lib['=O']
                elif len(atom_bonds) == 3:
                    logger.warning(f"Oxygen atom {i} has 3 bonds - treating as '-O-' type")
                    dispersive_parameter = dispersive_parameter_lib['-O-']
                else:
                    raise ValueError(f'O with unexpected number of bonds: {len(atom_bonds)}')
                
            elif atom_name_i == 'F':
                dispersive_parameter = dispersive_parameter_lib['F']
                
            elif atom_name_i == 'Cl':
                dispersive_parameter = dispersive_parameter_lib['Cl']
                
            elif atom_name_i == 'H':
                if self.is_water:
                    dispersive_parameter = dispersive_parameter_lib['H(water)']
                elif 'O' in bonded_atoms:
                    dispersive_parameter = dispersive_parameter_lib['H(OH)']
                elif 'N' in bonded_atoms:
                    dispersive_parameter = dispersive_parameter_lib['H(NH)']
                else:
                    dispersive_parameter = dispersive_parameter_lib['H(other)']
            else:
                invalid_atom = True
                dispersive_parameter = np.nan
            
            if dispersive_parameter is not None:
                Natom_nonzero += 1
                dispersive_molecule += dispersive_parameter
            
            logger.debug(f"Atom {i} ({atom_name_i}): parameter = {dispersive_parameter}")

        # Calculate average dispersive value
        if Natom_nonzero > 0:
            dispersive_molecule /= Natom_nonzero
        
        if invalid_atom:
            dispersive_molecule = np.nan

        # Determine hydrogen bonding flags
        possible_Hbonders = ['O', 'N', 'F']
        if any(atom in set(self.df_atom.atom) for atom in possible_Hbonders):
            Hbond = False
            for i, atom_name_i in enumerate(self.df_atom.atom):
                if atom_name_i in possible_Hbonders:
                    bonded_atoms = [atom_name for _, atom_name in all_bonds[i]]
                    if 'H' in bonded_atoms:
                        Hbond = True
                        break
        
            dispersion_flag = 'HB-DONOR-ACCEPTOR' if Hbond else 'HB-ACCEPTOR'

        return DispersiveValues(dispersion_flag, dispersive_molecule, Nbonds, has_COOH)

    def average_sigmas(self, sigmavals):
        """
        Calculate the averaged charge densities on each segment, in e/\AA^2
        """

        # This code also works, kept for demonstration purposes, but the vectorized
        # numpy code is much faster, if a bit harder to understand
        # def calc_sigma_m(m):
        #     increments = rn2*r_av2/(rn2+r_av2)*np.exp(-f_decay*dist_mat_squared[m,:]/(rn2+r_av2))
        #     return np.sum(increments*sigma)/np.sum(increments)
        # def calc_sigma_m_mat(m):
        #     increments = THEMAT[m,:]
        #     return np.sum(increments*sigma)/np.sum(increments)

        if self.averaging == 'Mullins':
            self.r_av2 = 0.8176300195 ** 2  # [A^2]  # Also equal to ((7.5/pi)**0.5*0.52917721092)**2
            self.f_decay = 1.0
        elif self.averaging == 'Hsieh':
            self.r_av2 = (7.25 / np.pi)  # [A^2]
            self.f_decay = 3.57
        else:
            raise ValueError("??")

        THEMAT = np.exp(-self.f_decay * self.dist_mat_squared / (self.rn2 + self.r_av2)) * self.rn2 * self.r_av2 / (
                    self.rn2 + self.r_av2)
        return np.sum(THEMAT * sigmavals, axis=1) / np.sum(THEMAT, axis=1)

    def split_profiles(self, sigmavals, num_profiles):
        """
        Split the samples into
        """

        if num_profiles == 1:
            sigma_nhb = [(s, a) for s, a in zip(sigmavals, self.df['area / A^2'])]
            sigma_OH = None
            sigma_OT = None
        elif num_profiles == 3:

            # indices = np.array(self.df.atom.astype(int)-1)
            indices = np.array(self.df.atom.astype(int))  # not -1 for orca

            self.df['atom_name'] = self.df_atom.atom[indices].reset_index(drop=True)
            self.df['hb_class'] = self.df_atom.hb_class[indices].reset_index(drop=True)

            # N.B. : The charge density in sigmavals is the averaged charge density,
            # not the charge density of the segment itself!!
            mask_OH = (
                    ((self.df.atom_name == 'O') & (sigmavals > 0.0) & (self.df.hb_class == 'OH'))
                    |
                    ((self.df.atom_name == 'H') & (sigmavals < 0.0) & (self.df.hb_class == 'OH'))
            )
            mask_OT = (
                    (self.df.atom_name.isin(['O', 'N', 'F']) & (sigmavals > 0.0) & (self.df.hb_class == 'OT'))
                    |
                    ((self.df.atom_name == 'H') & (sigmavals < 0.0) & (self.df.hb_class == 'OT'))
            )
            mask_nhb = ~(mask_OT | mask_OH)
            sigma_nhb = np.c_[sigmavals[mask_nhb], self.df[mask_nhb]['area / A^2']]
            sigma_OH = np.c_[sigmavals[mask_OH], self.df[mask_OH]['area / A^2']]
            sigma_OT = np.c_[sigmavals[mask_OT], self.df[mask_OT]['area / A^2']]
        else:
            raise ValueError('Number of profiles [{0}] is invalid'.format(num_profiles))
        return sigma_nhb, sigma_OH, sigma_OT

    def get_meta(self):
        """
        Return a dictionary with the metadata about this set of sigma profiles
        """
        return {
            'name': '?',
            'CAS': '?',
            'area [A^2]': self.area_A2,
            'volume [A^3]': self.volume_A3,
            'r_av [A]': self.r_av2 ** 0.5,
            'f_decay': self.f_decay,
            'sigma_hb [e/A^2]': 0.0084,
            'averaging': self.averaging
        }

    def get_outputs(self):
        # The points where the surface-charge density
        # will be evaluated, -0.025 to 0.025, in increments of 0.001
        bin_width = 0.001
        sigmas = np.arange(-self.max_sigma, self.max_sigma + 1e-6, bin_width)  # [e/A^2]. Sometimes OMG polymers have sigmas out of this range.
        meta = self.get_meta()

        if self.num_profiles == 1:
            psigmaA = weightbin_sigmas(self.sigma_nhb, sigmas)
            # print('cumulative time after reweighting', timeit.default_timer()-tic,'s')
            # assert (abs(sum(psigmaA) - meta['area [A^2]']) < 0.001)
            absolute_difference = abs(sum(psigmaA) - meta['area [A^2]'])
            if not (absolute_difference < 0.001):
                print(f'[Caution] The absolute difference between sum(psigmA) and actual area is {absolute_difference:.4f}')
                print(f'[Caution] The relative difference (%) from total area is {absolute_difference / meta["area [A^2]"] * 100:.3f}%')
                print(f'Following has an absolute difference larger than the threshold = 0.001')
                raise ValueError
            return Dmol3COSMO(sigmas, psigmaA, None, None, self.df_atom, self.get_meta())

        elif self.num_profiles == 3:
            psigmaA_nhb = weightbin_sigmas(self.sigma_nhb, sigmas)
            psigmaA_OH = weightbin_sigmas(self.sigma_OH, sigmas)
            psigmaA_OT = weightbin_sigmas(self.sigma_OT, sigmas)
            sigma_0 = 0.007  # [e/A^2]

            psigmaA_hb = psigmaA_OT + psigmaA_OH

            P_hb = 1 - np.exp(-sigmas ** 2 / (2 * sigma_0 ** 2))
            psigmaA_OH *= P_hb
            psigmaA_OT *= P_hb

            psigmaA_nhb = psigmaA_nhb + psigmaA_hb * (1 - P_hb)

            dispersion_flag = 'NHB'

            # Determine dispersion flag for the molecule
            if self.is_water:
                dispersion_flag = 'H2O'
            elif self.disp.has_COOH:
                dispersion_flag = 'COOH'
            else:
                dispersion_flag = self.disp.dispersion_flag

            meta['disp. flag'] = dispersion_flag
            meta['disp. e/kB [K]'] = self.disp.dispersive_molecule
            # print('cumulative time after reweighting', timeit.default_timer()-tic,'s')
            return Dmol3COSMO(sigmas, psigmaA_nhb, psigmaA_OH, psigmaA_OT, self.df_atom, meta)

    def validate_bonds(self, all_bonds):
        """Validate expected number of bonds for each atom type."""
        for i, atom_name in enumerate(self.df_atom.atom):
            n_bonds = len(all_bonds[i])
            bonded_atoms = [atom_name for _, atom_name in all_bonds[i]]
            
            logger.debug(f"Atom {i} ({atom_name}) has {n_bonds} bonds to: {bonded_atoms}")
            
            if atom_name == 'C' and n_bonds not in [2,3,4]:
                logger.warning(f"Carbon atom {i} has unusual number of bonds: {n_bonds}")
                logger.warning(f"Bonded to: {bonded_atoms}")
                
            elif atom_name == 'O' and n_bonds not in [1,2,3]:  # Allow 3 bonds for now
                logger.warning(f"Oxygen atom {i} has unusual number of bonds: {n_bonds}")
                logger.warning(f"Bonded to: {bonded_atoms}")
                
            elif atom_name == 'H' and n_bonds != 1:
                logger.warning(f"Hydrogen atom {i} has unusual number of bonds: {n_bonds}")
                logger.warning(f"Bonded to: {bonded_atoms}")


def read_orca(inpath, geometry_path, mol_path=None, **kwargs):
    """
    A convenience function that will pass all arguments along to class and then return the outputs
    
    Parameters:
    -----------
    inpath : str
        Path to the ORCA COSMO output file
    geometry_path : str
        Path to the geometry file (xyz format)
    mol_path : str, optional
        Path to the mol file containing bond information
    **kwargs : dict
        Additional arguments to pass to ORCACOSMOParser
    """
    return ORCACOSMOParser(inpath, geometry_path, mol_path=mol_path, **kwargs).get_outputs()


def overlay_profile(sigmas, psigmaA, path):
    for profile in psigmaA:
        plt.plot(sigmas, profile, 'o-', mfc='none')
        print(np.c_[sigmas, profile])
    df = pandas.read_csv(path, names=['charge/area / e/A^2', 'p(sigma)*A / A^2'], skiprows=4, sep=' ')
    plt.plot(df['charge/area / e/A^2'], df['p(sigma)*A / A^2'], '^-', mfc='none')
    plt.xlabel(r'$\sigma$ / e/A$^2$')
    plt.ylabel(r'$p(\sigma)A_i$ / A$^2$')
    plt.show()


Delaware_template = """# meta: {meta:s}
# Rows are given as: sigma [e/A^2] followed by a space, then psigmaA [A^2]
# In the case of three sigma profiles, the order is NHB, OH, then OT
"""


def write_sigma(dmol, ofpath, header='Delaware', force=True):
    if header == 'Delaware':
        out = Delaware_template.format(meta=json.dumps(dmol.meta))
    else:
        raise ValueError("Bad header option")
    summ = 0
    for profile in [dmol.psigmaA_nhb, dmol.psigmaA_OH, dmol.psigmaA_OT]:
        if profile is not None:
            for ir in range(profile.shape[0]):
                # out += '{0:0.3f} {1:17.14e}\n'.format(dmol.sigmas[ir], profile[ir])
                out += '{0:0.4f} {1:17.14e}\n'.format(dmol.sigmas[ir], profile[ir])
                summ += profile[ir]

    if os.path.exists(ofpath) and not force:
        raise ValueError("File [{0:s}] already exists and force has not been requested".format(ofpath))

    with open(ofpath, 'w') as fp:
        fp.write(out)

# checked
def get_sigma_profiles(inpath, outpath, geometry_path, mol_path=None, num_profiles=1, averaging='Mullins', max_sigma=0.025):
    """
    This function generates & saves a sigma profile from a COSMO file to the output path.
    """
    logger.debug(f"Starting sigma profile generation with inputs:")
    logger.debug(f"inpath: {inpath}")
    logger.debug(f"outpath: {outpath}")
    logger.debug(f"geometry_path: {geometry_path}")
    logger.debug(f"mol_path: {mol_path}")
    logger.debug(f"num_profiles: {num_profiles}")
    logger.debug(f"averaging: {averaging}")
    logger.debug(f"max_sigma: {max_sigma}")

    # Check if input files exist
    if not os.path.exists(inpath):
        logger.error(f"COSMO file not found: {inpath}")
        raise FileNotFoundError(f"COSMO file not found: {inpath}")
    
    if not os.path.exists(geometry_path):
        logger.error(f"Geometry file not found: {geometry_path}")
        raise FileNotFoundError(f"Geometry file not found: {geometry_path}")

    # Read ORCA output
    try:
        logger.debug("Attempting to read ORCA output...")
        orca = read_orca(inpath=inpath, 
                        num_profiles=num_profiles, 
                        averaging=averaging, 
                        geometry_path=geometry_path,
                        mol_path=mol_path,
                        max_sigma=max_sigma)
        if orca is None:
            logger.error("read_orca returned None")
            raise ValueError("Failed to read ORCA output")
        logger.debug("Successfully read ORCA output")
    except Exception as e:
        logger.error(f"Error reading ORCA output: {str(e)}")
        raise

    # Write sigma profile
    try:
        logger.debug(f"Attempting to write sigma profile to {outpath}")
        write_sigma(orca, outpath)
        logger.debug("Successfully wrote sigma profile")
    except Exception as e:
        logger.error(f"Error writing sigma profile: {str(e)}")
        raise

    return None

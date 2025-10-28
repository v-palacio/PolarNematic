import os
import re
import math
import numpy as np
from scipy import stats
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Geometry import Point3D
from molecular_descriptors import (
    descriptors_3d, get_additional_descriptors, 
    analyze_conformers, calculate_omg_descriptors
)

def create_mol_dict(mol_dir, suffix='.mol'):
    """
    Create a dictionary mapping molecule names to RDKit molecule objects.
    
    Parameters:
    -----------
    mol_dir : str
        Directory containing molecular structure files
    suffix : str
        File suffix for molecular structure files
        
    Returns:
    --------
    dict
        Dictionary mapping molecule names to RDKit molecule objects
    list
        List of molecule names in order
    list
        List of RDKit molecule objects in order
    """
    mols = []
    mol_names = []
    mol_dict = {}
    
    for filename in os.listdir(mol_dir):
        if filename.endswith(suffix):
            name = filename.replace(suffix, '')
            molpath = os.path.join(mol_dir, filename)
            tmpmol = Chem.MolFromMolFile(molpath)
            if tmpmol is not None:
                Chem.SanitizeMol(tmpmol)
                mols.append(tmpmol)
                mol_names.append(name)
                mol_dict[name] = tmpmol
    
    return mol_dict, mol_names, mols

def select_target(row):
    """Helper function to select appropriate transition temperature."""
    if row['Transition type'] == 2:
        return row['I-N']
    elif row['Transition type'] == 4:
        return row['N-Nx']
    elif row['Transition type'] == 5:
        return row['Nx-Nf']
    else:
        return np.nan

def check_duplicates(df, mol_dict):
    """
    Check for various types of duplicates in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The processed dataframe
    mol_dict : dict
        Dictionary of molecule names to RDKit molecule objects
    
    Returns:
    --------
    dict
        Dictionary containing duplicate information and statistics
    """
    duplicate_info = {}
    
    # Check for duplicate names
    name_duplicates = df[df.duplicated(['Name_x'], keep=False)]
    duplicate_info['name_duplicates'] = name_duplicates
    
    # Check for duplicate SMILES
    smiles_duplicates = df[df.duplicated(['SMILES'], keep=False)]
    duplicate_info['smiles_duplicates'] = smiles_duplicates
    
    # Check for structural duplicates using RDKit canonical SMILES
    canonical_smiles = {name: Chem.MolToSmiles(mol, canonical=True) 
                       for name, mol in mol_dict.items()}
    canonical_duplicates = {}
    
    for name1, smiles1 in canonical_smiles.items():
        for name2, smiles2 in canonical_smiles.items():
            if name1 < name2 and smiles1 == smiles2:
                if smiles1 not in canonical_duplicates:
                    canonical_duplicates[smiles1] = []
                canonical_duplicates[smiles1].append((name1, name2))
    # Find indices of all duplicate molecules
    duplicate_indices = []
    for smiles, pairs in canonical_duplicates.items():
        for name1, name2 in pairs:
            idx1 = df.index[df['Name_x'] == name1].tolist()[0]
            idx2 = df.index[df['Name_x'] == name2].tolist()[0]

            # Keep the one with smaller index
            if idx1 < idx2:
                duplicate_indices.append(idx2)
            else:
                duplicate_indices.append(idx1)
    
    duplicate_info['structural_duplicates'] = canonical_duplicates
    duplicate_info['duplicate_indices'] = sorted(duplicate_indices)
    
    # Generate summary statistics
    stats = {
        'total_molecules': len(df),
        'unique_names': len(df['Name_x'].unique()),
        'unique_smiles': len(df['SMILES'].unique()),
        'unique_structures': len(set(canonical_smiles.values())),
        'name_duplicate_count': len(name_duplicates),
        'smiles_duplicate_count': len(smiles_duplicates),
        'structural_duplicate_count': sum(len(pairs) for pairs in canonical_duplicates.values())
    }
    
    duplicate_info['statistics'] = stats
    
    return duplicate_info

def load_and_process_data(mol_dir='./structure/mol/', 
                         temp_file='Temperature.xlsx',
                         output_file='final_descriptors.csv',
                         force_recalc=False,
                         check_dups=True,
                         descriptor_type='full'):
    """
    Load and process molecular data and temperature data.
    
    Parameters:
    -----------
    mol_dir : str
        Directory containing molecular structure files
    temp_file : str
        Path to temperature data Excel file
    output_file : str
        Path to save processed data
    force_recalc : bool
        If True, recalculate descriptors even if output file exists
    check_dups : bool
        If True, perform duplicate checking
    descriptor_type : str
        Type of descriptors to calculate: 'full' or 'omg'
        
    Returns:
    --------
    pd.DataFrame
        Processed and merged dataframe with transition temperatures
    dict
        Dictionary mapping molecule names to RDKit molecule objects
    dict, optional
        Duplicate information if check_dups is True
    """
    
    # Adjust output filename based on descriptor type
    if descriptor_type == 'omg':
        output_file = output_file.replace('final_descriptors.csv', 'omg_descriptors.csv')
    
    # Check if processed data already exists
    if os.path.exists(output_file) and not force_recalc:
        print(f"Loading existing processed data from {output_file}")
        final_df = pd.read_csv(output_file)
        mol_dict, _, _ = create_mol_dict(mol_dir)
        
        if check_dups:
            duplicate_info = check_duplicates(final_df, mol_dict)
            print("\nDuplicate Analysis:")
            print(f"Total molecules: {duplicate_info['statistics']['total_molecules']}")
            print(f"Unique names: {duplicate_info['statistics']['unique_names']}")
            print(f"Unique SMILES: {duplicate_info['statistics']['unique_smiles']}")
            print(f"Unique structures: {duplicate_info['statistics']['unique_structures']}")
            return final_df, mol_dict, duplicate_info
        return final_df, mol_dict
    
    print(f"Processing molecular data using {descriptor_type} descriptors...")
    
    # Load molecules and maintain mapping
    mol_dict, mol_names, mols = create_mol_dict(mol_dir)

    # Calculate descriptors based on type
    descrs = []
    for mol in mols:
        # Generate 3D conformation
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        if descriptor_type == 'full':
            # Calculate all descriptors
            desc_2d = Descriptors.CalcMolDescriptors(mol)
            desc_3d = descriptors_3d(mol)
            desc_additional = get_additional_descriptors(mol)
            conformer_stats = analyze_conformers(mol, n_conformers=10)
            
            # Combine all descriptors
            all_desc = {**desc_2d, **desc_3d}
            if conformer_stats:
                all_desc.update(conformer_stats)
            if desc_additional:
                all_desc.update(desc_additional)
            descrs.append(all_desc)
        
        elif descriptor_type == 'omg':
            # Calculate only OMG descriptors
            desc = calculate_omg_descriptors(mol)
            if desc is not None:
                descrs.append(desc)
            else:
                print(f"Warning: Failed to calculate OMG descriptors for molecule")
        
        else:
            raise ValueError(f"Unknown descriptor type: {descriptor_type}")

    # Create DataFrame with descriptors
    descriptors_df = pd.DataFrame(descrs)
    descriptors_df['Name'] = mol_names
    descriptors_df['SMILES'] = [Chem.MolToSmiles(mol) for mol in mols]

    # Load temperature data
    temp_df = pd.read_excel(temp_file)
    print("Temperature DataFrame columns:", temp_df.columns)

    # Merge dataframes
    merged_df = pd.merge(temp_df, descriptors_df, on='Name', how='inner')

    # Add SMILES to temperature data if needed
    if 'SMILES' not in temp_df.columns:
        print("Creating SMILES column in temperature DataFrame...")
        for idx, row in temp_df.iterrows():
            mol_file = os.path.join(mol_dir, row['Name'] + '.mol')
            if os.path.exists(mol_file):
                mol = Chem.MolFromMolFile(mol_file)
                if mol is not None:
                    temp_df.loc[idx, 'SMILES'] = Chem.MolToSmiles(mol)
        merged_df = pd.merge(temp_df, descriptors_df, on='SMILES', how='inner')

    # Remove duplicates
    merged_df = merged_df.drop_duplicates(subset=['Name_x', 'SMILES'], keep='first')
    
    # Process transition temperatures
    print("Processing transition temperatures...")
    
    # Add the transition temperature column
    merged_df['transition_temp'] = merged_df.apply(select_target, axis=1)
    
    # Drop rows where transition_temp is NaN
    merged_df = merged_df.dropna(subset=['transition_temp'])
    
    # Drop the original temperature columns
    #columns_to_drop = ['I-N', 'N-Nx', 'Nx-Nf']
 
    columns_to_drop = ['I-N', 'N-K']
    final_df = merged_df.drop(columns=columns_to_drop)
    
    # Before returning, check for duplicates if requested
    if check_dups:
        duplicate_info = check_duplicates(final_df, mol_dict)
        print("\nDuplicate Analysis:")
        print(f"Total molecules: {duplicate_info['statistics']['total_molecules']}")
        print(f"Unique names: {duplicate_info['statistics']['unique_names']}")
        print(f"Unique SMILES: {duplicate_info['statistics']['unique_smiles']}")
        print(f"Unique structures: {duplicate_info['statistics']['unique_structures']}")
        print(f"Name duplicates: {duplicate_info['statistics']['name_duplicate_count']}")
        print(f"SMILES duplicates: {duplicate_info['statistics']['smiles_duplicate_count']}")
        print(f"Structural duplicates: {duplicate_info['statistics']['structural_duplicate_count']}")
        print(f"Duplicate indices: {duplicate_info['duplicate_indices']}")

        final_df_clean = final_df.copy()

        # Drop all remaining duplicates
        final_df_clean = final_df_clean.drop(index=duplicate_info['duplicate_indices'])

        print(f"Original dataset size: {len(final_df)}")
        print(f"Clean dataset size: {len(final_df_clean)}")
        print(f"Removed {len(final_df) - len(final_df_clean)} duplicate entries")
        # Create a clean mol_dict that only includes molecules in final_df_clean
        mol_dict_clean = {}
        for name in final_df_clean['Name_x'].unique():
            if name in mol_dict:
                mol_dict_clean[name] = mol_dict[name]

        print(f"Original mol_dict size: {len(mol_dict)}")
        print(f"Clean mol_dict size: {len(mol_dict_clean)}")
        # Save processed data
    if check_dups:
        final_df_clean.to_csv(output_file, index=False)
        print(f"Saved processed data to {output_file}")
        return final_df_clean, mol_dict_clean, duplicate_info
    else:
        final_df.to_csv(output_file, index=False)
        print(f"Saved processed data to {output_file}")
        return final_df, mol_dict, duplicate_info

def read_final_single_point_energy_with_solvation(output_file_path):
    """
    This function reads the final single point energy from a property calculation of one molecule. If this function is
    used for a chain calculation, this function reads the energy from the first job.
    This function considers a solvation effect (gas phase reaction at 1 atm dissolved at a solvent)
    Note that the molecule is still in a gas phase (additional 1.89 kcal/mol should be added for a liquid phase)
    :return: Final single point energy (kcal/mol) (ORCA - 5.0.3) -> everything is included.
    """
    # reactant
    flag = 0
    Eh_to_kcal_mol = 627.5
    with open(output_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'FINAL SINGLE POINT ENERGY' in line:
                tmp_line = line.split()
                single_point_energy = float(tmp_line[-1])  # Eh
                single_point_energy *= Eh_to_kcal_mol  # kcal/mol
                flag += 1

            if flag == 1:
                break
        f.close()

    return single_point_energy

def read_partial_charges(file_path) -> list:
    """
    This function reads the partial charges from the CHELPG section of the output file
    :return: list of tuples (index, element, partial_charge)
    """
    partial_charges = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        in_chelpg_section = False
        
        for line in lines:
            if 'CHELPG Charges' in line:
                in_chelpg_section = True
                continue
            elif in_chelpg_section:
                # Skip the separator line
                if '----' in line:
                    continue
                # Stop if we hit an empty line or new section
                if line.strip() == '' or not line.strip()[0].isdigit():
                    break
                
                # Parse the line: "  0   C   :      -0.053517"
                parts = line.split()
                if len(parts) >= 4 and parts[2] == ':':
                    index = int(parts[0])
                    element = parts[1]
                    charge = float(parts[3])
                    partial_charges.append((index, element, charge))
    
    return partial_charges

def read_electric_properties(file_path) -> list:

    """
    This function reads dipole moment, quadrupole moment, and polarizability (all of them are a.u.)
    :return: [dipole moment, quadrupole moment, polarizability]
    * quadrupole moment -> isotropic quadrupole moment
    * polarizability -> isotropic polarizability
    """
    # get electric properties
    dipole_moment = None
    quadrupole_moment = None
    polarizability = None
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line_idx, line in enumerate(lines):
            tmp_line = line.split()
            if 'Magnitude (a.u.)' in line:
                dipole_moment = float(tmp_line[-1])
            if 'Isotropic quadrupole' in line:
                quadrupole_moment = float(tmp_line[-1])
            if 'Isotropic polarizability' in line:
                polarizability = float(tmp_line[-1])
            if 'JOB NUMBER  2' in line:  # stop if moving over to the second job.
                break
        file.close()

    return [dipole_moment, quadrupole_moment, polarizability]

def read_magnitude_of_dipole_moment(output_file_path):
    """
    This functions reads a dipole moment of the output file (one molecule)
    :return: dipole_moment (a.u.)
    """
    with open(output_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Magnitude (a.u.)' in line:
                tmp_line = line.split()
                dipole_moment = float(tmp_line[-1])  # a.u.
                break

    return dipole_moment

def extract_coordinates_from_xyz(xyz_file_path):
    """
    This function extracts coordinate vectors from .xyz file
    :return: np.array[N, 3] where N is the number of atoms
    """
    coordinate_ls = list()
    with open(xyz_file_path, 'r') as file:
        lines = file.readlines()
        for line_idx, line in enumerate(lines):
            if line_idx >= 2:  # coordinate information
                tmp_line = line.split()
                coordinate_ls.append([float(tmp_line[1]), float(tmp_line[2]), float(tmp_line[3])])

    return np.array(coordinate_ls)

def prepare_mol_with_xyz(geometry_xyz_path, mol_file_path):  # function to obtain a mol object
    """
    This function obtains a mol object with a modified xyz with geometry_xyz_path
    :return: mol object
    """
    mol_to_calculate = Chem.MolFromMolFile(mol_file_path, removeHs=False)  # create a mol object with explicit hydrogen
    conformer = mol_to_calculate.GetConformer(id=0)  # only one conformer / # print(conformer.GetPositions())
    xyz_to_use = extract_coordinates_from_xyz(geometry_xyz_path)  # load geometry to use
    for atom_idx in range(mol_to_calculate.GetNumAtoms()):  # update coordinates
        x, y, z = xyz_to_use[atom_idx]
        conformer.SetAtomPosition(atom_idx, Point3D(x, y, z))

    return mol_to_calculate

def mol_from_xyz_file(xyz_path, mol_ref_path, output_path, print_bond_info=False):
    """
    This function writes a mol file with the geometry of conformers from xyz_path
    and the bonds from mol_ref_path, preserving all atom properties from the reference
    :return: mol file
    """
    ref_mol = Chem.MolFromMolFile(mol_ref_path, removeHs=False)
    xyz_mol = Chem.MolFromXYZFile(str(xyz_path))
    
    # Create editable mol
    rwmol = Chem.RWMol(xyz_mol)
    
    # First, remove any existing bonds
    for bond in xyz_mol.GetBonds():
        rwmol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    
    # Copy ALL atom properties from reference
    for i, ref_atom in enumerate(ref_mol.GetAtoms()):
        if i < rwmol.GetNumAtoms():
            edit_atom = rwmol.GetAtomWithIdx(i)
            
            # Copy basic atom properties
            edit_atom.SetFormalCharge(ref_atom.GetFormalCharge())
            edit_atom.SetNoImplicit(True)
            edit_atom.SetNumExplicitHs(ref_atom.GetNumExplicitHs())
            edit_atom.SetIsAromatic(ref_atom.GetIsAromatic())
            edit_atom.SetChiralTag(ref_atom.GetChiralTag())
            edit_atom.SetHybridization(ref_atom.GetHybridization())
            edit_atom.SetNumRadicalElectrons(ref_atom.GetNumRadicalElectrons())
    
            # Copy MOL file valence if present
            if ref_atom.HasProp('_MolFileValence'):
                edit_atom.SetProp('_MolFileValence', ref_atom.GetProp('_MolFileValence'))
    
            # Safely copy atom properties from MOL file
            for prop_key in ref_atom.GetPropNames():
                try:
                    if ref_atom.HasProp(prop_key):
                        prop_val = ref_atom.GetProp(prop_key)
                        edit_atom.SetProp(prop_key, str(prop_val))  # Convert to string to be safe
                except Exception as e:
                    print(f"Warning: Could not copy property {prop_key} for atom {i}: {e}")
    
    # Rest of the function remains the same
    aromatic_atoms = {atom.GetIdx() for atom in ref_mol.GetAtoms() if atom.GetIsAromatic()}
    
    for bond in ref_mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        if begin_idx in aromatic_atoms and end_idx in aromatic_atoms:
            rwmol.AddBond(begin_idx, end_idx, Chem.BondType.AROMATIC)
        else:
            rwmol.AddBond(begin_idx, end_idx, bond.GetBondType())
    
    try:
        mol = rwmol.GetMol()
        Chem.SanitizeMol(mol, 
                        sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE^Chem.SANITIZE_SETAROMATICITY)
        
        # Preserve atom properties when writing MOL file
        Chem.MolToMolFile(mol, output_path, kekulize=False)
        
        if print_bond_info:
            print("\nAtom and Bond Information:")
            for atom in mol.GetAtoms():
                print(f"\nAtom {atom.GetIdx()} ({atom.GetSymbol()}):")
                print(f"Formal charge: {atom.GetFormalCharge()}")
                print(f"Explicit Hs: {atom.GetNumExplicitHs()}")
                print(f"Valence: {atom.GetTotalValence()}")
                print(f"MOL file valence: {atom.GetProp('_MolFileValence') if atom.HasProp('_MolFileValence') else 'not set'}")
                print(f"Hybridization: {atom.GetHybridization()}")
            
            for bond in mol.GetBonds():
                print(f"\nBond {bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()}: "
                      f"Type={bond.GetBondType().name}, "
                      f"Aromatic={bond.GetIsAromatic()}")
        
        return mol
        
    except Exception as e:
        print(f"Error during sanitization: {e}")
        return rwmol.GetMol()

def calculate_rmsd_between_two_molecules(geometry_xyz_path_1, geometry_xyz_path_2, mol_file_path):
    """
    This function calculates the RMSD between two molecules considering symmetry
    :return: RMSD (angstorms)
    """
    # obtain mol objects -- ConfId 0 has been modified
    mol_1 = prepare_mol_with_xyz(geometry_xyz_path=geometry_xyz_path_1, mol_file_path=mol_file_path)
    mol_2 = prepare_mol_with_xyz(geometry_xyz_path=geometry_xyz_path_2, mol_file_path=mol_file_path)

    # calculate rmsd
    rmsd = AlignMol(prbMol=mol_1, refMol=mol_2, prbCid=0, refCid=0)  # AlignMol -> doesn't change the atom order (not considering permutation). The geometry of a probmoleule is changed. The geometry of a "refMol" is not changed.

    return rmsd

def extract_energy_xtb(xtb_geometry_output_path):
    """
    This function extracts the energy from the XTB geometry optimization output file
    :return: Energy (kcal/mol)
    """
    xtb_contents = open(xtb_geometry_output_path).read()
    re_search = re.search('TOTAL ENERGY[\\s]+([-0-9.]+) Eh', xtb_contents, re.DOTALL)
    energy = float(re_search.group(1))  # Eh
    energy *= 627.5  # kcal/mol

    return energy

def select_conformers(list_of_geometry_xyz_path, list_of_geometry_output_path, mol_file_path, max_num_conformers=5, energy_window=6.0, energy_threshold=0.1, rmsd_threshold=0.125, rotatation_constant_threshold=15.0):
    """
    This functions selects conformers from the given list of geometries with the criteria of (1) energy window, (2) energy threshold,
    (3) rmsd_threshold, and (4) rotational constant threshold. This threshold values are from the CREST paper: https://pubs.rsc.org/en/content/articlelanding/2020/CP/C9CP06869D (Pracht, P.; Bohle, F.; Grimme, S. Automated Exploration of the Low-Energy Chemical Space with Fast Quantum Chemical Methods. Phys. Chem. Chem. Phys. 2020)

    :param list_of_geometry_xyz_path: geometry list of conformers to be selected
    :param list_of_geometry_output_path: geometry output list of conformers to be selected. The order should be matched with "list_of_geometry_xyz_path"
    :param mol_file_path: the reference mol file to be used for RMSD calculations by replacing xyz coordinates
    :param max_num_conformers: the maximum number of conformers to be selected
    :param energy_window: the relative energy (kcal/mol) from the lowest conformer energy. Default: 6.0 kcal/mol
    :param energy_threshold: the energy threshold (kcal/mol) to be used to remove duplicates (Fig. 3 from the paper). Default: 0.1 kcal/mol
    :param rmsd_threshold: the rmsd threshold to be used to remove duplicates (Fig. 3 from the paper). Default: 0.125 angstroms
    :param rotatation_constant_threshold: the rotation constant threshold (MHz) to be used to remove duplicates (Fig. 3 from the paper). Default: 15.0 MHz
    B = (h_bar)**2 / 2I
    B_bar (rotational constant) = B / hc
    The output of XTB2 sorts the rotational constant from large to small
    :return: the list of selected conformer idx (maximum length: num_conformers). (energy has the ascending order)
    Note: This is index of list_of_geometry_xyz_path and list_of_geometry_output_path, not "conformer idx"
    """
    # choose the lowest energy conformer
    energy_arr = np.array([extract_energy_xtb(geometry_output) for geometry_output in list_of_geometry_output_path])
    sorted_idx = np.argsort(energy_arr)  # ascending order
    
    # filter geometry based on the energy window
    max_energy_allowed = energy_arr[sorted_idx[0]] + energy_window
    filtered_sorted_idx_list = sorted_idx[np.where(energy_arr[sorted_idx] <= max_energy_allowed)[0]]

    # pairwise comparison to drop duplicates
    selected_idx_list = [filtered_sorted_idx_list[0]]  # start with the lowest energy conformer
    for candidate_idx in filtered_sorted_idx_list[1:]:
        flag = 1  # assume to be added
        for selected_idx in selected_idx_list:  # pairwise comparison
            # apply energy threshold
            cond_1 = energy_arr[candidate_idx] - energy_arr[selected_idx] <= energy_threshold

            # apply rmsd threshold
            rmsd = calculate_rmsd_between_two_molecules(geometry_xyz_path_1=list_of_geometry_xyz_path[candidate_idx], geometry_xyz_path_2=list_of_geometry_xyz_path[selected_idx], mol_file_path=mol_file_path)
            cond_2 = rmsd <= rmsd_threshold

            # apply rotational constants threshold -- to differentiate between conformer and rotamer
            # candidate_rotational_constants = extract_rotational_constants_xtb(list_of_geometry_output_path[candidate_idx])
            # selected_rotational_constants = extract_rotational_constants_xtb(list_of_geometry_output_path[selected_idx])
            # cond_3 = len(np.where(np.abs(candidate_rotational_constants - selected_rotational_constants) <= rotatation_constant_threshold)[0]) == 3  # all values are less than the threshold

            # decide
            # if cond_1 & cond_2 & cond_3:  # energy <= threshold & rmsd <= threshold & rotational constant <= threshold
            if cond_1 & cond_2:  # energy <= threshold & rmsd <= threshold
                flag = 0
                break  # no need for next iterations

        # update
        if flag == 1:
            selected_idx_list.append(candidate_idx)

        # check num_conformers
        if len(selected_idx_list) == max_num_conformers:
            break

    return selected_idx_list


def position_moments(df, x_col, y_col, ddof=0, eps=1e-6):
    """
    Calculate the weighted moments of eigenvalues (λ) 
    weighted by spectral power.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing eigenvalues and their powers.
    x_col : str
        Column name for eigenvalues lambda.
    y_col : str
        Column name for spectral power (weights).
    ddof : int, default=0
        Delta degrees of freedom for variance calculation.
    eps : float, default=1e-6
        Small value to prevent division by zero.

    Returns
    -------
    tuple
        (mean, variance, skewness) of λ weighted by power.
    """
    x = df[x_col].to_numpy(dtype=float)
    w = df[y_col].to_numpy(dtype=float)

    if len(x) == 0 or len(w) == 0 or len(x) != len(w):
        return np.nan, np.nan, np.nan
    if np.any(np.isnan(x)) or np.any(np.isnan(w)):
        return np.nan, np.nan, np.nan
    if np.any(np.isinf(x)) or np.any(np.isinf(w)):
        return np.nan, np.nan, np.nan
    if not np.any(w):
        return np.nan, np.nan, np.nan

    w = w / w.sum()
    mean = np.sum(w * x)
    c = x - mean
    var = np.sum(w * c**2)
    if ddof != 0:
        var *= len(x) / (len(x) - ddof)
    skew = 0.0 if var < eps else np.sum(w * c**3) / (var**1.5)

    return mean, var, skew

def shape_moments(df, y_col, ddof=0, eps=1e-6):
    """
    Calculate the distribution moments of spectral power coefficients.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing spectral powers.
    y_col : str
        Column name for spectral power coefficients (non-negative).
    ddof : int, default=0
        Delta degrees of freedom for variance calculation.
    eps : float, default=1e-6
        Small value to prevent division by zero.

    Returns
    -------
    tuple
        (mean, variance, skewness) of normalized spectral power distribution.
    """
    s = df[y_col].to_numpy(dtype=float)

    if len(s) == 0:
        return np.nan, np.nan, np.nan
    if np.any(np.isnan(s)) or np.any(np.isinf(s)):
        return np.nan, np.nan, np.nan
    if np.all(s == 0):
        return np.nan, np.nan, np.nan

    # normalize to a probability distribution
    p = s / s.sum()

    mean = p.mean()
    c = p - mean
    var = p.var(ddof=ddof)
    skew = 0.0 if var < eps else np.mean(c**3) / (var**1.5)

    return mean, var, skew

def shape_compact_descriptors(df, y_col, *, K_top=10, eps=1e-15, return_series=True):
    """
    Compact descriptors of the normalized spectral power distribution p_k.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing spectral powers.
    y_col : str
        Column name with non-negative spectral power coefficients (e.g., |û_k|^2).
    K_top : int, default=10
        Number of largest coefficients to accumulate for 'topk'.
    eps : float, default=1e-15
        Small value to stabilize logs/divisions.
    return_series : bool, default=True
        If True, return a pandas.Series; otherwise return a dict.

    Returns
    -------
    pandas.Series or dict with keys:
        - hhi: sum(p^2)  (Herfindahl–Hirschman index; higher = more concentrated)
        - pr:  1 / hhi   (Participation ratio; higher = more spread)
        - entropy: -sum(p log p)  (Shannon entropy, nats)
        - renyi2: -log(sum(p^2))  (Rényi entropy of order 2, nats)
        - topk: sum of largest-K p's
        - gini: 1 - 2 * sum_i (cumulative_p_i) / n + 1/n  (inequality / sparsity)
    """
    s = df[y_col].to_numpy(dtype=float)
    if s.size == 0 or np.any(np.isnan(s)) or np.any(np.isinf(s)) or np.all(s <= 0):
        out = dict(hhi=np.nan, pr=np.nan, entropy=np.nan, renyi2=np.nan, topk=np.nan, gini=np.nan)
        return pd.Series(out) if return_series else out

    # Normalize to probability distribution
    p = np.clip(s, 0.0, None)
    p = p / p.sum()

    # HHI & PR
    hhi = float(np.sum(p**2))
    pr = float(1.0 / (hhi + eps))

    # Entropies
    entropy = float(-np.sum(p * np.log(p + eps)))
    renyi2 = float(-np.log(hhi + eps))

    # Top-K mass (cap K to n)
    n = p.size
    K = int(min(max(1, K_top), n))
    topk = float(np.sort(p)[-K:].sum())

    # Gini coefficient (0 = equal, 1 = very unequal)
    # Sort ascending, compute cumulative, then standard discrete Gini
    p_sorted = np.sort(p)
    cum = np.cumsum(p_sorted)
    gini = float(1.0 - 2.0 * np.sum(cum) / (n * 1.0) + 1.0 / n)

    out = dict(hhi=hhi, pr=pr, entropy=entropy, renyi2=renyi2, topk=topk, gini=gini)
    return pd.Series(out) if return_series else out

def magnitude_descriptors(df, y_col):
    """
    Calculate the magnitude descriptors of the given column.
    """
    X = df[y_col].to_numpy(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    area = float(X.sum())
    energy = float((X**2).sum())
    p95 = float(np.percentile(X, 95))
    return area, energy, p95

def spectrum_abs_descriptors(df, y_col):
    """
    Calculate the absolute descriptors of the given column.
    """
    eps = 1e-12

    y = df[y_col].to_numpy(float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    mean = y.mean()
    std = y.std()    
    skew = stats.skew(y, bias=False)
    p95 = np.percentile(y, 95)
    max_val = y.max()
    peak_to_mean = max_val/(mean+eps)
    
    return mean, std, skew, p95, max_val, peak_to_mean
 

def boltzmann_distribution(mol_dir):
    """
    Calculate the Boltzmann distribution of DFT properties.
    
    Parameters:
    -----------
    mol_dir : str
        Directory containing the molecule's conformer data
        
    Returns:
    --------
    tuple
        A tuple containing:
        - weighted_sigma_profile (pd.DataFrame): Boltzmann-weighted sigma profile
        - weighted_spectrum_df (pd.DataFrame): Boltzmann-weighted spectrum profile  
        - dft_avg_results (pd.DataFrame): DataFrame containing Boltzmann-averaged properties and statistics
        
    Raises:
    -------
    FileNotFoundError
        If required files or directories are missing
    ValueError
        If data is invalid or calculations fail
    """
    # Constants
    temperature = 298.15  # K
    Boltzmann_constant = 1.38 * 10 ** (-23)  # J/K
    joule_to_kcal_mol = 6.02 / 4184 * 10 ** 23
    beta = 1 / (Boltzmann_constant * temperature * joule_to_kcal_mol)  # 1/(kcal/mol)
    
    # Check if parent directory exists
    if not os.path.exists(mol_dir):
        raise FileNotFoundError(f"Parent directory not found: {mol_dir}")
    
    # Get all conformer directories and extract their numbers
    conformer_dirs = [d for d in os.listdir(mol_dir) 
                     if d.startswith('conformer') and os.path.isdir(os.path.join(mol_dir, d))]
    if not conformer_dirs:
        raise ValueError(f"No conformer directories found in {mol_dir}")
        
    # Extract conformer numbers and find min/max
    conformer_numbers = []
    for d in conformer_dirs:
        try:
            num = int(d.replace('conformer', ''))
            conformer_numbers.append(num)
        except ValueError:
            continue
            
    if not conformer_numbers:
        raise ValueError("No valid conformer numbers found in directory names")
        
    min_conf = min(conformer_numbers)
    max_conf = max(conformer_numbers)
    print(f"Found conformers from {min_conf} to {max_conf}")
    
    # Initialize dictionary to store results
    results_dict = {}
    # Collect energy data
    energy_list = []
    sigma_profile_list = []  # list of tuples (conformer_idx, sigma_profile_df)
    spectrum_list = []
    dft_results_list = []  # Store DFT results as list to maintain order
    valid_conformers = []

    for conf_dir in sorted(conformer_dirs):
        conf_path = os.path.join(mol_dir, conf_dir)
        dft_file = os.path.join(conf_path, 'dft_results.csv')
        sigma_file = os.path.join(conf_path, 'sigma_cosmo.out')
        spectrum_file = os.path.join(conf_path, 'spectrum.csv')

        if not os.path.exists(dft_file):
            print(f"Warning: DFT results file not found: {dft_file}, skipping...")
            continue
            
        try:
            tmp_results = pd.read_csv(dft_file)
            
            # Validate required columns
            required_cols = ['dipole_moment', 
                           'quadrupole_moment', 'polarizability']
            missing_cols = [col for col in required_cols if col not in tmp_results.columns]
            if missing_cols:
                print(f"Warning: Missing required columns in {dft_file}: {missing_cols}, skipping...")
                continue
            
            # Get energy
            if 'energy (kcal/mol)' in tmp_results.columns:
                energy = tmp_results['energy (kcal/mol)'].values[0]
            else:
                print(f"Warning: No energy column found in {dft_file}, skipping...")
                continue

                
            energy_list.append(energy)
            valid_conformers.append(conf_dir)
            current_conf_idx = len(energy_list) - 1
            
            # Store DFT results for this conformer (do this early to maintain consistency)
            dft_results_list.append(tmp_results)
            
            # Get sigma profile
            if os.path.exists(sigma_file):
                sigma_profile = pd.read_csv(sigma_file, comment='#', header=None, index_col=False, skiprows=1, sep=r'\s+', names=['sigma', 'pA'])
                sigma_profile_list.append((current_conf_idx, sigma_profile))
            else:
                print(f"Warning: Sigma profile file not found: {sigma_file}")

            # Get spectrum profile
            spectrum_path = os.path.join(conf_path, spectrum_file)
            if os.path.exists(spectrum_path):
                spectrum_profile = pd.read_csv(spectrum_path)
                spectrum_list.append((current_conf_idx, spectrum_profile))
            else:
                print(f"Warning: Spectrum profile file not found: {spectrum_path}")
                
        except Exception as e:
            print(f"Warning: Error processing {dft_file}: {str(e)}, skipping...")
            continue
    
    if not energy_list:
        raise ValueError("No valid DFT results found in any conformer directory")
    
    # Create dft_results DataFrame from the list (only conformers with valid energy)
    dft_results = pd.concat(dft_results_list, ignore_index=True)
    
    # Validate that we have the same number of conformers for energy and DFT results
    if len(energy_list) != len(dft_results_list):
        raise ValueError(f"Mismatch between energy data ({len(energy_list)}) and DFT results ({len(dft_results_list)})")
    
    # Calculate Boltzmann weights
    energy_arr = np.array(energy_list)
    
    # Mean shift for numerical stability
    energy_arr -= energy_arr.mean() 
        
    boltzmann_weight_arr = np.exp(-beta * energy_arr)
    weight_sum = boltzmann_weight_arr.sum()
    
    if weight_sum == 0:
        raise ValueError("All Boltzmann weights are zero")
        
    boltzmann_weight_arr /= weight_sum
    
    print(f"Successfully processed {len(valid_conformers)} out of {len(conformer_dirs)} conformers")
    print("Valid conformers:", valid_conformers)
    print(f"Energy array shape: {energy_arr.shape}")
    print(f"Boltzmann weights shape: {boltzmann_weight_arr.shape}")
    print(f"DFT results shape: {dft_results.shape}")
    
    # Calculate Boltzmann-weighted sigma profile if we have sigma profiles
    if sigma_profile_list:
        if len(sigma_profile_list) == 0:
            raise ValueError("No sigma profiles found in the list")
            
        # Ensure all profiles have the same sigma values
        sigma_values = sigma_profile_list[0][1]['sigma'].values
        for i, (_, profile) in enumerate(sigma_profile_list[1:], 1):
            if not np.array_equal(profile['sigma'].values, sigma_values):
                raise ValueError(f"Sigma profile {i} has different sigma values than the first profile")
            
            # Check for NaN or infinite values
            if profile['pA'].isna().any() or np.isinf(profile['pA']).any():
                raise ValueError(f"Sigma profile {i} contains NaN or infinite values")
        
        # Calculate weighted average of pA values
        weighted_pA = np.zeros_like(sigma_values)
        # gather subset weights corresponding to available sigma profiles
        for i, profile in enumerate(sigma_profile_list):
            weighted_pA += profile[1]['pA'].values * boltzmann_weight_arr[i]

        # Create DataFrame for weighted sigma profile
        weighted_sigma_profile = pd.DataFrame({
            'sigma': sigma_values,
            'pA': weighted_pA
        })
        
        # Verify the weighted profile
        if weighted_sigma_profile['pA'].isna().any() or np.isinf(weighted_sigma_profile['pA']).any():
            raise ValueError("Weighted sigma profile contains NaN or infinite values")
       
        # Calculate the areas
        try:
            areas = sigma_binned_area(weighted_sigma_profile)
        except NameError:
            raise ValueError("sigma_binned_area function is not defined or imported")
        except Exception as e:
            raise ValueError(f"Error calculating sigma binned areas: {str(e)}")

        # Calculate statistical moments for the averaged profile
        mean, variance, skew = position_moments(weighted_sigma_profile,x_col="sigma",y_col="pA")
            # Add to results dictionary
        results_dict['sigma_boltzmann_pos_mean'] = [mean]
        results_dict['sigma_boltzmann_pos_std'] = [np.sqrt(variance)]
        results_dict['sigma_boltzmann_pos_skewness'] = [skew]

        mean, variance, skew = shape_moments(weighted_sigma_profile,y_col="pA")
        results_dict['sigma_boltzmann_shape_mean'] = [mean]
        results_dict['sigma_boltzmann_shape_std'] = [np.sqrt(variance)]
        results_dict['sigma_boltzmann_shape_skewness'] = [skew]

        desc = shape_compact_descriptors(weighted_sigma_profile, y_col="pA", K_top=10)
        results_dict['sigma_boltzmann_shapecompact_hhi'] = [desc['hhi']]
        results_dict['sigma_boltzmann_shapecompact_pr'] = [desc['pr']]
        results_dict['sigma_boltzmann_shapecompact_entropy'] = [desc['entropy']]
        results_dict['sigma_boltzmann_shapecompact_renyi2'] = [desc['renyi2']]
        results_dict['sigma_boltzmann_shapecompact_topk'] = [desc['topk']]
        results_dict['sigma_boltzmann_shapecompact_gini'] = [desc['gini']]

        area, energy, p95 = magnitude_descriptors(weighted_sigma_profile, y_col="pA")
        results_dict['sigma_boltzmann_magnitude_area'] = [area]
        results_dict['sigma_boltzmann_magnitude_energy'] = [energy]
        results_dict['sigma_boltzmann_magnitude_p95'] = [p95]

        results_dict['sigma_HBD_fraction'] = [areas['HBD_fraction']]
        results_dict['sigma_HBA_fraction'] = [areas['HBA_fraction']]
        results_dict['sigma_NP_fraction'] = [areas['NP_fraction']]
        results_dict['sigma_total_area'] = [areas['total_area']]
    
    # Calculate spectrum profile
    if spectrum_list:
        if len(spectrum_list) == 0:
            raise ValueError("No spectrum profiles found in the list")
    
        # Ensure all profiles have the same eigenvalues
        eigenvalues = spectrum_list[0][1]['eigenvalues'].values
        for i, profile in enumerate(spectrum_list[1:], 1):
            if not np.array_equal(profile[1]['eigenvalues'].values, eigenvalues):
                raise ValueError(f"Spectrum profile {i} has different eigenvalues than the first profile")
            # Check for NaN or infinite values
            if profile[1]['squared_coeffs'].isna().any() or np.isinf(profile[1]['squared_coeffs']).any():
                raise ValueError(f"Spectrum profile {i} contains NaN or infinite values")
        
        # Calculate weighted average of squared coefficients
        weighted_squared_coeffs = np.zeros_like(eigenvalues)
        for i, profile in enumerate(spectrum_list):
            weighted_squared_coeffs += profile[1]['squared_coeffs'].values * boltzmann_weight_arr[i]

        # Create DataFrame for weighted spectrum profile
        weighted_spectrum_df = pd.DataFrame({
            'eigenvalues': eigenvalues,
            'squared_coeffs': weighted_squared_coeffs
        })

      
        # Verify the weighted profile
        if weighted_spectrum_df['squared_coeffs'].isna().any() or np.isinf(weighted_spectrum_df['squared_coeffs']).any():
            raise ValueError("Weighted spectrum profile contains NaN or infinite values")

        mean, std, skew, p95, max_val, peak_to_mean = spectrum_abs_descriptors(weighted_spectrum_df, y_col="squared_coeffs")
        results_dict['spectrum_boltzmann_abs_mean'] = [mean]
        results_dict['spectrum_boltzmann_abs_std'] = [std]
        results_dict['spectrum_boltzmann_abs_skewness'] = [skew]
        results_dict['spectrum_boltzmann_abs_p95'] = [p95]
        results_dict['spectrum_boltzmann_abs_max_val'] = [max_val]
        results_dict['spectrum_boltzmann_abs_peak_to_mean'] = [peak_to_mean]

        mean, variance, skew = position_moments(weighted_spectrum_df, x_col="eigenvalues", y_col="squared_coeffs") 
        # Add to results dictionary
        results_dict['spectrum_boltzmann_pos_mean'] = [mean]
        results_dict['spectrum_boltzmann_pos_std'] = [np.sqrt(variance)]
        results_dict['spectrum_boltzmann_pos_skewness'] = [skew]

        mean, variance, skew = shape_moments(weighted_spectrum_df,y_col="squared_coeffs")
        results_dict['spectrum_boltzmann_shape_mean'] = [mean]
        results_dict['spectrum_boltzmann_shape_std'] = [np.sqrt(variance)]
        results_dict['spectrum_boltzmann_shape_skewness'] = [skew]

        desc = shape_compact_descriptors(weighted_spectrum_df, y_col="squared_coeffs", K_top=10)
        results_dict['spectrum_boltzmann_shapecompact_hhi'] = [desc['hhi']]
        results_dict['spectrum_boltzmann_shapecompact_pr'] = [desc['pr']]
        results_dict['spectrum_boltzmann_shapecompact_entropy'] = [desc['entropy']]
        results_dict['spectrum_boltzmann_shapecompact_renyi2'] = [desc['renyi2']]
        results_dict['spectrum_boltzmann_shapecompact_topk'] = [desc['topk']]
        results_dict['spectrum_boltzmann_shapecompact_gini'] = [desc['gini']]

        area, energy, p95 = magnitude_descriptors(weighted_spectrum_df, y_col="squared_coeffs")
        results_dict['spectrum_boltzmann_magnitude_area'] = [area]
        results_dict['spectrum_boltzmann_magnitude_energy'] = [energy]
        results_dict['spectrum_boltzmann_magnitude_p95'] = [p95]

    # Calculate properties
    dft_columns = ['energy (kcal/mol)', 'dipole_moment', 
                  'quadrupole_moment', 'polarizability']
    
    for col in dft_columns:
        values = dft_results[col].to_numpy()
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid values found in column {col}")
            
        # Boltzmann statistics
        boltzmann_average = np.average(values, weights=boltzmann_weight_arr)
        boltzmann_variance = np.average(
            (values - boltzmann_average) ** 2,
            weights=boltzmann_weight_arr
        )
        
        # Regular statistics
        mean = np.mean(values)
        variance = np.var(values)
        
        # Store results in dictionary
        results_dict[f'{col}_Boltzmann_average'] = [boltzmann_average]
        results_dict[f'{col}_Boltzmann_std'] = [np.sqrt(boltzmann_variance)]
        results_dict[f'{col}_Boltzmann_skewness'] = [np.average((values - boltzmann_average) ** 3, weights=boltzmann_weight_arr) / np.sqrt(boltzmann_variance) ** 3]
        results_dict[f'{col}_mean'] = [mean]
        results_dict[f'{col}_std'] = [np.sqrt(variance)]
        results_dict[f'{col}_skewness'] = [np.mean((values - mean) ** 3) / np.sqrt(variance) ** 3]
        
        # print(f'{col} Boltzmann average: {boltzmann_average} kcal/mol')
        # print(f'{col} Boltzmann std: {np.sqrt(boltzmann_variance)} kcal/mol')
        # print(f'{col} Mean: {mean} kcal/mol')
        # print(f'{col} Std: {np.sqrt(variance)} kcal/mol')
        # print('--------------------------------')
    
    # Create DataFrame from dictionary
    dft_avg_results = pd.DataFrame(results_dict)

    # Return None for profiles that weren't calculated
    if not sigma_profile_list:
        weighted_sigma_profile = None
    if not spectrum_list:
        weighted_spectrum_df = None

    return weighted_sigma_profile, weighted_spectrum_df, dft_avg_results

def sigma_binned_area(sigma_profile):
    """
    Calculate the normalized areas for different regions of the sigma profile.
    
    Parameters:
    -----------
    sigma_profile : pd.DataFrame
        DataFrame containing sigma values and their corresponding pA values
        Must have columns 'sigma' and 'pA'
        
    Returns:
    --------
    dict
        Dictionary containing normalized areas for:
        - HBD (Hydrogen-Bond Donor): σ < -0.0084 e/Å²
        - HBA (Hydrogen-Bond Acceptor): σ > +0.0084 e/Å²
        - NP (Non-Polar): -0.0084 e/Å² < σ < +0.0084 e/Å²
    """
    # Define the threshold
    threshold = 0.0084  # e/Å²
    
    # Calculate total area
    total_area = sigma_profile['pA'].sum()
    
    # Calculate areas for each region
    hbd_area = sigma_profile[sigma_profile['sigma'] < -threshold]['pA'].sum()
    hba_area = sigma_profile[sigma_profile['sigma'] > threshold]['pA'].sum()
    np_area = sigma_profile[(sigma_profile['sigma'] >= -threshold) & 
                           (sigma_profile['sigma'] <= threshold)]['pA'].sum()
    
    # Normalize by total area
    hbd_fraction = hbd_area / total_area
    hba_fraction = hba_area / total_area
    np_fraction = np_area / total_area
    
    return {
        'HBD_fraction': hbd_fraction,
        'HBA_fraction': hba_fraction,
        'NP_fraction': np_fraction,
        'HBD_area': hbd_area,
        'HBA_area': hba_area,
        'NP_area': np_area,
        'total_area': total_area
    }

def resample_profile( 
    df : pd.DataFrame,
    n_points: int = 60
)-> pd.DataFrame:
    """
    Interpolate a profile to a given number of points.
    """
    cols = df.columns.tolist()
    xx = df[cols[0]].to_numpy()
    yy = df[cols[1]].to_numpy()
   
    x_new = np.linspace(xx.min(), xx.max(), num=n_points)
    
    xmin = x_new.min()
    xmax = x_new.max()
    y_new = np.interp(x_new, xx, yy)   
    
    return xmin, xmax, y_new

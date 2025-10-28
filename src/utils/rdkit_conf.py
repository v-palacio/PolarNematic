from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
import numpy as np

def generate_conformers(mol, n_conf=15, random_seed=42):
    """
    Generate conformers using RDKit
    
    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object
    n_conf : int
        Number of conformers to generate
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    mol : rdkit.Chem.rdchem.Mol
        Molecule with conformers
    """
    if mol is None:
        raise ValueError("Input molecule is None")
    
    # Add hydrogens if they're not present
    mol = Chem.AddHs(mol)
    if mol is None:
        raise ValueError("Failed to add hydrogens to molecule")
    
    # Generate conformers
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0  # Use all available CPUs
    params.enforceChirality = True
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True
    
    # Generate conformers
    n_generated = AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=params)
    if n_generated == -1:
        raise ValueError("Conformer generation failed")
    print(f"Generated {n_generated} conformers")
    
    return mol

def optimize_conformers(mol, max_iters=2500):
    """
    Optimize conformers using UFF
    
    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule with conformers
    max_iters : int
        Maximum number of optimization iterations
    
    Returns:
    --------
    energies : list
        List of final energies for each conformer
    """
    if mol is None:
        raise ValueError("Input molecule is None")
    
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformers to optimize")
    
    energies = []
    
    # Optimize each conformer
    for conf_id in range(mol.GetNumConformers()):
        try:
            # Optimize using UFF
            converged = AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=max_iters)
            
            # Get the energy
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            if ff is None:
                raise ValueError(f"Failed to create force field for conformer {conf_id}")
            
            energy = ff.CalcEnergy()
            if not np.isfinite(energy):
                raise ValueError(f"Invalid energy value for conformer {conf_id}")
            
            energies.append(energy)
            print(f"Conformer {conf_id}: Energy = {energy:.2f}, Converged = {converged}")
            
        except Exception as e:
            print(f"Error optimizing conformer {conf_id}: {str(e)}")
            energies.append(float('inf'))  # Mark failed conformers with infinite energy
    
    return energies

def save_conformers_xyz(mol, energies, filename):
    """
    Save conformers in XYZ format
    
    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule with conformers
    energies : list
        List of energies for each conformer
    filename : str or Path
        Output filename
    """
    if mol is None:
        raise ValueError("Input molecule is None")
    
    if len(energies) != mol.GetNumConformers():
        raise ValueError("Number of energies does not match number of conformers")
    
    try:
        with open(filename, 'w') as f:
            for conf_id in range(mol.GetNumConformers()):
                conf = mol.GetConformer(conf_id)
                
                # Write number of atoms
                f.write(f"{mol.GetNumAtoms()}\n")
                
                # Write energy
                f.write(f"{energies[conf_id]:.6f}\n")
                
                # Write coordinates
                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    symbol = mol.GetAtomWithIdx(i).GetSymbol()
                    f.write(f"{symbol:2s} {pos.x:10.6f} {pos.y:10.6f} {pos.z:10.6f}\n")
                    
    except IOError as e:
        raise IOError(f"Failed to write conformers to file {filename}: {str(e)}")

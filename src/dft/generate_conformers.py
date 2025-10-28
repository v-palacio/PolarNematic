#!/usr/bin/env python3
"""
Conformer Generation Pipeline

This script generates conformers for molecules using RDKit and prepares them
for DFT calculations with ORCA.
"""

import sys
import os
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.molecular_descriptors import get_additional_descriptors
from rdkit_conf import generate_conformers, optimize_conformers, save_conformers_xyz

def generate_conformer_ensemble(mol_file, output_dir, n_conformers=15, random_seed=42):
    """Create an RDKit conformer ensemble and basic stats; write XYZ file."""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load molecule
    mol = Chem.MolFromMolFile(str(mol_file))
    if mol is None:
        raise ValueError(f"Could not load molecule from {mol_file}")
    
    print(f"Processing {mol_file.name}")
    
    # Generate conformers
    mol_with_confs = generate_conformers(mol, n_conformers, random_seed)
    
    # Optimize conformers
    energies = optimize_conformers(mol_with_confs)
    
    # Filter out failed conformers (infinite energy)
    valid_confs = []
    valid_energies = []
    
    for i, energy in enumerate(energies):
        if np.isfinite(energy):
            valid_confs.append(i)
            valid_energies.append(energy)
    
    print(f"Valid conformers: {len(valid_confs)}/{len(energies)}")
    
    if len(valid_confs) == 0:
        raise ValueError("No valid conformers generated")
    
    # Save conformers
    conformer_file = output_dir / f"{mol_file.stem}_conformers.xyz"
    save_conformers_xyz(mol_with_confs, energies, conformer_file)
    
    # Calculate Boltzmann weights
    energies_kcal = np.array(valid_energies) * 627.509  # Convert to kcal/mol
    min_energy = np.min(energies_kcal)
    relative_energies = energies_kcal - min_energy
    
    # Boltzmann weights at 298K
    kT = 0.593  # kcal/mol at 298K
    weights = np.exp(-relative_energies / kT)
    weights = weights / np.sum(weights)
    
    # Create conformer info
    conformer_info = {
        'molecule_name': mol_file.stem,
        'n_conformers': len(valid_confs),
        'energies_kcal_mol': valid_energies,
        'boltzmann_weights': weights.tolist(),
        'conformer_file': str(conformer_file),
        'min_energy_kcal_mol': min_energy
    }
    
    return conformer_info

def process_molecule_batch(mol_dir, output_dir, n_conformers=15, random_seed=42):
    """Batch-generate conformers for all .mol files and save a summary CSV."""
    
    mol_dir = Path(mol_dir)
    output_dir = Path(output_dir)
    
    mol_files = list(mol_dir.glob("*.mol"))
    print(f"Found {len(mol_files)} MOL files")
    
    conformer_data = []
    
    for mol_file in mol_files:
        try:
            mol_output_dir = output_dir / mol_file.stem
            info = generate_conformer_ensemble(
                mol_file, mol_output_dir, n_conformers, random_seed
            )
            conformer_data.append(info)
            
        except Exception as e:
            print(f"Error processing {mol_file.name}: {str(e)}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(conformer_data)
    
    # Save summary
    summary_file = output_dir / "conformer_summary.csv"
    df.to_csv(summary_file, index=False)
    
    print(f"Processed {len(conformer_data)} molecules")
    print(f"Summary saved to {summary_file}")
    
    return df

pass  # no CLI integration

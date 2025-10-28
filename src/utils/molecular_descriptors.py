import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Fragments
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors3D
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Descriptors3D import (
    NPR1, NPR2, Asphericity, SpherocityIndex, RadiusOfGyration,
    InertialShapeFactor, PMI1, PMI2, PMI3
)
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
import os


# Additional descriptors specific to liquid crystals
def get_additional_descriptors(mol):
    return {
        # Aromatic character
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'FractionAromaticAtoms': rdMolDescriptors.CalcFractionCSP3(mol),
        
        # Flexibility indicators
        'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),

        # Polarity features
        'NumHAcceptors': rdMolDescriptors.CalcNumHBA(mol),
        'NumHDonors': rdMolDescriptors.CalcNumHBD(mol),
        'TPSA': Descriptors.TPSA(mol),
        
        # Shape descriptors
        'NumRings': rdMolDescriptors.CalcNumRings(mol),
  
        # Fragment analysis
        'NumBenzeneRings': Fragments.fr_benzene(mol),
        'NumPhenols': Fragments.fr_phenol(mol),
        'NumAlkylHalides': Fragments.fr_alkyl_halide(mol),
        'NumEthers': Fragments.fr_ether(mol),
        'NumEsters': Fragments.fr_ester(mol),
        'Dipole_Gasteiger': calculate_dipole(mol), # Estimated dipole moment in Debye
        'Dipole_MMFF': calculate_dipole_mmff(mol), # Estimated dipole moment in Debye
        'Dipole_Conformer': calculate_dipole_with_conformers(mol) # Estimated dipole moment in Debye
    }

def descriptors_3d(mol):
    return {   
        'PMI1': Descriptors3D.PMI1(mol),
        'PMI2': Descriptors3D.PMI2(mol),
        'PMI3': Descriptors3D.PMI3(mol),
        'NPR1': Descriptors3D.NPR1(mol),
        'NPR2': Descriptors3D.NPR2(mol),
        'Asphericity': Descriptors3D.Asphericity(mol),
        'Eccentricity': Descriptors3D.Eccentricity(mol),
        'InertialShapeFactor': Descriptors3D.InertialShapeFactor(mol),
        'RadiusOfGyration': Descriptors3D.RadiusOfGyration(mol),
        'SpherocityIndex': Descriptors3D.SpherocityIndex(mol)
        }

def analyze_conformers(mol, n_conformers=10):
    """
    Generate and analyze multiple conformers of a molecule.
    
    Args:
        mol: RDKit molecule
        n_conformers: Number of conformers to generate
    """
    try:      
        
        # Generate multiple conformers
        AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, randomSeed=42)
        
        # Optimize all conformers
        energies = []
        # Setup MMFF94 properties and force field
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        
        for conf_id in range(mol.GetNumConformers()):
            # Create force field with proper parameters
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=conf_id)
            if ff is None:
                print("Could not create force field")
                continue
            # Calculate energy
            energy = ff.CalcEnergy()
            energies.append(energy)
        
        if energies:  # Only calculate stats if we have energies
            conformer_stats = {
                'MinEnergy': min(energies),  # Most stable conformer
                'MaxEnergy': max(energies),  # Least stable conformer
                'EnergyRange': max(energies) - min(energies),  # Range of conformer energies
                'MeanEnergy': np.mean(energies),  # Average energy
                'StdEnergy': np.std(energies)  # Energy variation
            }
            return conformer_stats
        else:
            print("No valid conformer energies calculated")
            return None
            
    except Exception as e:
        print(f"Could not calculate conformer energies for molecule. Error: {str(e)}")
        return None
    

def calculate_dipole(mol):
    """Calculate an approximate molecular dipole moment using Gasteiger charges"""
    # Add hydrogens and generate 3D coordinates if not present
    mol = Chem.AddHs(mol)
    if not mol.GetNumConformers():
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    
    # Calculate Gasteiger charges
    AllChem.ComputeGasteigerCharges(mol)
    
    # Get the conformer
    conf = mol.GetConformer()
    
    # Initialize dipole components
    dx = dy = dz = 0.0
    
    # Calculate dipole from atomic charges and positions
    for atom in mol.GetAtoms():
        charge = float(atom.GetProp('_GasteigerCharge'))
        pos = conf.GetAtomPosition(atom.GetIdx())
        dx += charge * pos.x
        dy += charge * pos.y
        dz += charge * pos.z
    
    # Calculate total dipole moment
    dipole = np.sqrt(dx*dx + dy*dy + dz*dz)
    return dipole * 4.8  # Convert to Debye (approximate)

def calculate_dipole_mmff(mol):
    """Calculate dipole using MMFF94 charges"""
    mol = Chem.AddHs(mol)
    if not mol.GetNumConformers():
        AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Get MMFF94 charges - force field based
    mp = AllChem.MMFFGetMoleculeProperties(mol)
    if mp is None:
        return None  # MMFF parameters not available for this molecule
        
    conf = mol.GetConformer()
    dx = dy = dz = 0.0
    
    for atom in mol.GetAtoms():
        # Get MMFF94 charge from force field
        idx = atom.GetIdx()
        charge = mp.GetMMFFPartialCharge(idx)
        pos = conf.GetAtomPosition(idx)
        dx += charge * pos.x
        dy += charge * pos.y
        dz += charge * pos.z
        
    return np.sqrt(dx*dx + dy*dy + dz*dz) * 4.8

def calculate_dipole_with_conformers(mol, n_conformers=10):
    """Calculate dipole using multiple conformers and energy minimization"""
    mol = Chem.AddHs(mol)
    
    # Generate multiple conformers
    AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, randomSeed=42)
    
    # Optimize all conformers and calculate dipoles
    dipoles = []
    energies = []
    
    for conf_id in range(mol.GetNumConformers()):
        # Optimize geometry
        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
        
        # Calculate energy
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=conf_id)
        energy = ff.CalcEnergy()
        
        # Calculate dipole for this conformer
        dx = dy = dz = 0.0
        AllChem.ComputeGasteigerCharges(mol)
        conf = mol.GetConformer(conf_id)
        
        for atom in mol.GetAtoms():
            charge = float(atom.GetProp('_GasteigerCharge'))
            pos = conf.GetAtomPosition(atom.GetIdx())
            dx += charge * pos.x
            dy += charge * pos.y
            dz += charge * pos.z
            
        dipole = np.sqrt(dx*dx + dy*dy + dz*dz) * 4.8
        dipoles.append(dipole)
        energies.append(energy)
    
    # Weight dipoles by Boltzmann distribution
    energies = np.array(energies)
    weights = np.exp(-energies/(0.001987*298.15))  # RT in kcal/mol at 298.15K
    weights /= np.sum(weights)
    
    return np.average(dipoles, weights=weights)

def calculate_dipole_ensemble(mol):
    """Calculate dipole using multiple charge models and conformers"""
    gasteiger_dipole = calculate_dipole(mol)
    mmff_dipole = calculate_dipole_mmff(mol)
    conformer_dipole = calculate_dipole_with_conformers(mol)
    
    # Weight the different methods (could be adjusted based on validation)
    weights = [0.2, 0.3, 0.5]  # Example weights
    return (gasteiger_dipole * weights[0] + 
            mmff_dipole * weights[1] + 
            conformer_dipole * weights[2])

def calculate_similarities(molecules):
    """Calculate various similarity metrics between molecules"""
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    simfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096, countSimulation=True)
    
    fps = [fpgen.GetFingerprint(m) for m in molecules]
    countfps = [fpgen.GetCountFingerprint(m) for m in molecules]
    simfps = [simfpgen.GetFingerprint(m) for m in molecules]
    
    countsims = []
    sims = []
    simsims = []
    for i in range(len(molecules)):
        for j in range(i+1, len(molecules)):
            countsims.extend(DataStructs.BulkTanimotoSimilarity(countfps[i], countfps[j:]))
            sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[j:]))
            simsims.extend(DataStructs.BulkTanimotoSimilarity(simfps[i], simfps[j:]))
            
    return countsims, sims, simsims

def calculate_omg_descriptors(mol):
    """
    Calculate specific molecular descriptors from Seonghwan's paper.
    
    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object
    
    Returns:
    --------
    dict
        Dictionary of descriptor names and values
    """
    # Ensure 3D coordinates are present
    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    
    descriptors = {}
    
    try:
        # Basic molecular properties
        descriptors['molecular_weight'] = Descriptors.ExactMolWt(mol)
        descriptors['logP'] = Descriptors.MolLogP(mol)  # Octanol-water partition coefficient
        descriptors['QED'] = Descriptors.qed(mol)  # Quantitative estimate of drug-likeness
        descriptors['TPSA'] = Descriptors.TPSA(mol)  # Topological polar surface area
        
        # Structural flexibility descriptors
        descriptors['rotatable_bonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol)  # Structural flexibility
        descriptors['backbone_rotatable_bonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol, strict=True)  # Backbone flexibility
        
        # Shape descriptors
        descriptors['asphericity'] = Asphericity(mol)  # Deviation from spherical form
        descriptors['anisometry_factor'] = NPR1(mol) / NPR2(mol)  # Anisometry descriptor
        descriptors['thermal_shape_factor'] = InertialShapeFactor(mol)  # Based on principal moments
        descriptors['radius_gyration'] = RadiusOfGyration(mol)
        descriptors['sphericity_index'] = SpherocityIndex(mol)
        
        # Principal moments of inertia ratios
        descriptors['PMI1'] = PMI1(mol)
        descriptors['PMI2'] = PMI2(mol)
        descriptors['PMI3'] = PMI3(mol)
        
        # Electronic properties (if available through ORCA calculations)
        # These would typically come from quantum calculations
        descriptors['HOMO_minus1_energy'] = None  # Placeholder
        descriptors['HOMO_energy'] = None
        descriptors['LUMO_energy'] = None
        descriptors['LUMO_plus1_energy'] = None
        
        # Electromagnetic properties
        descriptors['dipole_moment'] = calculate_dipole_ensemble(mol)
        descriptors['quadrupole_moment'] = None  # Requires quantum calculation
        descriptors['polarizability'] = None  # Requires quantum calculation
        
        # Excited state properties (would come from quantum calculations)
        descriptors['lowest_singlet_energy'] = None
        descriptors['singlet_excitation_energy'] = None
        descriptors['oscillator_strength'] = None
        descriptors['lowest_triplet_energy'] = None
        
        # Flory-Huggins parameters (would need additional calculation)
        descriptors['flory_huggins_ethanol'] = None  # ε = 68.4
        descriptors['flory_huggins_chloroform'] = None  # ε = 4.9
        
    except Exception as e:
        print(f"Error calculating descriptors: {str(e)}")
        return None
    
    return descriptors

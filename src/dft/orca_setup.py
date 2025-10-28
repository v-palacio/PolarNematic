from rdkit import Chem
from rdkit.Chem import AllChem
import os
from pathlib import Path

def save_submit(filename, idx, conf):


    with open(filename, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"#$ -N ORCA-mol{idx}-conf{conf}\n")
        f.write("#$ -cwd\n") 
        f.write("#$ -o run.out\n")
        f.write("#$ -e run.err\n")
        f.write("#$ -pe orte 4\n")
        f.write("#$ -q all.q\n\n")
    
        f.write("module load openmpi/4.1.6\n")
        f.write(f"/share/apps/orca/orca_6_0_0/orca property_mol{idx}_conf{conf}.inp > property_mol{idx}_conf{conf}.out\n")

def save_xtb_submit(filename, idx, conf):

    with open(filename, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"#$ -N xtb-mol{idx}-conf{conf}\n")
        f.write("#$ -cwd\n") 
        f.write("#$ -o run.out\n")
        f.write("#$ -e run.err\n")
        f.write("#$ -pe orte 1\n")
        f.write("#$ -q all.q\n\n")
        f.write("xtb_path=/share/apps/orca/orca_6_0_0/otool_xtb\n")

        f.write(f"$xtb_path ff_optimized.xyz --opt tight -P 1 --namespace geometry --cycles 300 > geometry.out\n\n")


def save_orca_input(dirpath, filename):

    orca_script = [
        f'! revPBE def2-SVP def2/J Hirshfeld CHELPG RIJONX D3 Energy KeepDens \n',  # approach basis aux_basis
        f'%pal\n',
        f'nprocs 4\n'
        f'end\n',
        f'%chelpg\n',
        f'DIPOLE TRUE\n',
        f'end\n',
        f'%plots\n',
        f'Format Gaussian_Cube\n',
        f'ElDens("eldens.cube");\n',
        f'end\n',
        f'%maxcore 3000\n',
        f'%elprop\n',
        f'Polar 1\n',
        f'Quadrupole True\n',
        f'end\n',
        f'* xyzfile 0 1 geometry.xtbopt.xyz\n',
    ]

    base_name = dirpath / 'property_cosmo'  # CAUTION: the same path with the geometry
       
    orca_script.extend([
            '\n',
            '# for cosmo solvent -> sigma profile\n',
            '$new_job\n',
            f'! revPBE def2-SVP def2/J CPCMC Hirshfeld CHELPG RIJONX D3 Energy\n',  # approach basis aux_basis,
            f'%base "property_cosmo"\n',
            f'%chelpg\n',
            f'DIPOLE TRUE\n',
            f'end\n',   
            f'%pal\n',
            f'nprocs 4\n'
            f'end\n',
            f'%maxcore 3000\n',
            f'* xyzfile 0 1 geometry.xtbopt.xyz\n',
            ])
    
    with open(dirpath / filename, 'w') as f:
        f.writelines(orca_script)


def save_goat_input(mol, filename):

    # Create input file for GOAT optimization
    with open(filename, 'w') as f:
        f.write(f"!GOAT xtb\n")
        f.write(f"%PAL\n")
        f.write(f" NPROCS 4\n")
        f.write(f"END\n")
        f.write(f"%GOAT\n")
        f.write(f" MAXOPTITER     64\n")
        f.write(f" MAXEN 4.0\n")
        f.write(f" GFNUPHILL    GFNFF\n")
        f.write(f"END\n")
        f.write(f"*XYZFILE 0 1 {mol}.xyz\n")

def save_xyz(mol, filename):
    """Save molecule geometry for ORCA calculation"""
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Write XYZ file
    with open(filename, 'w') as f:
        conf = mol.GetConformer()
        f.write(f"{mol.GetNumAtoms()}\n\n")
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            symbol = atom.GetSymbol()
            f.write(f"{symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")

def check_if_orca_normally_finished(output_file_path):
        with open(output_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'ORCA TERMINATED NORMALLY' in line:
                    return True
            return False
        
def parse_ensemble_info(filepath):
    ensemble_data = []
    reading_data = False
    
    with open(filepath, 'r') as f:
        for line in f:
            if '# Final ensemble info #' in line:
                reading_data = True
                # Skip the header lines
                next(f)  # Skip "Conformer     Energy..."
                next(f)  # Skip "              (kcal/mol)" 
                next(f)  # Skip "------..."
                continue
                
            if reading_data:
                # Stop at empty line
                if line.strip() == '':
                    break
                    
                # Parse data line
                try:
                    data = line.strip().split()
                    if len(data) >= 4:
                        ensemble_data.append({
                            'Conformer': int(data[0]),
                            'Energy_kcal_mol': float(data[1]), 
                            'Degeneracy': float(data[2]),
                            'Percent_Total': float(data[3]),
                            'Percent_Cumulative': float(data[4])
                        })
                except (ValueError, IndexError):
                    continue
                    
    return pd.DataFrame(ensemble_data)
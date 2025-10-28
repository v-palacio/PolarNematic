#!/usr/bin/env python3
"""
ORCA Input Generation

This script generates ORCA input files for DFT calculations on conformers.
"""

import sys
from pathlib import Path
import argparse

def create_orca_input(conformer_file, output_file, method="revPBE", basis="def2-SVP", 
                     nprocs=4, maxcore=3000, include_cosmo=True):
    """Write an ORCA input for a conformer XYZ."""
    
    conformer_file = Path(conformer_file)
    output_file = Path(output_file)
    
    # Create ORCA input content
    orca_input = [
        f"! {method} {basis} def2/J Hirshfeld CHELPG RIJONX D3 Energy KeepDens\n",
        f"%pal\n",
        f"nprocs {nprocs}\n",
        f"end\n",
        f"%chelpg\n",
        f"DIPOLE TRUE\n",
        f"end\n",
        f"%plots\n",
        f"Format Gaussian_Cube\n",
        f"ElDens(\"eldens.cube\");\n",
        f"end\n",
        f"%maxcore {maxcore}\n",
        f"%elprop\n",
        f"Polar 1\n",
        f"Quadrupole True\n",
        f"end\n",
        f"* xyzfile 0 1 {conformer_file.name}\n"
    ]
    
    # Add COSMO calculation if requested
    if include_cosmo:
        orca_input.extend([
            "\n",
            "# COSMO calculation for sigma profile\n",
            "$new_job\n",
            f"! {method} {basis} def2/J CPCMC Hirshfeld CHELPG RIJONX D3 Energy\n",
            f"%base \"property_cosmo\"\n",
            f"%chelpg\n",
            f"DIPOLE TRUE\n",
            f"end\n",
            f"%pal\n",
            f"nprocs {nprocs}\n",
            f"end\n",
            f"%maxcore {maxcore}\n",
            f"* xyzfile 0 1 {conformer_file.name}\n"
        ])
    
    # Write input file
    with open(output_file, 'w') as f:
        f.writelines(orca_input)
    
    print(f"Created ORCA input: {output_file}")

def create_xtb_input(conformer_file, output_file):
    """Write an xTB GOAT input for geometry optimization."""
    
    conformer_file = Path(conformer_file)
    output_file = Path(output_file)
    
    xtb_input = [
        "!GOAT xtb\n",
        "%PAL\n",
        " NPROCS 4\n",
        "END\n",
        "%GOAT\n",
        " MAXOPTITER     64\n",
        " MAXEN 4.0\n",
        " GFNUPHILL    GFNFF\n",
        "END\n",
        f"*XYZFILE 0 1 {conformer_file.name}\n"
    ]
    
    with open(output_file, 'w') as f:
        f.writelines(xtb_input)
    
    print(f"Created xTB input: {output_file}")

def create_submit_script(job_name, orca_input, output_dir, nprocs=4):
    """Write a simple SGE submit script for the ORCA job."""
    
    output_dir = Path(output_dir)
    submit_file = output_dir / f"submit_{job_name}.sh"
    
    submit_script = [
        "#!/bin/bash\n",
        f"#$ -N {job_name}\n",
        "#$ -cwd\n",
        "#$ -o run.out\n",
        "#$ -e run.err\n",
        f"#$ -pe orte {nprocs}\n",
        "#$ -q all.q\n\n",
        "module load openmpi/4.1.6\n",
        f"/share/apps/orca/orca_6_0_0/orca {orca_input.name} > {orca_input.stem}.out\n"
    ]
    
    with open(submit_file, 'w') as f:
        f.writelines(submit_script)
    
    # Make executable
    submit_file.chmod(0o755)
    
    print(f"Created submit script: {submit_file}")

def process_conformer_directory(conformer_dir, output_dir, method="revPBE", 
                              basis="def2-SVP", nprocs=4, maxcore=3000):
    """Generate ORCA/xTB inputs and submit scripts for all *_conformers.xyz in a folder."""
    
    conformer_dir = Path(conformer_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find conformer files
    conformer_files = list(conformer_dir.glob("*conformers.xyz"))
    
    if not conformer_files:
        print(f"No conformer files found in {conformer_dir}")
        return
    
    print(f"Found {len(conformer_files)} conformer files")
    
    for conformer_file in conformer_files:
        molecule_name = conformer_file.stem.replace("_conformers", "")
        
        # Create ORCA input
        orca_input_file = output_dir / f"{molecule_name}_orca.inp"
        create_orca_input(
            conformer_file, orca_input_file, method, basis, nprocs, maxcore
        )
        
        # Create submit script
        create_submit_script(
            f"ORCA-{molecule_name}", orca_input_file, output_dir, nprocs
        )
        
        # Create xTB input for comparison
        xtb_input_file = output_dir / f"{molecule_name}_xtb.inp"
        create_xtb_input(conformer_file, xtb_input_file)

del argparse  # no CLI integration

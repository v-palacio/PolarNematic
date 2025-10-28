#!/usr/bin/env python3
"""
ORCA Output Parsing

This script parses ORCA output files to extract electronic properties,
dipole moments, and other calculated quantities.
"""

import sys
import re
import pandas as pd
from pathlib import Path
import argparse

def parse_orca_output(output_file):
    """Extract key properties from an ORCA .out file."""
    
    output_file = Path(output_file)
    
    if not output_file.exists():
        raise FileNotFoundError(f"Output file not found: {output_file}")
    
    properties = {}
    
    with open(output_file, 'r') as f:
        content = f.read()
    
    # Check if calculation completed normally
    if "ORCA TERMINATED NORMALLY" not in content:
        print(f"Warning: ORCA calculation may not have completed normally in {output_file}")
        return None
    
    # Extract final energy
    energy_match = re.search(r'FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)', content)
    if energy_match:
        properties['final_energy_hartree'] = float(energy_match.group(1))
        properties['final_energy_kcal_mol'] = float(energy_match.group(1)) * 627.509
    
    # Extract dipole moment
    dipole_match = re.search(r'Magnitude \(Debye\)\s+:\s+(\d+\.\d+)', content)
    if dipole_match:
        properties['dipole_moment_debye'] = float(dipole_match.group(1))
    
    # Extract dipole components
    dipole_x_match = re.search(r'X\s+(-?\d+\.\d+)', content)
    dipole_y_match = re.search(r'Y\s+(-?\d+\.\d+)', content)
    dipole_z_match = re.search(r'Z\s+(-?\d+\.\d+)', content)
    
    if dipole_x_match and dipole_y_match and dipole_z_match:
        properties['dipole_x'] = float(dipole_x_match.group(1))
        properties['dipole_y'] = float(dipole_y_match.group(1))
        properties['dipole_z'] = float(dipole_z_match.group(1))
    
    # Extract quadrupole moment
    quadrupole_match = re.search(r'QUADRUPOLE MOMENT TENSOR', content)
    if quadrupole_match:
        # Extract quadrupole components (simplified)
        quadrupole_pattern = r'Q\((\w+),(\w+)\)\s+(-?\d+\.\d+)'
        quadrupole_matches = re.findall(quadrupole_pattern, content)
        
        if quadrupole_matches:
            properties['quadrupole_components'] = len(quadrupole_matches)
    
    # Extract polarizability
    polarizability_match = re.search(r'ISOTROPIC POLARIZABILITY\s+(\d+\.\d+)', content)
    if polarizability_match:
        properties['polarizability_au'] = float(polarizability_match.group(1))
    
    # Extract HOMO-LUMO gap
    homo_match = re.search(r'HOMO\s+(-?\d+\.\d+)', content)
    lumo_match = re.search(r'LUMO\s+(-?\d+\.\d+)', content)
    
    if homo_match and lumo_match:
        homo = float(homo_match.group(1))
        lumo = float(lumo_match.group(1))
        properties['homo_ev'] = homo
        properties['lumo_ev'] = lumo
        properties['homo_lumo_gap_ev'] = lumo - homo
    
    # Extract COSMO information if present
    if "COSMO" in content:
        properties['cosmo_calculation'] = True
        
        # Extract COSMO area
        cosmo_area_match = re.search(r'Total COSMO area\s+(\d+\.\d+)', content)
        if cosmo_area_match:
            properties['cosmo_area'] = float(cosmo_area_match.group(1))
        
        # Extract COSMO volume
        cosmo_volume_match = re.search(r'Total COSMO volume\s+(\d+\.\d+)', content)
        if cosmo_volume_match:
            properties['cosmo_volume'] = float(cosmo_volume_match.group(1))
    else:
        properties['cosmo_calculation'] = False
    
    return properties

def parse_cosmo_output(output_file):
    """Extract basic COSMO summary values from a COSMO .out file."""
    
    output_file = Path(output_file)
    
    if not output_file.exists():
        return None
    
    cosmo_properties = {}
    
    with open(output_file, 'r') as f:
        content = f.read()
    
    # Extract sigma profile statistics
    sigma_mean_match = re.search(r'Sigma mean\s+(-?\d+\.\d+)', content)
    sigma_std_match = re.search(r'Sigma std\s+(\d+\.\d+)', content)
    sigma_skew_match = re.search(r'Sigma skewness\s+(-?\d+\.\d+)', content)
    
    if sigma_mean_match:
        cosmo_properties['sigma_mean'] = float(sigma_mean_match.group(1))
    if sigma_std_match:
        cosmo_properties['sigma_std'] = float(sigma_std_match.group(1))
    if sigma_skew_match:
        cosmo_properties['sigma_skewness'] = float(sigma_skew_match.group(1))
    
    return cosmo_properties

def process_orca_directory(output_dir):
    """Parse all *.out files in a folder and return a results DataFrame."""
    
    output_dir = Path(output_dir)
    
    # Find ORCA output files
    output_files = list(output_dir.glob("*.out"))
    
    if not output_files:
        print(f"No ORCA output files found in {output_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(output_files)} ORCA output files")
    
    all_properties = []
    
    for output_file in output_files:
        molecule_name = output_file.stem.replace("_orca", "")
        
        print(f"Processing {output_file.name}")
        
        # Parse main ORCA output
        properties = parse_orca_output(output_file)
        
        if properties is None:
            print(f"Failed to parse {output_file.name}")
            continue
        
        properties['molecule_name'] = molecule_name
        properties['output_file'] = str(output_file)
        
        # Parse COSMO output if present
        cosmo_file = output_dir / f"{molecule_name}_cosmo.out"
        if cosmo_file.exists():
            cosmo_properties = parse_cosmo_output(cosmo_file)
            if cosmo_properties:
                properties.update(cosmo_properties)
        
        all_properties.append(properties)
    
    # Create DataFrame
    df = pd.DataFrame(all_properties)
    
    # Save results
    results_file = output_dir / "orca_results.csv"
    df.to_csv(results_file, index=False)
    
    print(f"Processed {len(all_properties)} molecules")
    print(f"Results saved to {results_file}")
    
    return df

del argparse  # no CLI integration

#!/usr/bin/env python3
"""
Complete DFT Pipeline

This script runs the complete RDKit → ORCA pipeline:
1. Generate conformers using RDKit
2. Create ORCA input files
3. Submit ORCA calculations
4. Parse ORCA outputs
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import subprocess
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dft.generate_conformers import process_molecule_batch
from dft.create_orca_inputs import process_conformer_directory
from dft.parse_orca_output import process_orca_directory

def run_dft_pipeline(mol_dir, output_dir, n_conformers=15, method="revPBE", 
                    basis="def2-SVP", nprocs=4, maxcore=3000, submit_jobs=False):
    """End-to-end: conformers → ORCA inputs → (optional submit) → parse → join."""
    
    mol_dir = Path(mol_dir)
    output_dir = Path(output_dir)
    
    print("=" * 60)
    print("DFT PIPELINE STARTING")
    print("=" * 60)
    
    # Step 1: Generate conformers
    print("\nStep 1: Generating conformers...")
    conformer_dir = output_dir / "conformers"
    conformer_summary = process_molecule_batch(
        mol_dir, conformer_dir, n_conformers
    )
    
    if conformer_summary.empty:
        print("No conformers generated. Exiting.")
        return pd.DataFrame()
    
    # Step 2: Create ORCA input files
    print("\nStep 2: Creating ORCA input files...")
    orca_input_dir = output_dir / "orca_inputs"
    process_conformer_directory(
        conformer_dir, orca_input_dir, method, basis, nprocs, maxcore
    )
    
    # Step 3: Submit ORCA jobs (if requested)
    if submit_jobs:
        print("\nStep 3: Submitting ORCA jobs...")
        submit_jobs_to_cluster(orca_input_dir)
        
        # Wait for jobs to complete
        print("Waiting for ORCA calculations to complete...")
        wait_for_jobs_completion(orca_input_dir)
    
    # Step 4: Parse ORCA outputs
    print("\nStep 4: Parsing ORCA outputs...")
    orca_output_dir = output_dir / "orca_outputs"
    results_df = process_orca_directory(orca_output_dir)
    
    # Step 5: Combine with conformer information
    if not results_df.empty and not conformer_summary.empty:
        print("\nStep 5: Combining results...")
        final_df = combine_results(conformer_summary, results_df)
        
        # Save final results
        final_results_file = output_dir / "dft_pipeline_results.csv"
        final_df.to_csv(final_results_file, index=False)
        
        print(f"Final results saved to {final_results_file}")
        return final_df
    
    return results_df

def submit_jobs_to_cluster(orca_input_dir):
    """Submit all submit_*.sh scripts via qsub."""
    
    submit_scripts = list(orca_input_dir.glob("submit_*.sh"))
    
    print(f"Found {len(submit_scripts)} submit scripts")
    
    for script in submit_scripts:
        try:
            # Submit job
            result = subprocess.run(
                ["qsub", str(script)], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                job_id = result.stdout.strip()
                print(f"Submitted {script.name}: {job_id}")
            else:
                print(f"Failed to submit {script.name}: {result.stderr}")
                
        except Exception as e:
            print(f"Error submitting {script.name}: {str(e)}")

def wait_for_jobs_completion(orca_input_dir, check_interval=300):
    """Poll for *.out files corresponding to *_orca.inp until all exist."""
    
    orca_inputs = list(orca_input_dir.glob("*_orca.inp"))
    expected_outputs = [inp.with_suffix('.out') for inp in orca_inputs]
    
    print(f"Waiting for {len(expected_outputs)} ORCA calculations to complete...")
    
    while True:
        completed = sum(1 for out in expected_outputs if out.exists())
        
        if completed == len(expected_outputs):
            print("All ORCA calculations completed!")
            break
        
        print(f"Completed: {completed}/{len(expected_outputs)}")
        time.sleep(check_interval)

def combine_results(conformer_summary, orca_results):
    """Merge conformer summary with parsed ORCA outputs by molecule name."""
    
    # Merge on molecule name
    combined = conformer_summary.merge(
        orca_results, 
        left_on='molecule_name', 
        right_on='molecule_name', 
        how='inner'
    )
    
    # Calculate Boltzmann-weighted properties
    if 'boltzmann_weights' in combined.columns and 'dipole_moment_debye' in combined.columns:
        # This would need to be expanded for multiple conformers
        # For now, just use the properties as-is
        pass
    
    return combined

del argparse  # no CLI integration

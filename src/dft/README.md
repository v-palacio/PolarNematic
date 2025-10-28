# DFT Pipeline Documentation

This directory contains scripts for running DFT calculations using ORCA.

## Scripts Overview

### 1. `generate_conformers.py`
Generates conformer ensembles using RDKit and prepares them for DFT calculations.

**Usage:**
```bash
python generate_conformers.py mol_dir output_dir --n_conformers 15
```

**Features:**
- Generates multiple conformers per molecule
- Optimizes conformers using UFF force field
- Calculates Boltzmann weights
- Saves conformers in XYZ format

### 2. `create_orca_inputs.py`
Creates ORCA input files for DFT calculations on conformers.

**Usage:**
```bash
python create_orca_inputs.py conformer_dir output_dir --method revPBE --basis def2-SVP
```

**Features:**
- Creates ORCA input files with specified method/basis
- Includes COSMO calculations for sigma profiles
- Generates cluster submission scripts
- Creates xTB input files for comparison

### 3. `parse_orca_output.py`
Parses ORCA output files to extract electronic properties.

**Usage:**
```bash
python parse_orca_output.py output_dir --output results.csv
```

**Features:**
- Extracts energies, dipole moments, polarizabilities
- Parses COSMO information
- Handles multiple conformers
- Saves results to CSV

### 4. `run_dft_pipeline.py`
Runs the complete DFT pipeline from conformer generation to result parsing.

**Usage:**
```bash
python run_dft_pipeline.py mol_dir output_dir --submit
```

**Features:**
- Complete automated pipeline
- Optional cluster job submission
- Combines all results
- Progress monitoring

## Example Files

- `example_orca_input.inp`: Example ORCA input file
- `example_output.out`: Example ORCA output file

## Dependencies

- RDKit (conformer generation)
- ORCA (DFT calculations)
- xTB (optional, for geometry optimization)
- pandas (data processing)

## Workflow

1. **Conformer Generation**: Generate multiple conformers for each molecule
2. **Input Creation**: Create ORCA input files for each conformer
3. **Job Submission**: Submit ORCA calculations to cluster (optional)
4. **Output Parsing**: Parse ORCA outputs to extract properties
5. **Result Combination**: Combine conformer and DFT results

## Notes

- The pipeline is designed for cluster computing environments
- Modify submission scripts for your specific cluster setup
- Adjust ORCA input parameters as needed for your calculations
- The pipeline handles both single-point and COSMO calculations

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem

from molecular_descriptors import calculate_similarities
from visualization_utils import (
    plot_feature_importance, plot_model_predictions, 
    plot_correlation_heatmap, plot_similarity_analysis,
    plot_top_features_grid, plot_feature_distributions,
    plot_clustering_results, plot_mds_projection, plot_dipole_analysis
)

def create_mol_dict(mol_dir):
    """Create a dictionary of molecule names to RDKit molecules."""
    mol_dict = {}
    for filename in os.listdir(mol_dir):
        if filename.endswith('.mol'):
            name = filename.replace('.mol', '')
            mol = Chem.MolFromMolFile(os.path.join(mol_dir, filename))
            if mol is not None:
                Chem.SanitizeMol(mol)
                mol_dict[name] = mol
    return mol_dict

def calculate_similarities(molecules):
    """
    Calculate different types of molecular similarities.
    
    Parameters:
    -----------
    molecules : list
        List of RDKit molecule objects
    
    Returns:
    --------
    tuple
        (count_similarities, bit_vector_similarities, simulated_count_similarities)
    """
    # Define fingerprint generators
    fpgen = GetMorganGenerator(radius=2, fpSize=1024)
    simfpgen = GetMorganGenerator(radius=2, fpSize=4096, countSimulation=True)
    
    # Generate fingerprints
    fps = [fpgen.GetFingerprint(m) for m in molecules]
    countfps = [fpgen.GetCountFingerprint(m) for m in molecules]
    simfps = [simfpgen.GetFingerprint(m) for m in molecules]
    
    # Calculate similarities
    countsims = []
    sims = []
    simsims = []
    
    for i in range(len(molecules)):
        for j in range(i+1, len(molecules)):
            countsims.extend(DataStructs.BulkTanimotoSimilarity(countfps[i], countfps[j:]))
            sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[j:]))
            simsims.extend(DataStructs.BulkTanimotoSimilarity(simfps[i], simfps[j:]))
    
    return countsims, sims, simsims

def analyze_transitions(final_df, mol_dict, output_dir):
    """
    Analyze transitions and generate visualizations.
    
    Parameters:
    -----------
    final_df : pd.DataFrame
        Processed dataframe from load_and_process_data
    mol_dict : dict
        Dictionary of molecule names to RDKit molecules
    output_dir : Path
        Path to output directory
    
    Returns:
    --------
    dict
        Results dictionary containing analysis results for each transition type
    """
    
    # Define transition types
    transition_types = {
        'I-N': 2,       
        'All': None
    }
    
    
    # Analysis by transition type
    results_by_type = {}
    
    for transition_name, transition_type in transition_types.items():
        print(f"\nAnalyzing {transition_name} transitions")
        
        # Filter data for specific transition type
        if transition_type is not None:
            current_df = final_df[final_df['Transition type'] == transition_type].copy()
        else:
            current_df = final_df.copy()

        # Get molecules for this transition type
        transition_mols = [mol_dict[name] for name in current_df['Name_x']]
         
        # Prepare features and target
        y = current_df['transition_temp']
        X = current_df.select_dtypes(include=[np.float64, np.int64])
        X = X.drop(['transition_temp', 'Transition type'], axis=1)
        
        # Feature selection and modeling
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Generate visualizations
        plot_feature_importance(feature_importance, transition_name, output_dir)
        
        # Select top features and train models
        top_features = feature_importance['feature'].head(10).tolist()
        X = X[top_features]
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models and get predictions
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'SVR': SVR(kernel='rbf'),
            'Random Forest': RandomForestRegressor(n_estimators=100)
        }
        
        results = {}
        predictions = {}
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            results[name] = {
                'R2': r2_score(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
            }
            predictions[name] = y_pred
        
        # Generate plots
        plot_model_predictions(y_test, predictions, results, transition_name, output_dir)
        plot_correlation_heatmap(X.join(pd.Series(y, name=f'{transition_name}_Temp')), 
                               transition_name, output_dir)
        
        # Calculate and plot similarities
        countsims, sims, simsims = calculate_similarities(transition_mols)
        plot_similarity_analysis(countsims, sims, simsims, transition_name, output_dir)
        
        # Store results
        results_by_type[transition_name] = {
            'X': X,
            'y': y,
            'feature_importance': feature_importance,
            'model_results': results
        }

    # Generate summary visualizations
    plot_top_features_grid(results_by_type, output_dir)
    plot_feature_distributions(results_by_type, output_dir)
    
    return results_by_type 

def perform_clustering_analysis(mol_dict, final_df, output_dir, n_clusters=4):
    """
    Perform clustering analysis on molecules using fingerprints.
    
    Parameters:
    -----------
    mol_dict : dict
        Dictionary of molecule names to RDKit molecules
    final_df : pd.DataFrame
        Processed dataframe with molecular data
    output_dir : Path
        Path to output directory
    n_clusters : int
        Number of clusters for K-means
        
    Returns:
    --------
    dict
        Dictionary containing clustering results
    """
    # Generate fingerprints
    fpgen = GetMorganGenerator(radius=2, fpSize=1024)
    mols = [mol_dict[name] for name in final_df['Name_x']]
    fps = [fpgen.GetFingerprint(m) for m in mols]
    
    # Convert fingerprints to numpy array
    fp_array = []
    for fp in fps:
        arr = np.zeros((0,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_array.append(arr)
    fp_array = np.array(fp_array)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    fp_pca = pca.fit_transform(fp_array)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(fp_array)
    
    # MDS projection
    mds = MDS(n_components=2, random_state=42)
    mds_embedding = mds.fit_transform(fp_array)
    
    # Hierarchical clustering
    dist_matrix = pdist(fp_array, metric='jaccard')
    linkage_matrix = linkage(dist_matrix, method='ward')
    
    # Scaffold analysis
    scaffolds = [MurckoScaffold.GetScaffoldForMol(mol) for mol in mols]
    scaffold_smiles = [Chem.MolToSmiles(scaffold) for scaffold in scaffolds]
    scaffold_counts = pd.Series(scaffold_smiles).value_counts()
    
    clustering_results = {
        'pca': fp_pca,
        'clusters': clusters,
        'mds_embedding': mds_embedding,
        'linkage_matrix': linkage_matrix,
        'mol_names': final_df['Name_x'].values,
        'transition_temps': final_df['transition_temp'].values,
        'transition_types': final_df['Transition type'].values,
        'scaffold_counts': scaffold_counts,
        'scaffolds': scaffolds
    }
    
    # Plot results
    plot_clustering_results(clustering_results, output_dir)
    
    return clustering_results

def calculate_mds_projection(mol_dict, final_df, output_dir):
    """
    Calculate MDS projection of molecular fingerprints.
    
    Parameters:
    -----------
    mol_dict : dict
        Dictionary of molecule names to RDKit molecules
    final_df : pd.DataFrame
        Processed dataframe with molecular data
    output_dir : Path
        Path to output directory
        
    Returns:
    --------
    dict
        Dictionary containing MDS results and associated data
    """
    # Generate fingerprints
    fpgen = GetMorganGenerator(radius=2, fpSize=1024)
    mols = [mol_dict[name] for name in final_df['Name_x']]
    fps = [fpgen.GetFingerprint(m) for m in mols]
    
    # Convert fingerprints to numpy array
    fp_array = []
    for fp in fps:
        arr = np.zeros((0,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_array.append(arr)
    fp_array = np.array(fp_array)
    
    # MDS projection
    mds = MDS(n_components=2, random_state=42)
    embedding = mds.fit_transform(fp_array)
    
    mds_results = {
        'embedding': embedding,
        'temperatures': final_df['transition_temp'].values,
        'transition_types': final_df['Transition type'].values
    }
    
    # Plot results
    plot_mds_projection(mds_results, output_dir)
    
    return mds_results

def calculate_dipole_moments(mol_dict, final_df, output_dir):
    """
    Calculate dipole moments for molecules.
    
    Parameters:
    -----------
    mol_dict : dict
        Dictionary of molecule names to RDKit molecules
    final_df : pd.DataFrame
        Processed dataframe with molecular data
    output_dir : Path
        Path to output directory
        
    Returns:
    --------
    dict
        Dictionary containing dipole moments and associated data
    """
    # Calculate dipoles for molecules
    dipoles = []
    filtered_mols = []
    filtered_names = []
    
    for name in final_df['Name_x']:
        mol = mol_dict[name]
        if mol is not None:
            # Generate 3D conformation if not present
            if mol.GetNumConformers() == 0:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
            
            # Calculate dipole
            dipole = calculate_dipole(mol)
            if dipole is not None:
                dipoles.append(dipole)
                filtered_mols.append(mol)
                filtered_names.append(name)
    
    dipole_results = {
        'dipoles': np.array(dipoles),
        'temperatures': final_df.loc[final_df['Name_x'].isin(filtered_names), 'transition_temp'].values,
        'transition_types': final_df.loc[final_df['Name_x'].isin(filtered_names), 'Transition type'].values,
        'filtered_mols': filtered_mols,
        'filtered_names': filtered_names
    }
    
    # Plot results
    plot_dipole_analysis(dipole_results, output_dir)
    
    return dipole_results

def calculate_dipole(mol):
    """
    Calculate dipole moment for a molecule.
    
    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object with 3D coordinates
        
    Returns:
    --------
    float or None
        Dipole moment in Debye, or None if calculation fails
    """
    try:
        AllChem.ComputeGasteigerCharges(mol)
        pos = mol.GetConformer().GetPositions()
        charges = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') 
                  for i in range(mol.GetNumAtoms())]
        
        dipole = np.zeros(3)
        for i in range(mol.GetNumAtoms()):
            dipole += charges[i] * pos[i]
        
        # Convert to Debye (1 e*Å ≈ 4.803 Debye)
        return np.linalg.norm(dipole) * 4.803
    except:
        return None 
    

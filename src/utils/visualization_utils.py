import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import dendrogram
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
import py3Dmol


def draw_with_spheres(mol):
    v = py3Dmol.view(width=300,height=300)
    IPythonConsole.addMolToView(mol,v)
    v.zoomTo()
    v.setStyle({'sphere':{'radius':0.3},'stick':{'radius':0.2}});
    v.show()

def plot_feature_importance(feature_importance_df, transition_name, output_dir, n_features=15):
    """Plot top N most important features for a given transition type."""
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=feature_importance_df.head(n_features), 
        x='importance', 
        y='feature'
    )
    plt.title(f'Top {n_features} Most Important Molecular Descriptors for {transition_name} Transition\n(n={len(feature_importance_df)})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Use output_dir for saving
    feature_dir = output_dir / 'figures' / 'feature'
    feature_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(feature_dir / f'feature_importance_{transition_name}.png')
    plt.close()

def plot_model_predictions(y_test, predictions, results, transition_name, output_dir):
    """Plot actual vs predicted values for all models."""
    plt.figure(figsize=(15, 10))
    for i, (name, y_pred) in enumerate(predictions.items(), 1):
        plt.subplot(2, 3, i)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], 
                 [y_test.min(), y_test.max()], 
                 'r--', lw=2)
        plt.xlabel(f'Actual {transition_name} Temperature')
        plt.ylabel(f'Predicted {transition_name} Temperature')
        plt.title(f'{name}\nR² = {results[name]["R2"]:.3f}\n(n={len(y_test)})')
    plt.tight_layout()
    
    feature_dir = output_dir / 'figures' / 'feature'
    feature_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(feature_dir / f'model_predictions_{transition_name}.png')
    plt.close()

def plot_correlation_heatmap(data, transition_name, output_dir):
    """Plot correlation heatmap for selected features."""
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0, 
                fmt='.2f')
    plt.title(f'Correlation Heatmap of Selected Features ({transition_name})\n(n={len(data)})')
    plt.tight_layout()
    
    feature_dir = output_dir / 'figures' / 'feature'
    feature_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(feature_dir / f'correlation_heatmap_{transition_name}.png')
    plt.close()

def plot_top_features_grid(results_by_type, output_dir, n_features=5):
    """Create a grid showing top N features for each transition type."""
    plt.figure(figsize=(15, 12))
    for i, (transition_name, data) in enumerate(results_by_type.items()):
        top_features = data['feature_importance'].head(n_features)
        plt.subplot(4, 1, i+1)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {n_features} Features for {transition_name} Transition')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
    plt.tight_layout()
    
    feature_dir = output_dir / 'figures' / 'feature'
    feature_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(feature_dir / f'top_features_grid.png')
    plt.close()

def plot_feature_distributions(results_by_type, output_dir, n_features=5):
    """Create violin plots for first N columns of each transition type."""
    plt.figure(figsize=(15, 12))
    for i, (transition_name, data) in enumerate(results_by_type.items(), 1):
        first_n_cols = data['X'].columns[:n_features].tolist()
        top_n = data['X'][first_n_cols]
        
        plt.subplot(2, 2, i)
        sns.violinplot(data=top_n)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Distribution of First {n_features} Features - {transition_name} Transition')
        plt.xlabel('Feature')
        plt.ylabel('Value')
    plt.tight_layout()
    
    feature_dir = output_dir / 'figures' / 'feature'
    feature_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(feature_dir / f'feature_distributions.png')
    plt.close()

def plot_similarity_analysis(countsims, sims, simsims, transition_name, output_dir):
    """Plot similarity analysis using hexbin plots."""
    bvr, _ = stats.spearmanr(countsims, sims)
    simr, _ = stats.spearmanr(countsims, simsims)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hexbin(countsims, sims, bins='log')
    plt.plot((0, 1), (0, 1), 'k-')
    plt.title(f'{transition_name}: bit vector similarity, r={bvr:.2f}')
    plt.xlabel('count similarity')
    plt.ylabel('bit vector similarity')
    
    plt.subplot(1, 2, 2)
    plt.hexbin(countsims, simsims, bins='log')
    plt.plot((0, 1), (0, 1), 'k-')
    plt.title(f'{transition_name}: simulated counts similarity, r={simr:.2f}')
    plt.xlabel('count similarity')
    plt.ylabel('simulated counts similarity')
    
    plt.tight_layout()
    
    feature_dir = output_dir / 'figures' / 'feature'
    feature_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(feature_dir / f'similarity_analysis_{transition_name}.png')
    plt.close()

def plot_clustering_results(clustering_results, output_dir):
    """Plot clustering analysis results."""
    # Unpack results
    fp_pca = clustering_results['pca']
    clusters = clustering_results['clusters']
    ttemps = clustering_results['transition_temps']
    transition_types = clustering_results['transition_types']
    n_clusters = len(np.unique(clusters))
    
    # Create subplots for PCA visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left subplot - colored by transition temperature
    scatter1 = ax1.scatter(fp_pca[:, 0], fp_pca[:, 1], c=ttemps, cmap='viridis')
    plt.colorbar(scatter1, ax=ax1, label='Transition Temperature (°C)')
    ax1.set_title('Clusters by Structure\n(colored by transition temperature)')
    ax1.set_xlabel('First Principal Component')
    ax1.set_ylabel('Second Principal Component')
    
    # Draw cluster boundaries for left subplot
    for i in range(n_clusters):
        mask = clusters == i
        cluster_points = fp_pca[mask]
        if len(cluster_points) >= 3:  # Need at least 3 points for ConvexHull
            hull = ConvexHull(cluster_points)
            for simplex in hull.simplices:
                ax1.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 
                        'k-', alpha=0.3, linewidth=2)
    
    # Right subplot - colored by transition type
    cmap = plt.cm.get_cmap('tab10', 4)
    scatter2 = ax2.scatter(fp_pca[:, 0], fp_pca[:, 1], c=transition_types, cmap=cmap)
    plt.colorbar(scatter2, ax=ax2, label='Transition Type')
    ax2.set_title('Clusters by Structure\n(colored by transition type)')
    ax2.set_xlabel('First Principal Component')
    ax2.set_ylabel('Second Principal Component')
    
    # Draw cluster boundaries and add cluster sizes
    for i in range(n_clusters):
        mask = clusters == i
        cluster_points = fp_pca[mask]
        if len(cluster_points) >= 3:
            hull = ConvexHull(cluster_points)
            for simplex in hull.simplices:
                ax2.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 
                        'k-', alpha=0.3, linewidth=2)
        
        # Calculate centroid and add cluster size text
        centroid = np.mean(cluster_points, axis=0)
        cluster_size = np.sum(mask)
        ax2.text(centroid[0], centroid[1], f'n={cluster_size}', 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    
    structure_dir = output_dir / 'figures' / 'structure'
    structure_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(structure_dir / 'structure_clusters.png')
    plt.close()
    
    # Plot dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(clustering_results['linkage_matrix'], 
              labels=clustering_results['mol_names'], 
              leaf_rotation=90)
    plt.title('Hierarchical Clustering of Molecules')
    plt.xlabel('Molecule ID')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(structure_dir / 'molecule_dendrogram.png')
    plt.close()
    
    # Plot scaffold analysis
    scaffold_counts = clustering_results['scaffold_counts']
    scaffolds = clustering_results['scaffolds']
    
    # Bar chart of scaffold frequencies
    plt.figure(figsize=(12, 6))
    scaffold_counts.head(10).plot(kind='bar')
    plt.title('Most Common Molecular Scaffolds')
    plt.xlabel('Scaffold SMILES')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(structure_dir / 'scaffold_counts.png')
    plt.close()
    
    # Molecular drawings of top scaffolds
    fig = plt.figure(figsize=(12, 8))
    top_scaffolds = scaffold_counts.head(6)
    
    for i, (smiles, count) in enumerate(top_scaffolds.items(), 1):
        mol = Chem.MolFromSmiles(smiles)
        ax = fig.add_subplot(2, 3, i)
        img = Draw.MolToImage(mol)
        ax.imshow(img)
        ax.set_title(f'Count: {count}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(structure_dir / 'scaffold_structures.png')
    plt.close()

def plot_mds_projection(mds_results, output_dir):
    """Plot MDS projection of chemical space."""
    embedding = mds_results['embedding']
    temperatures = mds_results['temperatures']
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                         c=temperatures, cmap='viridis')
    plt.colorbar(scatter, label='Transition Temperature (°C)')
    plt.title('MDS Projection of Chemical Space')
    plt.xlabel('MDS1')
    plt.ylabel('MDS2')
    
    structure_dir = output_dir / 'figures' / 'structure'
    structure_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(structure_dir / 'mds_projection.png')
    plt.close()

def plot_dipole_analysis(dipole_results, output_dir):
    """Plot dipole moment analysis."""
    dipoles = dipole_results['dipoles']
    temperatures = dipole_results['temperatures']
    transition_types = dipole_results['transition_types']
    
    # Plot dipole vs transition temperature
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.get_cmap('tab10', 4)
    scatter = plt.scatter(dipoles, temperatures, 
                         c=transition_types, cmap=cmap)
    plt.xlabel('Estimated Dipole Moment (Debye)')
    plt.ylabel('Transition Temperature (°C)')
    plt.title('Transition Temperature vs Estimated Molecular Dipole')
    plt.colorbar(scatter, label='Transition Type')
    plt.tight_layout()
    
    structure_dir = output_dir / 'figures' / 'structure'
    structure_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(structure_dir / 'dipole_vs_transition_temperature.png')
    plt.close()
    
    # Calculate and print correlation
    correlation = np.corrcoef(dipoles, temperatures)[0,1]
    print(f"Correlation between dipole and transition temperature: {correlation:.3f}") 


# Create a function to plot distributions
def plot_descriptor_distributions(df, columns, title, n_cols=3, save_name=None, figures_dir=None):
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    
    # Handle single subplot case
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        if col in df.columns:
            # Clean the data for plotting
            plot_data = df[col].dropna()
            
            # Check if we have valid data to plot
            if len(plot_data) > 0 and not plot_data.isin([np.inf, -np.inf]).all():
                # Remove infinite values and convert to numpy array
                plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(plot_data) > 0:
                    try:
                        # Convert to numpy array to avoid indexing issues
                        plot_data_np = np.array(plot_data, dtype=float)
                        
                        # Additional check for valid numeric data
                        if np.isfinite(plot_data_np).any():
                            # Use matplotlib histogram instead of seaborn to avoid indexing issues
                            axes[idx].hist(plot_data_np, bins=30, alpha=0.7, density=True, edgecolor='black')
                            axes[idx].set_title(col, fontsize=12)
                            axes[idx].set_xlabel('')
                            axes[idx].tick_params(axis='x', rotation=45)
                        else:
                            axes[idx].text(0.5, 0.5, f'No finite values for {col}', 
                                         ha='center', va='center', transform=axes[idx].transAxes)
                            axes[idx].set_title(col, fontsize=12)
                    except Exception as e:
                        print(f"Error plotting {col}: {e}")
                        print(f"Data type: {type(plot_data)}")
                        print(f"Data shape: {plot_data.shape if hasattr(plot_data, 'shape') else 'no shape'}")
                        print(f"Sample data: {plot_data.head() if hasattr(plot_data, 'head') else plot_data[:5]}")
                        axes[idx].text(0.5, 0.5, f'Error plotting {col}', 
                                     ha='center', va='center', transform=axes[idx].transAxes)
                        axes[idx].set_title(col, fontsize=12)
                else:
                    axes[idx].text(0.5, 0.5, f'No valid data for {col}', 
                                 ha='center', va='center', transform=axes[idx].transAxes)
                    axes[idx].set_title(col, fontsize=12)
            else:
                axes[idx].text(0.5, 0.5, f'No valid data for {col}', 
                             ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(col, fontsize=12)
    
    # Remove empty subplots
    for idx in range(len(columns), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout()
    
    # Save figure if save_name and figures_dir provided
    if save_name and figures_dir:
        save_path = figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    plt.show()
    return fig


def plot_combined_spectrum_analysis(spectrum_profiles_dict, spectrum_indices_dict, 
                                   title=None, save_name=None, figures_dir=None, figsize=(12, 8)):
    """
    Plot average charge spectra for each dataset with standard deviation and confidence intervals.
    
    Parameters:
    -----------
    spectrum_profiles_dict : dict
        Dictionary where keys are dataset names and values are lists of spectrum profile dataframes
        e.g., {'Ferronematic': spectrum_profiles_fn, 'Nematic': spectrum_profiles_n}
    spectrum_indices_dict : dict
        Dictionary where keys are dataset names and values are lists of molecule indices
        e.g., {'Ferronematic': spectrum_profiles_indices_fn, 'Nematic': spectrum_profiles_indices_n}
    title : str, optional
        Main title for the plot
    save_name : str, optional
        Name to save the figure (without extension)
    figures_dir : Path, optional
        Directory to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    
    # Set seaborn style
    sns.set_style("ticks")
    sns.set_context("talk")
    
    # Use seaborn color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    print("Calculating average charge spectra...")
    
    for i, (dataset_name, spectrum_profiles) in enumerate(spectrum_profiles_dict.items()):
        if not spectrum_profiles:
            print(f"No spectrum profiles found for {dataset_name}")
            continue
            
        print(f"\nProcessing {dataset_name} dataset ({len(spectrum_profiles)} profiles)")
        
        # Find the spectrum columns (eigenvalues and squared coefficients)
        eigenvalue_col = None
        coeff_col = None
        
        for col in spectrum_profiles[0].columns:
            if 'eigenvalue' in col.lower():
                eigenvalue_col = col
            elif 'squared_coeff' in col.lower():
                coeff_col = col
        
        if not eigenvalue_col or not coeff_col:
            print(f"Could not find spectrum columns in {dataset_name}")
            print(f"Available columns: {spectrum_profiles[0].columns.tolist()}")
            continue
        
        # Collect all spectra for this dataset
        all_eigenvalues = []
        all_coeffs = []
        
        for sp in spectrum_profiles:
            if eigenvalue_col in sp.columns and coeff_col in sp.columns:
                all_eigenvalues.append(sp[eigenvalue_col].values)
                all_coeffs.append(sp[coeff_col].values)
        
        if not all_eigenvalues:
            print(f"No valid spectra found for {dataset_name}")
            continue
        
        # Handle variable-length spectra by interpolating to a common grid
        n_spectra = len(all_eigenvalues)
        
        # Find the range of eigenvalues across all spectra
        all_eigenvalues_flat = np.concatenate(all_eigenvalues)
        eigenvalue_min = np.min(all_eigenvalues_flat)
        eigenvalue_max = np.max(all_eigenvalues_flat)
        
        # Calculate average number of eigenvalues across spectra
        avg_n_eigenvalues = int(np.mean([len(ev) for ev in all_eigenvalues]))
        n_grid_points = avg_n_eigenvalues
        print(f"  Using {n_grid_points} grid points based on average spectrum length")
        common_eigenvalues = np.linspace(eigenvalue_min, eigenvalue_max, n_grid_points)
        
        # Interpolate each spectrum to the common grid
        interpolated_coeffs = []
        for j in range(n_spectra):
            # Interpolate coefficients to common eigenvalue grid
            from scipy.interpolate import interp1d
            try:
                f_interp = interp1d(all_eigenvalues[j], all_coeffs[j], 
                                   kind='linear', bounds_error=False, fill_value=0)
                interpolated_coeff = f_interp(common_eigenvalues)
                interpolated_coeffs.append(interpolated_coeff)
            except Exception as e:
                print(f"  Warning: Could not interpolate spectrum {j}: {e}")
                continue
        
        if not interpolated_coeffs:
            print(f"  No valid interpolated spectra for {dataset_name}")
            continue
        
        # Convert to numpy array for statistics
        coeffs_array = np.array(interpolated_coeffs)  # Shape: (n_molecules, n_grid_points)
        
        # Calculate statistics
        mean_coeffs = np.mean(coeffs_array, axis=0)
        std_coeffs = np.std(coeffs_array, axis=0)
        n_valid_spectra = len(interpolated_coeffs)
        # Calculate 95% confidence interval for coefficients
        sem_coeffs = std_coeffs / np.sqrt(n_valid_spectra)  # Standard error of mean
        ci_95 = stats.t.ppf(0.975, n_valid_spectra - 1) * sem_coeffs  # 95% CI
        
        # Plot mean spectrum with stem plot
        color = colors[i % len(colors)]
        
        # Create stem plot of average spectrum
        markerline, stemlines, baseline = ax.stem(common_eigenvalues, mean_coeffs,
                                                 label=f'{dataset_name} (n={n_valid_spectra})')
        
        # Style the stem plot components
        plt.setp(markerline, color=color, markersize=6)
        plt.setp(stemlines, color=color, linewidth=1.5)
        plt.setp(baseline, color=color, linewidth=1)
                
        print(f"  Eigenvalue range: {eigenvalue_min:.4f} to {eigenvalue_max:.4f}")
        print(f"  Mean coefficients range: {mean_coeffs.min():.4f} to {mean_coeffs.max():.4f}")
        print(f"  Average std: {np.mean(std_coeffs):.4f}")
        print(f"  Average 95% CI width: {np.mean(ci_95) * 2:.4f}")
        print(f"  Interpolated to {n_grid_points} grid points")
    
    # Customize plot with charge spectrum styling
    ax.set_xlabel('Eigenvalue (λ)', fontsize=14)
    ax.set_ylabel('|c_k|²', fontsize=14)
    
    # Move legend inside the figure
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=16, pad=20, fontweight='bold')
    else:
        ax.set_title('Average Charge Spectra Comparison', fontsize=16, pad=20, fontweight='bold')
    
    # Add grid for better readability (similar to plot_charge_spectrum)
    ax.grid(True, alpha=0.3)
    
    # Add text annotation explaining the spectrum (similar to plot_charge_spectrum)
    textstr = '\n'.join([
        'Normalized Laplacian (λ in [0,2]):',
        '- λ ≈ 0: Constant component',
        '- λ ≤ 1: Smooth variations', 
        '- λ > 1: Localized patterns'
    ])
    
    # Position text box in upper left
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save figure if save_name and figures_dir provided
    if save_name and figures_dir:
        save_path = figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    plt.show()
    return fig

def plot_descriptor_distributions_comparison(datasets, columns, title, n_cols=3, save_name=None, figures_dir=None):
    """
    Plot descriptor distributions for multiple datasets in the same plot for comparison.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary where keys are dataset names and values are dataframes
        e.g., {'Ferronematic': df_fn, 'Nematic': df_n}
    columns : list
        List of column names to plot
    title : str
        Main title for the plot
    n_cols : int
        Number of columns in the subplot grid
    save_name : str, optional
        Name to save the figure (without extension)
    figures_dir : Path, optional
        Directory to save the figure
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    
    # Handle single subplot case
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Define colors for different datasets
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        
        # Plot each dataset
        for i, (dataset_name, df) in enumerate(datasets.items()):
            if col in df.columns:
                # Clean the data for plotting
                plot_data = df[col].dropna()
                
                # Check if we have valid data to plot
                if len(plot_data) > 0 and not plot_data.isin([np.inf, -np.inf]).all():
                    # Remove infinite values and convert to numpy array
                    plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(plot_data) > 0:
                        try:
                            # Convert to numpy array to avoid indexing issues
                            plot_data_np = np.array(plot_data, dtype=float)
                            
                            # Additional check for valid numeric data
                            if np.isfinite(plot_data_np).any():
                                # Use matplotlib histogram instead of seaborn to avoid indexing issues
                                ax.hist(plot_data_np, bins=30, alpha=0.6, density=True, 
                                       label=dataset_name, color=colors[i % len(colors)], 
                                       edgecolor='black')
                            else:
                                print(f"No finite values for {col} in {dataset_name}")
                                continue
                        except Exception as e:
                            print(f"Error plotting {col} for {dataset_name}: {e}")
                            print(f"Data type: {type(plot_data)}")
                            print(f"Sample data: {plot_data.head() if hasattr(plot_data, 'head') else plot_data[:5]}")
                            continue
        
        ax.set_title(col, fontsize=12)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='upper left')
    
    # Remove empty subplots
    for idx in range(len(columns), len(axes)):
        fig.delaxes(axes[idx])
    
    #plt.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout()
    
    # Save figure if save_name and figures_dir provided
    if save_name and figures_dir:
        save_path = figures_dir / f"{save_name}_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    plt.show()
    return fig

def plot_boxplot_comparison(datasets, x_col, y_col, title, save_name=None, figures_dir=None, figsize=(12, 6)):
    """
    Plot boxplots for multiple datasets in the same plot for comparison.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary where keys are dataset names and values are dataframes
        e.g., {'Ferronematic': df_fn, 'Nematic': df_n}
    x_col : str
        Column name for x-axis (categorical variable)
    y_col : str
        Column name for y-axis (continuous variable)
    title : str
        Title for the plot
    save_name : str, optional
        Name to save the figure (without extension)
    figures_dir : Path, optional
        Directory to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Combine all datasets with a dataset identifier
    combined_data = []
    
    for dataset_name, df in datasets.items():
        if x_col in df.columns and y_col in df.columns:
            temp_df = df[[x_col, y_col]].copy()
            temp_df['Dataset'] = dataset_name
            combined_data.append(temp_df)
    
    if not combined_data:
        print(f"No data found for columns {x_col} and {y_col}")
        return None
    
    # Concatenate all datasets
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Create boxplot with hue for dataset comparison
    sns.boxplot(data=combined_df, x=x_col, y=y_col, hue='Dataset', palette='Set2')
    
    plt.title(title, fontsize=14)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Dataset', loc='upper right')
    
    plt.tight_layout()
    
    # Save figure if save_name and figures_dir provided
    if save_name and figures_dir:
        save_path = figures_dir / f"{save_name}_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    plt.show()
    return plt.gcf()

def plot_scatter_comparison(datasets, descriptors, target_col, hue_col=None, title=None, 
                          save_name=None, figures_dir=None, figsize=(18, 12), n_cols=3):
    """
    Create scatter plots comparing relationships between descriptors and a target variable across multiple datasets.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary where keys are dataset names and values are dataframes
        e.g., {'Ferronematic': df_fn, 'Nematic': df_n}
    descriptors : list
        List of descriptor column names to plot
    target_col : str
        Column name for the target variable (y-axis)
    hue_col : str, optional
        Column name for color coding points (e.g., 'Transition type')
    title : str, optional
        Main title for the plot
    save_name : str, optional
        Name to save the figure (without extension)
    figures_dir : Path, optional
        Directory to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    n_cols : int, optional
        Number of columns in the subplot grid
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    n_rows = (len(descriptors) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single subplot case
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Create a combined dataset for consistent color mapping
    combined_data = []
    for dataset_name, df in datasets.items():
        temp_df = df.copy()
        temp_df['Dataset'] = dataset_name
        combined_data.append(temp_df)
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Get unique values for hue column if specified
    if hue_col and hue_col in combined_df.columns:
        hue_values = combined_df[hue_col].unique()
        palette = sns.color_palette("husl", n_colors=len(hue_values))
    
    print(f"Creating scatter plots for: {descriptors}")
    
    # Define colors for datasets outside the loop
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, descriptor in enumerate(descriptors):
        ax = axes[i]
        
        # Plot each dataset
        for j, (dataset_name, df) in enumerate(datasets.items()):
            if descriptor in df.columns and target_col in df.columns:
                # Create scatter plot
                if hue_col and hue_col in df.columns:
                    # Use different markers for different datasets
                    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
                    marker = markers[j % len(markers)]
                    
                    # Plot each hue category separately to control markers
                    for k, hue_val in enumerate(hue_values):
                        mask = df[hue_col] == hue_val
                        if mask.any():
                            ax.scatter(df.loc[mask, descriptor], df.loc[mask, target_col], 
                                     c=[palette[k]], marker=marker, alpha=0.7, s=60,
                                     label=f'{dataset_name} - {hue_val}' if i == 0 else "")
                else:
                    # Simple scatter without hue
                    ax.scatter(df[descriptor], df[target_col], 
                             c=colors[j % len(colors)], alpha=0.7, s=60,
                             label=dataset_name if i == 0 else "")
                
                # Calculate and display correlation
                corr = df[descriptor].corr(df[target_col])
                ax.text(0.05, 0.95 - j*0.1, f'{dataset_name}: r = {corr:.3f}', 
                       transform=ax.transAxes, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[j % len(colors)], alpha=0.3))
        
        ax.set_title(f'{descriptor}')
        ax.set_xlabel(descriptor)
        ax.set_ylabel(target_col if i >= len(descriptors) - n_cols else '')
        
        # Add legend only to the first subplot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Remove empty subplots
    for idx in range(len(descriptors), len(axes)):
        fig.delaxes(axes[idx])
    
    if title:
        plt.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save figure if save_name and figures_dir provided
    if save_name and figures_dir:
        save_path = figures_dir / f"{save_name}_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    plt.show()
    return fig

def plot_correlation_comparison(datasets, target_col, top_n=20, exclude_cols=None, 
                               title=None, save_name=None, figures_dir=None, figsize=(16, 10)):
    """
    Create correlation analysis comparison for multiple datasets showing which descriptors 
    correlate most strongly with a target variable.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary where keys are dataset names and values are dataframes
        e.g., {'Ferronematic': df_fn, 'Nematic': df_n}
    target_col : str
        Column name for the target variable (e.g., 'transition_temp')
    top_n : int, optional
        Number of top correlations to show
    exclude_cols : list, optional
        Column names to exclude from correlation analysis (e.g., ['SMILES'])
    title : str, optional
        Main title for the plot
    save_name : str, optional
        Name to save the figure (without extension)
    figures_dir : Path, optional
        Directory to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    correlations_dict : dict
        Dictionary containing correlation results for each dataset
    """
    if exclude_cols is None:
        exclude_cols = ['SMILES']
    
    correlations_dict = {}
    n_datasets = len(datasets)
    
    # Create subplots - one for each dataset
    fig, axes = plt.subplots(1, n_datasets, figsize=figsize)
    
    # Handle single dataset case
    if n_datasets == 1:
        axes = [axes]
    
    colors = ['red', 'orange', 'lightblue']
    
    for i, (dataset_name, df) in enumerate(datasets.items()):
        ax = axes[i]
        
        # Drop non-numeric columns
        numeric_df = df.copy()
        for col in exclude_cols:
            if col in numeric_df.columns:
                numeric_df = numeric_df.drop(col, axis=1)
        
        # Calculate correlations with target
        if target_col in numeric_df.columns:
            correlations_with_target = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
            
            # Remove target itself and get top correlations
            top_correlations = correlations_with_target.drop(target_col).head(top_n)
            correlations_dict[dataset_name] = top_correlations
            
            # Create color scheme based on correlation strength
            bar_colors = [colors[0] if x > 0.3 else colors[1] if x > 0.2 else colors[2] 
                         for x in top_correlations.values]
            
            # Create horizontal bar plot
            bars = ax.barh(range(len(top_correlations)), top_correlations.values, color=bar_colors)
            ax.set_yticks(range(len(top_correlations)))
            ax.set_yticklabels(top_correlations.index, fontsize=8)
            ax.set_xlabel('Absolute Correlation with ' + target_col)
            ax.set_title(f'{dataset_name}\nTop {len(top_correlations)} Correlations')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for j, (bar, value) in enumerate(zip(bars, top_correlations.values)):
                ax.text(value + 0.01, j, f'{value:.3f}', va='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, f'{target_col} not found in {dataset_name}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    if title:
        plt.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save figure if save_name and figures_dir provided
    if save_name and figures_dir:
        save_path = figures_dir / f"{save_name}_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    plt.show()
    
    # Print comparison summary
    print("\n" + "="*60)
    print("CORRELATION COMPARISON SUMMARY")
    print("="*60)
    
    for dataset_name, correlations in correlations_dict.items():
        print(f"\n{dataset_name.upper()} - Top {min(10, len(correlations))} correlations:")
        for desc, corr in correlations.head(10).items():
            print(f"  {desc}: {corr:.3f}")
    
    # Find common high correlations
    if len(correlations_dict) > 1:
        print(f"\n{'='*60}")
        print("COMMON HIGH CORRELATIONS (>0.2 in all datasets):")
        print("="*60)
        
        # Get descriptors with high correlation in all datasets
        all_descriptors = set()
        for correlations in correlations_dict.values():
            all_descriptors.update(correlations.index)
        
        common_high_corr = []
        for desc in all_descriptors:
            corr_values = []
            for dataset_name, correlations in correlations_dict.items():
                if desc in correlations.index:
                    corr_values.append(correlations[desc])
                else:
                    corr_values.append(0.0)  # If descriptor not in top correlations
            
            if all(corr > 0.2 for corr in corr_values):
                common_high_corr.append((desc, corr_values))
        
        if common_high_corr:
            common_high_corr.sort(key=lambda x: min(x[1]), reverse=True)
            for desc, corr_values in common_high_corr:
                corr_str = " | ".join([f"{list(correlations_dict.keys())[i]}: {corr:.3f}" 
                                     for i, corr in enumerate(corr_values)])
                print(f"  {desc}: {corr_str}")
        else:
            print("  No descriptors with >0.2 correlation in all datasets")
    
    return fig, correlations_dict

def plot_sigma_profile_comparison(sigma_profiles_dict, sigma_indices_dict, 
                                 title=None, save_name=None, figures_dir=None, 
                                 figsize=(15, 10), n_samples=5):
    """
    Plot sigma profile comparisons between datasets.
    
    Parameters:
    -----------
    sigma_profiles_dict : dict
        Dictionary where keys are dataset names and values are lists of sigma profile dataframes
        e.g., {'Ferronematic': sigma_profiles_fn, 'Nematic': sigma_profiles_n}
    sigma_indices_dict : dict
        Dictionary where keys are dataset names and values are lists of molecule indices
        e.g., {'Ferronematic': sigma_profiles_indices_fn, 'Nematic': sigma_profiles_indices_n}
    title : str, optional
        Main title for the plot
    save_name : str, optional
        Name to save the figure (without extension)
    figures_dir : Path, optional
        Directory to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    n_samples : int, optional
        Number of sample molecules to show individual profiles for
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    n_datasets = len(sigma_profiles_dict)
    
    # Create subplots: 2 rows - top for averages, bottom for individual samples
    fig, axes = plt.subplots(2, n_datasets, figsize=figsize)
    
    # Handle single dataset case
    if n_datasets == 1:
        axes = axes.reshape(2, 1)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (dataset_name, sigma_profiles) in enumerate(sigma_profiles_dict.items()):
        if not sigma_profiles:
            continue
            
        # Get the sigma column name (assuming it's consistent)
        sigma_col = None
        profile_cols = []
        
        for col in sigma_profiles[0].columns:
            if 'sigma' in col.lower() and 'profile' in col.lower():
                profile_cols.append(col)
            elif 'sigma' in col.lower():
                sigma_col = col
        
        if not profile_cols and sigma_col:
            profile_cols = [sigma_col]
        
        if not profile_cols:
            print(f"No sigma profile columns found in {dataset_name}")
            continue
        
        # Calculate average sigma profile
        all_profiles = []
        for sp in sigma_profiles:
            if profile_cols[0] in sp.columns:
                all_profiles.append(sp[profile_cols[0]].values)
        
        if all_profiles:
            avg_profile = np.mean(all_profiles, axis=0)
            std_profile = np.std(all_profiles, axis=0)
            
            # Plot average profile (top row)
            x_values = np.arange(len(avg_profile))
            axes[0, i].plot(x_values, avg_profile, color=colors[i], linewidth=2, 
                           label=f'{dataset_name} Average')
            axes[0, i].fill_between(x_values, avg_profile - std_profile, 
                                   avg_profile + std_profile, alpha=0.3, color=colors[i])
            axes[0, i].set_title(f'{dataset_name}\nAverage Sigma Profile (n={len(all_profiles)})')
            axes[0, i].set_xlabel('Sigma Profile Index')
            axes[0, i].set_ylabel('Sigma Value')
            axes[0, i].grid(True, alpha=0.3)
            
            # Plot individual samples (bottom row)
            sample_indices = np.linspace(0, len(sigma_profiles)-1, 
                                       min(n_samples, len(sigma_profiles)), dtype=int)
            
            for j, idx in enumerate(sample_indices):
                if profile_cols[0] in sigma_profiles[idx].columns:
                    sample_profile = sigma_profiles[idx][profile_cols[0]].values
                    axes[1, i].plot(x_values, sample_profile, alpha=0.7, 
                                   label=f'mol{sigma_indices_dict[dataset_name][idx]}')
            
            axes[1, i].set_title(f'{dataset_name}\nSample Individual Profiles')
            axes[1, i].set_xlabel('Sigma Profile Index')
            axes[1, i].set_ylabel('Sigma Value')
            axes[1, i].legend(fontsize=8)
            axes[1, i].grid(True, alpha=0.3)
    
    if title:
        plt.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save figure if save_name and figures_dir provided
    if save_name and figures_dir:
        save_path = figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    plt.show()
    return fig

def analyze_sigma_profile_statistics(sigma_profiles_dict, sigma_indices_dict):
    """
    Analyze and compare sigma profile statistics between datasets.
    
    Parameters:
    -----------
    sigma_profiles_dict : dict
        Dictionary where keys are dataset names and values are lists of sigma profile dataframes
    sigma_indices_dict : dict
        Dictionary where keys are dataset names and values are lists of molecule indices
    
    Returns:
    --------
    stats_dict : dict
        Dictionary containing statistical analysis results
    """
    stats_dict = {}
    
    print("SIGMA PROFILE STATISTICAL ANALYSIS")
    print("="*50)
    
    for dataset_name, sigma_profiles in sigma_profiles_dict.items():
        if not sigma_profiles:
            continue
            
        print(f"\n{dataset_name.upper()} DATASET:")
        print("-" * 30)
        
        # Find sigma profile columns
        profile_cols = []
        for col in sigma_profiles[0].columns:
            if 'sigma' in col.lower():
                profile_cols.append(col)
        
        dataset_stats = {}
        
        for col in profile_cols:
            all_values = []
            for sp in sigma_profiles:
                if col in sp.columns:
                    all_values.extend(sp[col].values)
            
            if all_values:
                stats = {
                    'mean': np.mean(all_values),
                    'std': np.std(all_values),
                    'min': np.min(all_values),
                    'max': np.max(all_values),
                    'median': np.median(all_values),
                    'q25': np.percentile(all_values, 25),
                    'q75': np.percentile(all_values, 75)
                }
                dataset_stats[col] = stats
                
                print(f"\n{col}:")
                print(f"  Mean: {stats['mean']:.4f}")
                print(f"  Std:  {stats['std']:.4f}")
                print(f"  Min:  {stats['min']:.4f}")
                print(f"  Max:  {stats['max']:.4f}")
                print(f"  Median: {stats['median']:.4f}")
                print(f"  Q25-Q75: {stats['q25']:.4f} - {stats['q75']:.4f}")
        
        stats_dict[dataset_name] = dataset_stats
        print(f"\nTotal profiles: {len(sigma_profiles)}")
    
    return stats_dict

def create_combined_dataset_with_sigma_profiles(datasets_dict, sigma_profiles_dict, sigma_indices_dict):
    """
    Create a combined dataframe with dataset identifiers and aggregated sigma profile data.
    
    Parameters:
    -----------
    datasets_dict : dict
        Dictionary where keys are dataset names and values are dataframes
        e.g., {'Ferronematic': df_fn, 'Nematic': df_n}
    sigma_profiles_dict : dict
        Dictionary where keys are dataset names and values are lists of sigma profile dataframes
        e.g., {'Ferronematic': sigma_profiles_fn, 'Nematic': sigma_profiles_n}
    sigma_indices_dict : dict
        Dictionary where keys are dataset names and values are lists of molecule indices
        e.g., {'Ferronematic': sigma_profiles_indices_fn, 'Nematic': sigma_profiles_indices_n}
    
    Returns:
    --------
    combined_df : pd.DataFrame
        Combined dataframe with dataset identifiers and sigma profile aggregations
    sigma_profile_summary : dict
        Summary statistics about the sigma profile aggregation
    """
    combined_data = []
    sigma_summary = {}
    
    print("Creating combined dataset with sigma profiles...")
    print("="*60)
    
    for dataset_name, df in datasets_dict.items():
        print(f"\nProcessing {dataset_name} dataset...")
        
        # Add dataset identifier
        temp_df = df.copy()
        temp_df['Dataset'] = dataset_name
        temp_df['Original_Index'] = temp_df.index
        
        # Get sigma profiles for this dataset
        sigma_profiles = sigma_profiles_dict.get(dataset_name, [])
        sigma_indices = sigma_indices_dict.get(dataset_name, [])
        
        # Initialize sigma profile columns
        sigma_cols_added = []
        
        if sigma_profiles and sigma_indices:
            print(f"  Found {len(sigma_profiles)} sigma profiles")
            
            # Get column names from first sigma profile
            first_profile = sigma_profiles[0]
            sigma_columns = [col for col in first_profile.columns if 'sigma' in col.lower()]
            
            print(f"  Sigma columns found: {sigma_columns}")
            
            # Initialize sigma profile columns in dataframe
            for col in sigma_columns:
                temp_df[f'sigma_{col}_mean'] = np.nan
                temp_df[f'sigma_{col}_std'] = np.nan
                temp_df[f'sigma_{col}_min'] = np.nan
                temp_df[f'sigma_{col}_max'] = np.nan
                temp_df[f'sigma_{col}_median'] = np.nan
                sigma_cols_added.extend([f'sigma_{col}_mean', f'sigma_{col}_std', 
                                       f'sigma_{col}_min', f'sigma_{col}_max', f'sigma_{col}_median'])
            
            # Map sigma profiles to dataframe rows
            profiles_mapped = 0
            for i, mol_idx in enumerate(sigma_indices):
                if mol_idx < len(temp_df):
                    profile = sigma_profiles[i]
                    
                    for col in sigma_columns:
                        if col in profile.columns:
                            values = profile[col].values
                            temp_df.loc[mol_idx, f'sigma_{col}_mean'] = np.mean(values)
                            temp_df.loc[mol_idx, f'sigma_{col}_std'] = np.std(values)
                            temp_df.loc[mol_idx, f'sigma_{col}_min'] = np.min(values)
                            temp_df.loc[mol_idx, f'sigma_{col}_max'] = np.max(values)
                            temp_df.loc[mol_idx, f'sigma_{col}_median'] = np.median(values)
                    
                    profiles_mapped += 1
            
            print(f"  Mapped {profiles_mapped} sigma profiles to dataframe rows")
            sigma_summary[dataset_name] = {
                'total_profiles': len(sigma_profiles),
                'mapped_profiles': profiles_mapped,
                'sigma_columns': sigma_columns,
                'aggregated_columns': sigma_cols_added
            }
        else:
            print(f"  No sigma profiles found for {dataset_name}")
            sigma_summary[dataset_name] = {
                'total_profiles': 0,
                'mapped_profiles': 0,
                'sigma_columns': [],
                'aggregated_columns': []
            }
        
        combined_data.append(temp_df)
    
    # Combine all datasets
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    print(f"\n{'='*60}")
    print("COMBINED DATASET SUMMARY")
    print("="*60)
    print(f"Total rows: {len(combined_df)}")
    print(f"Datasets: {combined_df['Dataset'].value_counts().to_dict()}")
    
    # Count sigma profile coverage
    sigma_cols = [col for col in combined_df.columns if col.startswith('sigma_')]
    if sigma_cols:
        print(f"Sigma profile columns added: {len(sigma_cols)}")
        print(f"Rows with sigma data: {combined_df[sigma_cols[0]].notna().sum()}")
    
    return combined_df, sigma_summary

def plot_combined_sigma_analysis(sigma_profiles_dict, sigma_indices_dict, 
                                title=None, save_name=None, figures_dir=None, figsize=(12, 8)):
    """
    Plot average sigma profiles for each dataset with standard deviation and confidence intervals.
    
    Parameters:
    -----------
    sigma_profiles_dict : dict
        Dictionary where keys are dataset names and values are lists of sigma profile dataframes
        e.g., {'Ferronematic': sigma_profiles_fn, 'Nematic': sigma_profiles_n}
    sigma_indices_dict : dict
        Dictionary where keys are dataset names and values are lists of molecule indices
        e.g., {'Ferronematic': sigma_profiles_indices_fn, 'Nematic': sigma_profiles_indices_n}
    title : str, optional
        Main title for the plot
    save_name : str, optional
        Name to save the figure (without extension)
    figures_dir : Path, optional
        Directory to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    
    # Set seaborn style
    sns.set_style("ticks")
    sns.set_context("talk")
    
    # Use seaborn color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    print("Calculating average sigma profiles...")
    
    for i, (dataset_name, sigma_profiles) in enumerate(sigma_profiles_dict.items()):
        if not sigma_profiles:
            print(f"No sigma profiles found for {dataset_name}")
            continue
            
        print(f"\nProcessing {dataset_name} dataset ({len(sigma_profiles)} profiles)")
        
        # Find the sigma column (assuming 'pA' is the main sigma profile column)
        profile_col = None
        sigma_col = None
        
        for col in sigma_profiles[0].columns:
            if col.lower() == 'pa':
                profile_col = col
                break
            elif 'sigma' in col.lower() and 'profile' in col.lower():
                profile_col = col
                break
        
        # Also get the sigma values (x-axis)
        for col in sigma_profiles[0].columns:
            if col.lower() == 'sigma':
                sigma_col = col
                break
        
        if not profile_col or not sigma_col:
            print(f"Could not find sigma profile columns in {dataset_name}")
            print(f"Available columns: {sigma_profiles[0].columns.tolist()}")
            continue
        
        # Collect all profiles for this dataset
        all_profiles = []
        sigma_values = None
        
        for sp in sigma_profiles:
            if profile_col in sp.columns and sigma_col in sp.columns:
                if sigma_values is None:
                    sigma_values = sp[sigma_col].values
                
                # Verify sigma values are consistent
                if not np.array_equal(sp[sigma_col].values, sigma_values):
                    print(f"Warning: Inconsistent sigma values in {dataset_name}")
                    continue
                
                all_profiles.append(sp[profile_col].values)
        
        if not all_profiles:
            print(f"No valid profiles found for {dataset_name}")
            continue
        
        # Convert to numpy array for easier calculations
        profiles_array = np.array(all_profiles)  # Shape: (n_molecules, n_sigma_points)
        
        # Calculate statistics
        mean_profile = np.mean(profiles_array, axis=0)
        std_profile = np.std(profiles_array, axis=0)
        n_profiles = len(all_profiles)
        
        # Calculate 95% confidence interval
        sem_profile = std_profile / np.sqrt(n_profiles)  # Standard error of mean
        ci_95 = stats.t.ppf(0.975, n_profiles - 1) * sem_profile  # 95% CI
        
        # Plot mean profile with seaborn styling
        color = colors[i]
        ax.plot(sigma_values, mean_profile, color=color, linewidth=3, 
                label=f'{dataset_name} (n={n_profiles})', alpha=0.8)
        
        # Plot standard deviation band
        ax.fill_between(sigma_values, 
                       mean_profile - std_profile, 
                       mean_profile + std_profile, 
                       alpha=0.2, color=color)
                
        print(f"  Mean profile range: {mean_profile.min():.4f} to {mean_profile.max():.4f}")
        print(f"  Average std: {np.mean(std_profile):.4f}")
        print(f"  Average 95% CI width: {np.mean(ci_95) * 2:.4f}")
    
    # Customize plot with seaborn styling
    ax.set_xlabel(r'$\sigma$ (e/Å$^2$)', fontsize=14)
    ax.set_ylabel(r'$p(\sigma)$', fontsize=14)
    
    # Move legend inside the figure
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=16, pad=20, fontweight='bold')
    else:
        ax.set_title('Average Sigma Profiles Comparison', fontsize=16, pad=20, fontweight='bold')
    
    # Add vertical lines at σ = ±0.084 with seaborn styling
    ax.axvline(x=0.0084, color='gray', linestyle='--', alpha=0.5, zorder=1)
    ax.axvline(x=-0.0084, color='gray', linestyle='--', alpha=0.5, zorder=1)
    
    # Add text annotations with consistent styling
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    
    ax.text(0.3, 0.95, r'$\sigma = -0.0084$ e/Å$^2$', 
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, rotation=90)
    
    ax.text(0.67, 0.95, r'$\sigma = 0.0084$ e/Å$^2$',
            transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', bbox=props, rotation=90)
    
    plt.tight_layout()
    
    # Save figure if save_name and figures_dir provided
    if save_name and figures_dir:
        save_path = figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    plt.show()
    return fig

def analyze_scaling_factors(combined_df, reference_property='MolWt', test_properties=None, 
                           reference_dataset='Nematic', target_dataset='Ferronematic',
                           title=None, save_name=None, figures_dir=None, figsize=(15, 10)):
    """
    Calculate scaling factor based on one property distribution and test if it applies to other properties.
    
    Parameters:
    -----------
    combined_df : pd.DataFrame
        Combined dataframe with Dataset column
    reference_property : str
        Property to use for calculating the scaling factor (e.g., 'MolWt')
    test_properties : list, optional
        List of properties to test the scaling factor on (e.g., ['dipole_moment_Boltzmann_average'])
    reference_dataset : str
        Dataset to use as reference (default: 'Nematic')
    target_dataset : str
        Dataset to scale to match reference (default: 'Ferronematic')
    title : str, optional
        Main title for the plot
    save_name : str, optional
        Name to save the figure (without extension)
    figures_dir : Path, optional
        Directory to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    
    Returns:
    --------
    dict: Dictionary containing scaling analysis results
    """
    from scipy import stats
    
    if test_properties is None:
        test_properties = ['dipole_moment_Boltzmann_average', 'polarizability_Boltzmann_average']
    
    # Set seaborn style
    sns.set_style("ticks")
    sns.set_context("talk")
    
    # Filter datasets
    ref_data = combined_df[combined_df['Dataset'] == reference_dataset]
    target_data = combined_df[combined_df['Dataset'] == target_dataset]
    
    if ref_data.empty or target_data.empty:
        print(f"Error: No data found for {reference_dataset} or {target_dataset}")
        return None
    
    print(f"SCALING FACTOR ANALYSIS")
    print(f"{'='*60}")
    print(f"Reference dataset: {reference_dataset} (n={len(ref_data)})")
    print(f"Target dataset: {target_dataset} (n={len(target_data)})")
    print(f"Reference property: {reference_property}")
    print(f"Test properties: {test_properties}")
    
    # Calculate scaling factor based on reference property
    if reference_property not in ref_data.columns or reference_property not in target_data.columns:
        print(f"Error: {reference_property} not found in datasets")
        return None
    
    ref_values = ref_data[reference_property].dropna()
    target_values = target_data[reference_property].dropna()
    
    if ref_values.empty or target_values.empty:
        print(f"Error: No valid values for {reference_property}")
        return None
    
    # Calculate different scaling factors
    scaling_factors = {
        'mean_ratio': ref_values.mean() / target_values.mean(),
        'median_ratio': ref_values.median() / target_values.median(),
        'std_ratio': ref_values.std() / target_values.std(),
        'range_ratio': (ref_values.max() - ref_values.min()) / (target_values.max() - target_values.min())
    }
    
    print(f"\nSCALING FACTORS based on {reference_property}:")
    for factor_name, factor_value in scaling_factors.items():
        print(f"  {factor_name}: {factor_value:.4f}")
    
    # Use mean ratio as the primary scaling factor
    primary_scaling_factor = scaling_factors['mean_ratio']
    
    # Create subplots
    n_properties = len(test_properties) + 1  # +1 for reference property
    n_cols = min(3, n_properties)
    n_rows = (n_properties + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    results = {'scaling_factors': scaling_factors, 'property_analysis': {}}
    
    # Plot reference property
    ax = axes[0]
    
    # Original distributions
    sns.histplot(data=ref_data, x=reference_property, alpha=0.6, 
                label=f'{reference_dataset}', color=colors[0], stat='density', ax=ax)
    sns.histplot(data=target_data, x=reference_property, alpha=0.6, 
                label=f'{target_dataset}', color=colors[1], stat='density', ax=ax)
    
    # Scaled target distribution
    scaled_target_values = target_values * primary_scaling_factor
    ax.hist(scaled_target_values, alpha=0.6, density=True, 
           label=f'{target_dataset} (scaled)', color=colors[2], bins=20)
    
    ax.set_title(f'{reference_property}\n(Reference for scaling)')
    ax.legend()
    ax.set_ylabel('Density')
    
    # Calculate goodness of fit for reference property
    ref_mean, ref_std = ref_values.mean(), ref_values.std()
    scaled_mean, scaled_std = scaled_target_values.mean(), scaled_target_values.std()
    
    # KS test between reference and scaled target
    ks_stat, ks_pvalue = stats.ks_2samp(ref_values, scaled_target_values)
    
    results['property_analysis'][reference_property] = {
        'original_target_mean': target_values.mean(),
        'scaled_target_mean': scaled_mean,
        'reference_mean': ref_mean,
        'mean_improvement': abs(ref_mean - scaled_mean) / abs(ref_mean - target_values.mean()),
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue
    }
    
    print(f"\nREFERENCE PROPERTY ({reference_property}) ANALYSIS:")
    print(f"  {reference_dataset} mean: {ref_mean:.4f}")
    print(f"  {target_dataset} original mean: {target_values.mean():.4f}")
    print(f"  {target_dataset} scaled mean: {scaled_mean:.4f}")
    print(f"  KS test p-value: {ks_pvalue:.4f}")
    
    # Test scaling factor on other properties
    for i, prop in enumerate(test_properties, 1):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if prop not in ref_data.columns or prop not in target_data.columns:
            ax.text(0.5, 0.5, f'{prop}\nnot found', ha='center', va='center', 
                   transform=ax.transAxes)
            continue
        
        ref_prop_values = ref_data[prop].dropna()
        target_prop_values = target_data[prop].dropna()
        
        if ref_prop_values.empty or target_prop_values.empty:
            ax.text(0.5, 0.5, f'{prop}\nno valid data', ha='center', va='center', 
                   transform=ax.transAxes)
            continue
        
        # Original distributions
        sns.histplot(data=ref_data, x=prop, alpha=0.6, 
                    label=f'{reference_dataset}', color=colors[0], stat='density', ax=ax)
        sns.histplot(data=target_data, x=prop, alpha=0.6, 
                    label=f'{target_dataset}', color=colors[1], stat='density', ax=ax)
        
        # Apply scaling factor
        scaled_prop_values = target_prop_values * primary_scaling_factor
        ax.hist(scaled_prop_values, alpha=0.6, density=True, 
               label=f'{target_dataset} (scaled)', color=colors[2], bins=20)
        
        # Calculate goodness of fit
        ref_prop_mean = ref_prop_values.mean()
        scaled_prop_mean = scaled_prop_values.mean()
        
        # KS test
        ks_stat_prop, ks_pvalue_prop = stats.ks_2samp(ref_prop_values, scaled_prop_values)
        
        # Calculate improvement in mean difference
        original_diff = abs(ref_prop_mean - target_prop_values.mean())
        scaled_diff = abs(ref_prop_mean - scaled_prop_mean)
        improvement = (original_diff - scaled_diff) / original_diff if original_diff > 0 else 0
        
        results['property_analysis'][prop] = {
            'original_target_mean': target_prop_values.mean(),
            'scaled_target_mean': scaled_prop_mean,
            'reference_mean': ref_prop_mean,
            'mean_improvement': improvement,
            'ks_statistic': ks_stat_prop,
            'ks_pvalue': ks_pvalue_prop
        }
        
        ax.set_title(f'{prop}\n(Improvement: {improvement:.1%})')
        ax.legend()
        ax.set_ylabel('Density')
        
        print(f"\n{prop.upper()} ANALYSIS:")
        print(f"  {reference_dataset} mean: {ref_prop_mean:.4f}")
        print(f"  {target_dataset} original mean: {target_prop_values.mean():.4f}")
        print(f"  {target_dataset} scaled mean: {scaled_prop_mean:.4f}")
        print(f"  Mean improvement: {improvement:.1%}")
        print(f"  KS test p-value: {ks_pvalue_prop:.4f}")
    
    # Remove empty subplots
    for idx in range(n_properties, len(axes)):
        fig.delaxes(axes[idx])
    
    if title:
        plt.suptitle(title, fontsize=16, y=0.98)
    else:
        plt.suptitle(f'Scaling Factor Analysis\n(Primary factor: {primary_scaling_factor:.4f})', 
                    fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save figure if save_name and figures_dir provided
    if save_name and figures_dir:
        save_path = figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    plt.show()
    
    # Summary
    print(f"\n{'='*60}")
    print("SCALING FACTOR SUMMARY")
    print(f"{'='*60}")
    print(f"Primary scaling factor (mean ratio): {primary_scaling_factor:.4f}")
    print("\nProperty scaling effectiveness:")
    for prop, analysis in results['property_analysis'].items():
        print(f"  {prop}: {analysis['mean_improvement']:.1%} improvement")
    
    return results


def calculate_simple_scaling_factor(combined_df, reference_property='MolWt', 
                                   test_properties=None, dataset1='Nematic', dataset2='Ferronematic',
                                   title=None, save_name=None, figures_dir=None, figsize=(12, 8)):
    """
    Calculate simple scaling factor to match one dataset to another and test on other properties.
    
    Parameters:
    -----------
    combined_df : pd.DataFrame
        Combined dataframe with Dataset column
    reference_property : str
        Property to calculate scaling factor from (e.g., 'MolWt')
    test_properties : list, optional
        Properties to test the scaling factor on
    dataset1 : str
        First dataset name (will be scaled)
    dataset2 : str
        Second dataset name (target to match)
    title : str, optional
        Plot title
    save_name : str, optional
        Save name for figure
    figures_dir : Path, optional
        Directory to save figure
    figsize : tuple, optional
        Figure size
    
    Returns:
    --------
    dict: Results including scaling factor and property comparisons
    """
    if test_properties is None:
        test_properties = ['dipole_moment_Boltzmann_average']
    
    # Set seaborn style
    sns.set_style("ticks")
    sns.set_context("talk")
    
    # Get data for each dataset
    data1 = combined_df[combined_df['Dataset'] == dataset1]
    data2 = combined_df[combined_df['Dataset'] == dataset2]
    
    if data1.empty or data2.empty:
        print(f"Error: No data found for {dataset1} or {dataset2}")
        return None
    
    # Calculate scaling factor
    mean1 = data1[reference_property].mean()
    mean2 = data2[reference_property].mean()
    scaling_factor = mean2 / mean1
    
    print(f"SIMPLE SCALING FACTOR ANALYSIS")
    print(f"{'='*50}")
    print(f"{dataset1} mean {reference_property}: {mean1:.2f}")
    print(f"{dataset2} mean {reference_property}: {mean2:.2f}")
    print(f"Scaling factor ({dataset1} → {dataset2}): {scaling_factor:.4f}")
    print(f"This means {dataset2} molecules are {scaling_factor:.2f}x larger on average")
    
    # Create subplots
    n_plots = len(test_properties) + 1  # +1 for reference property
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    results = {'scaling_factor': scaling_factor, 'reference_property': reference_property}
    
    # Plot reference property
    ax = axes[0]
    
    # Original data
    ax.hist(data1[reference_property].dropna(), alpha=0.6, label=f'{dataset1}', 
           density=True, bins=20, color='#1f77b4')
    ax.hist(data2[reference_property].dropna(), alpha=0.6, label=f'{dataset2}', 
           density=True, bins=20, color='#ff7f0e')
    
    # Scaled data
    scaled_data1 = data1[reference_property] * scaling_factor
    ax.hist(scaled_data1.dropna(), alpha=0.6, label=f'{dataset1} (scaled)', 
           density=True, bins=20, color='#2ca02c')
    
    ax.set_title(f'{reference_property}\n(Scaling Factor: {scaling_factor:.3f})')
    ax.legend()
    ax.set_ylabel('Density')
    
    # Test on other properties
    for i, prop in enumerate(test_properties, 1):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if prop not in data1.columns or prop not in data2.columns:
            ax.text(0.5, 0.5, f'{prop}\nnot available', ha='center', va='center', 
                   transform=ax.transAxes)
            continue
        
        # Original means
        prop_mean1 = data1[prop].mean()
        prop_mean2 = data2[prop].mean()
        
        # Scaled mean
        scaled_prop_mean1 = prop_mean1 * scaling_factor
        
        # How close does scaling get us?
        original_diff = abs(prop_mean2 - prop_mean1)
        scaled_diff = abs(prop_mean2 - scaled_prop_mean1)
        improvement = (original_diff - scaled_diff) / original_diff * 100 if original_diff > 0 else 0
        
        # Plot distributions
        ax.hist(data1[prop].dropna(), alpha=0.6, label=f'{dataset1}', 
               density=True, bins=20, color='#1f77b4')
        ax.hist(data2[prop].dropna(), alpha=0.6, label=f'{dataset2}', 
               density=True, bins=20, color='#ff7f0e')
        
        scaled_prop_data1 = data1[prop] * scaling_factor
        ax.hist(scaled_prop_data1.dropna(), alpha=0.6, label=f'{dataset1} (scaled)', 
               density=True, bins=20, color='#2ca02c')
        
        ax.set_title(f'{prop}\n(Improvement: {improvement:.1f}%)')
        ax.legend()
        ax.set_ylabel('Density')
        
        results[prop] = {
            'original_mean1': prop_mean1,
            'original_mean2': prop_mean2,
            'scaled_mean1': scaled_prop_mean1,
            'improvement_percent': improvement
        }
        
        print(f"\n{prop}:")
        print(f"  {dataset1} mean: {prop_mean1:.4f}")
        print(f"  {dataset2} mean: {prop_mean2:.4f}")
        print(f"  {dataset1} scaled mean: {scaled_prop_mean1:.4f}")
        print(f"  Improvement: {improvement:.1f}%")
    
    # Remove empty subplots
    for idx in range(n_plots, len(axes)):
        fig.delaxes(axes[idx])
    
    if title:
        plt.suptitle(title, fontsize=16, y=0.98)
    else:
        plt.suptitle(f'Scaling Factor Analysis\n{dataset1} × {scaling_factor:.3f} → {dataset2}', 
                    fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save figure
    if save_name and figures_dir:
        save_path = figures_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    plt.show()
    
    return results


def create_combined_dataset_with_spectrum_profiles(datasets_dict, spectrum_profiles_dict, spectrum_indices_dict):
    """
    Create a combined dataframe with dataset identifiers and aggregated spectrum profile data.
    
    Parameters:
    -----------
    datasets_dict : dict
        Dictionary where keys are dataset names and values are dataframes
        e.g., {'Ferronematic': df_fn, 'Nematic': df_n}
    spectrum_profiles_dict : dict
        Dictionary where keys are dataset names and values are lists of spectrum profile dataframes
        e.g., {'Ferronematic': spectrum_profiles_fn, 'Nematic': spectrum_profiles_n}
    spectrum_indices_dict : dict
        Dictionary where keys are dataset names and values are lists of molecule indices
        e.g., {'Ferronematic': spectrum_profiles_indices_fn, 'Nematic': spectrum_profiles_indices_n}
    
    Returns:
    --------
    combined_df : pd.DataFrame
        Combined dataframe with dataset identifiers and spectrum profile aggregations
    spectrum_profile_summary : dict
        Summary statistics about the spectrum profile aggregation
    """
    combined_data = []
    spectrum_summary = {}
    
    print("Creating combined dataset with spectrum profiles...")
    print("="*60)
    
    for dataset_name, df in datasets_dict.items():
        print(f"\nProcessing {dataset_name} dataset...")
        
        # Add dataset identifier
        temp_df = df.copy()
        temp_df['Dataset'] = dataset_name
        temp_df['Original_Index'] = temp_df.index
        
        # Get spectrum profiles for this dataset
        spectrum_profiles = spectrum_profiles_dict.get(dataset_name, [])
        spectrum_indices = spectrum_indices_dict.get(dataset_name, [])
        
        # Initialize spectrum profile columns
        spectrum_cols_added = []
        
        if spectrum_profiles and spectrum_indices:
            print(f"  Found {len(spectrum_profiles)} spectrum profiles")
            
            # Get column names from first spectrum profile
            first_profile = spectrum_profiles[0]
            spectrum_columns = [col for col in first_profile.columns if 'eigenvalue' in col.lower() or 'squared_coeff' in col.lower()]
            
            print(f"  Spectrum columns found: {spectrum_columns}")
            
            # Initialize spectrum profile columns in dataframe
            for col in spectrum_columns:
                temp_df[f'spectrum_{col}_mean'] = np.nan
                temp_df[f'spectrum_{col}_std'] = np.nan
                temp_df[f'spectrum_{col}_min'] = np.nan
                temp_df[f'spectrum_{col}_max'] = np.nan
                temp_df[f'spectrum_{col}_median'] = np.nan
                spectrum_cols_added.extend([f'spectrum_{col}_mean', f'spectrum_{col}_std', 
                                          f'spectrum_{col}_min', f'spectrum_{col}_max', f'spectrum_{col}_median'])
            
            # Map spectrum profiles to dataframe rows
            profiles_mapped = 0
            for i, mol_idx in enumerate(spectrum_indices):
                if mol_idx < len(temp_df):
                    profile = spectrum_profiles[i]
                    
                    for col in spectrum_columns:
                        if col in profile.columns:
                            values = profile[col].values
                            temp_df.loc[mol_idx, f'spectrum_{col}_mean'] = np.mean(values)
                            temp_df.loc[mol_idx, f'spectrum_{col}_std'] = np.std(values)
                            temp_df.loc[mol_idx, f'spectrum_{col}_min'] = np.min(values)
                            temp_df.loc[mol_idx, f'spectrum_{col}_max'] = np.max(values)
                            temp_df.loc[mol_idx, f'spectrum_{col}_median'] = np.median(values)
                    
                    profiles_mapped += 1
            
            print(f"  Mapped {profiles_mapped} spectrum profiles to dataframe rows")
            spectrum_summary[dataset_name] = {
                'total_profiles': len(spectrum_profiles),
                'mapped_profiles': profiles_mapped,
                'spectrum_columns': spectrum_columns,
                'aggregated_columns': spectrum_cols_added
            }
        else:
            print(f"  No spectrum profiles found for {dataset_name}")
            spectrum_summary[dataset_name] = {
                'total_profiles': 0,
                'mapped_profiles': 0,
                'spectrum_columns': [],
                'aggregated_columns': []
            }
        
        combined_data.append(temp_df)
    
    # Combine all datasets
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    print(f"\n{'='*60}")
    print("COMBINED DATASET SUMMARY")
    print("="*60)
    print(f"Total rows: {len(combined_df)}")
    print(f"Datasets: {combined_df['Dataset'].value_counts().to_dict()}")
    
    # Count spectrum profile coverage
    spectrum_cols = [col for col in combined_df.columns if col.startswith('spectrum_')]
    if spectrum_cols:
        print(f"Spectrum profile columns added: {len(spectrum_cols)}")
        print(f"Rows with spectrum data: {combined_df[spectrum_cols[0]].notna().sum()}")
    
    return combined_df, spectrum_summary


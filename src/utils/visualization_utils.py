import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import dendrogram
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
import py3Dmol
from PIL import Image
import io

# Load default style for all plots
_style_path = Path(__file__).parent / 'jacs_fig.mplstyle'
if _style_path.exists():
    plt.style.use(str(_style_path))

def draw_with_spheres(mol):
    v = py3Dmol.view(width=300,height=300)
    IPythonConsole.addMolToView(mol,v)
    v.zoomTo()
    v.setStyle({'sphere':{'radius':0.3},'stick':{'radius':0.2}});
    v.show()


def plot_molecular_signal(mol: Chem.Mol, charges: list | np.ndarray,
                          title: str = "Molecular Structure and Signal",
                          ) -> plt.Figure:
    """Plot molecular structure (left) and graph signal (right) with charges."""
    # Get adjacency matrix and coordinates
    adj_matrix = Chem.GetAdjacencyMatrix(mol).astype(float)
    AllChem.Compute2DCoords(mol)
    coords = np.array([[mol.GetConformer().GetAtomPosition(atom.GetIdx()).x,
                       mol.GetConformer().GetAtomPosition(atom.GetIdx()).y]
                      for atom in mol.GetAtoms()])
    
    # Create figure with two subplots - let axes adjust to data
    # First axis (molecular structure) occupies 1/4 of height, second axis 3/4
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 3), 
                                    gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.15})
    
    # Get axis dimensions in pixels to size the molecular structure appropriately
    # Figure is 4 inches wide, first subplot is 1/4 of 3 inches = 0.75 inches tall
    # At 100 DPI: width ~400px, height ~75px (but we'll use a reasonable aspect ratio)
    # Calculate appropriate size based on axis dimensions
    fig_width_inches = 4
    fig_height_inches = 3
    ax1_height_ratio = 1 / (1 + 3)  # 1/4 of total height
    ax1_height_inches = fig_height_inches * ax1_height_ratio * 0.9  # 90% to account for spacing
    ax1_width_inches = fig_width_inches * 0.9  # 90% to account for margins
    
    # Convert to pixels at reasonable DPI for display
    dpi_display = 600
    img_width = int(ax1_width_inches * dpi_display)
    img_height = int(ax1_height_inches * dpi_display)
    
    # Top subplot: molecular structure
    # Use Cairo drawer for better control over bond thickness and font size
    d2d = Draw.MolDraw2DCairo(img_width, img_height)
    dopts = d2d.drawOptions()
    dopts.addAtomIndices = False
    dopts.bondLineWidth = 4.0  # Thick bonds
    dopts.minFontSize = 20  # Larger minimum font
    dopts.maxFontSize = 28  # Larger maximum font
    dopts.baseFontSize = 1.2  # Larger base font
    dopts.scalingFactor = 400  # Higher scaling for better resolution
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    
    # Convert to PIL Image then numpy array for matplotlib
    img_data = d2d.GetDrawingText()
    img = Image.open(io.BytesIO(img_data))
    img_array = np.array(img)
    
    # Display the image to fill the subplot
    ax1.imshow(img_array, aspect='auto', origin='upper', extent=[0, img_width, img_height, 0])
    ax1.set_xlim(0, img_width)
    ax1.set_ylim(img_height, 0)
    ax1.axis('off')
    
    # Bottom subplot: signal plot
    charges_arr = np.asarray(charges)
    # Plot edges (bonds)
    for i in range(len(mol.GetAtoms())):
        for j in range(i+1, len(mol.GetAtoms())):
            if adj_matrix[i, j] > 0:
                ax2.plot([coords[i, 0], coords[j, 0]], 
                        [coords[i, 1], coords[j, 1]], 
                        'k--', linewidth=1, alpha=0.6)
    # Plot nodes (atoms) colored by charges
    scatter = ax2.scatter(coords[:, 0], coords[:, 1], 
                         c=charges_arr, s=50, cmap='RdBu', 
                         edgecolors='black', linewidths=1.5, zorder=3)
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.6, orientation='horizontal')
    cbar.ax.set_title('Partial charges', fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    # Add atom labels
    for i, atom in enumerate(mol.GetAtoms()):
        ax2.annotate(atom.GetSymbol(), (coords[i, 0], coords[i, 1]),
                    xytext=(0, 0), textcoords='offset points', fontsize=6, fontweight='bold',
                    ha='center', va='center', color='#f9f9f9', zorder=4)
    ax2.set_title('Graph and Partial Charges')
    # Let axes auto-adjust to data with some padding
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    padding = 0.25
    x_min = coords[:, 0].min() - padding * x_range
    x_max = coords[:, 0].max() + padding * x_range
    y_min = coords[:, 1].min() - padding * y_range
    y_max = coords[:, 1].max() + padding * y_range
    
    # Set limits first, then adjust for equal aspect
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_aspect('equal', adjustable='box')
    ax2.axis('off')
    
    fig.suptitle(title, y=0.995, ha='center', va='top', x=0.5)
    fig.tight_layout()
    return fig 

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

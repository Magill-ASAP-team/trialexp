#%%
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from snakehelper.SnakeIOHelper import getSnake
from trialexp import config
from kilosort.io import load_ops
from spikeinterface.core.template import Templates  
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import trialexp.process.ephys.plot_utils as ephys_plot 
#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [config.debug_folder + r'/processed/df_celltypes.pkl'],
  'cell_classification')

#%%
df_quality_metrics = pd.read_pickle(sinput.df_quality_metrics)
    
#%% Cell type classification based on properties
def classify_celltype(cell):
    # Basd on Andy Peters 202
    cell_type = None

    if cell['peak_to_valley'] <= 0.4/1000:
        # Narrow spike: FSI or UIN
        if cell['long_isi_portion'] > 0.1:
            cell_type = 'UIN'
        else:
            # No long pause, almost tunic firing
            cell_type = 'FSI'
        
    else:
        if cell['post_spike_suppression_ms'] > 40:
            cell_type = 'TAN'
        else:
            cell_type = 'MSN'
    
    return cell_type

def classify_spike_origin(cell):
    # Based on Ye et al. 2025
    # Axon recording in Neuropixel 1.0 should be around 6%
    origin = None
    if cell['peak_to_valley'] <= 0.4/1000:
        # narrow spikes
        if cell['spatial_footprint_um'] <= 20:
            origin = 'axon' 
        else:
            origin = 'soma' # interneuron
    else:
        origin = 'soma'
        
    return origin
        
    

df_quality_metrics['cell_type'] = df_quality_metrics.apply(classify_celltype, axis=1)
df_quality_metrics['spike_origin'] = df_quality_metrics.apply(classify_spike_origin, axis=1)

#%% Plot a pie chart of cell type distribution
import matplotlib.pyplot as plt

cell_type_counts = df_quality_metrics['cell_type'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(cell_type_counts.values, labels=cell_type_counts.index, autopct='%1.1f%%')
plt.title('Cell Type Distribution')
plt.axis('equal')
plt.show()

# Plot a pie chart of spike origin distribution
spike_origin_counts = df_quality_metrics['spike_origin'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(spike_origin_counts.values, labels=spike_origin_counts.index, autopct='%1.1f%%')
plt.title('Spike Origin Distribution')
plt.axis('equal')
plt.show()
            
#%% Create pairplot for cell type classification features
plt.figure()
pairplot_data = df_quality_metrics[['peak_to_valley', 
                                    'long_isi_portion', 
                                    'post_spike_suppression_ms',
                                    'spatial_footprint_um',
                                    'firing_rate',
                                    'cell_type']].copy()
# Convert peak_to_valley from seconds to milliseconds for better readability
pairplot_data['peak_to_valley'] = pairplot_data['peak_to_valley'] * 1000

sns.pairplot(pairplot_data,  diag_kind='hist', corner=True, diag_kws={'bins':50})
plt.suptitle('Cell Type Classification Features', y=1.01)
plt.show()
 

#%% Double check the spike shapes

#%%

# Prepare features for DBSCAN clustering
feature_cols = ['peak_to_valley', 'long_isi_portion', 'post_spike_suppression_ms', 
        'spatial_footprint_um', 'firing_rate']
X = df_quality_metrics[feature_cols].copy()

# Standardize features (important for DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find the best eps parameter for DBSCAN

# Calculate k-nearest neighbors distances (k = min_samples)
k = 5
nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)

# Sort and plot the k-distance graph
distances = np.sort(distances[:, k-1], axis=0)
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel('Points sorted by distance')
plt.ylabel(f'{k}-NN Distance')
plt.title('K-distance Graph for eps Selection')
plt.grid(True)
plt.show()

# Try multiple eps values and evaluate
eps_values = np.arange(0.3, 2.0, 0.1)
results = []

for eps in eps_values:
  dbscan = DBSCAN(eps=eps, min_samples=5)
  labels = dbscan.fit_predict(X_scaled)
  n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
  n_noise = sum(labels == -1)
  results.append({'eps': eps, 'n_clusters': n_clusters, 'n_noise': n_noise})

results_df = pd.DataFrame(results)
print(results_df)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(results_df['eps'], results_df['n_clusters'], marker='o')
axes[0].set_xlabel('eps')
axes[0].set_ylabel('Number of Clusters')
axes[0].grid(True)

axes[1].plot(results_df['eps'], results_df['n_noise'], marker='o', color='red')
axes[1].set_xlabel('eps')
axes[1].set_ylabel('Number of Noise Points')
axes[1].grid(True)
plt.tight_layout()
plt.show()

# Apply DBSCAN with chosen eps (adjust based on plots above)
best_eps = 0.8  # Adjust this based on the plots
dbscan = DBSCAN(eps=best_eps, min_samples=5)
df_quality_metrics['dbscan_cluster'] = dbscan.fit_predict(X_scaled)

# Print cluster statistics
print(f"\nUsing eps={best_eps}:")
print(f"Number of clusters: {len(set(df_quality_metrics['dbscan_cluster'])) - (1 if -1 in df_quality_metrics['dbscan_cluster'] else 0)}")
print(f"Number of noise points: {sum(df_quality_metrics['dbscan_cluster'] == -1)}")
print(df_quality_metrics['dbscan_cluster'].value_counts().sort_index())

# Create pairplot colored by DBSCAN clusters
pairplot_dbscan = df_quality_metrics[feature_cols + ['dbscan_cluster']].copy()
pairplot_dbscan['peak_to_valley'] = pairplot_dbscan['peak_to_valley'] * 1000
pairplot_dbscan['dbscan_cluster'] = pairplot_dbscan['dbscan_cluster'].astype(str)

sns.pairplot(pairplot_dbscan, hue='dbscan_cluster', diag_kind='hist', corner=True, 
    diag_kws={'bins':50}, palette='tab10')
plt.suptitle('DBSCAN Clustering of Cell Features', y=1.01)
plt.show()

#%%
template = np.array(df_quality_metrics.iloc[0].templates)
plt.imshow(template.T)
plt.gca().invert_yaxis()

max_chan = df_quality_metrics.iloc[0]['extremum_channel']

print(max_chan)


#%%
cellidx2plot = 1

cell = df_quality_metrics.iloc[cellidx2plot]
template = np.array(cell.templates)
max_chan = cell['extremum_channel']
chan_locs = df_quality_metrics.attrs['chan_locs']


ephys_plot.plot_template(template, max_chan, chan_locs)

#%%
# plot some MSN
df_msn = df_quality_metrics[df_quality_metrics['cell_type']=='MSN']

cell = df_msn.iloc[1]
template = np.array(cell.templates)
max_chan = cell['extremum_channel']
ephys_plot.plot_template(template, max_chan, chan_locs)

import pandas as pd
import matplotlib.patches as patches
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

def get_region_boundary(df_cell, dep_col,group_method='consecutive'):
    df_cell = df_cell.sort_values(dep_col)
    df_cell['group'] = (df_cell['name']!=df_cell['name'].shift()).cumsum() #detect region boundaries
    #Find the region boundaries
    def get_boundary(df,method):
        if method =='consecutive':
            return pd.Series({'min_mm': df[dep_col].min(),
                    'max_mm': df[dep_col].max(),
                    'name':df.iloc[0]['name'],
                'acronym': df.iloc[0].acronym})
        else:
           return pd.Series({'min_mm': df[dep_col].min(),
                    'max_mm': df[dep_col].max()}) 
    
    if group_method == 'consecutive':
        region_boundary = df_cell.groupby(['group']).apply(get_boundary, method=group_method)
    else:
        region_boundary = df_cell.groupby(['name','acronym']).apply(get_boundary, method=group_method)

    region_boundary = region_boundary.sort_values('min_mm').reset_index()
    return region_boundary

def assign_region_layer(df):
    # assign non-overlapping regions to its own layer for easier plotting later

    region_boundary = df.copy()

    region_boundary['layer'] = -1
    region_boundary.loc[0,'layer'] = 0
    
    def check_overlap(region1, region2):
      return not ((region1.min_mm > region2.max_mm) or (region1.max_mm < region2.min_mm))
    
    for idx, region in region_boundary.iloc[1:].iterrows():
    
        # loop through all the existing layer
        for i in range(region_boundary.layer.max()+1):
            region_in_layer = region_boundary[region_boundary.layer == i]
    
            #check if the current region can be added
            if not any([check_overlap(region, r) for _,r in region_in_layer.iterrows()]):
                region_boundary.loc[idx, 'layer'] = i
                break
        else: #execute when for loop is finished, i.e. no break is encountered
            region_boundary.loc[idx, 'layer'] = i+1        

    return region_boundary

def add_regions(ax, region_boundary, dv_bins):
    colorIdx = 0
    dv_bin_size = np.mean(np.diff(dv_bins))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    init_x_lim = xlim[1]
    colors = plt.cm.tab20.colors
    rect_width = (xlim[1]-xlim[0])*0.2
    
    unique_region = region_boundary.acronym.unique()
    color_table = {r:colors[i%len(colors)] for i,r in enumerate(unique_region)}
    
    for idx, row in region_boundary.iterrows():
        # convert the dv coordinates to the bin coordinates so that the plot looks right
        y = (row.min_mm - dv_bins[0])/dv_bin_size
        height = (row.max_mm - row.min_mm)/dv_bin_size
        x = init_x_lim+rect_width+row.layer*rect_width
        region = row.acronym
    
        rect = patches.Rectangle((x, y), rect_width, height, color=color_table[region])
        ax.add_patch(rect)
        colorIdx += 1
    
        # also add the region text
        t = ax.text(x+rect_width/4, y+height/2, region, fontsize=10)

    # expand the xlim
    ax.set_xlim([0, init_x_lim+rect_width*1.2+(region_boundary.layer.max()+1)*rect_width])
    ax.set_ylim([ylim[0]+1, ylim[1]])
    return ax


def format_cell4merge(df_cell):
    # Convert the dataframe into proper format for merging

    df_cell = df_cell.copy()

    def get_session_date(session_id):
        if type(session_id) is str:
        # only return the date of the session
            return '-'.join(session_id.split('-')[:-1])
    
    # extract the session ID and probe name from the cluID so that we can merge to Sharptrack results
    df_cell[['session_id','probe','id']] = df_cell.cluID.str.split('_',expand=True)
    df_cell['session_date'] = df_cell['session_id'].apply(get_session_date)
    df_cell['probe']  = df_cell['probe'].str.replace('Probe','')
    return df_cell

def draw_region_legend(ax, region_boundary):
    y = ax.get_ylim()[1] +2
    x = ax.get_xlim()[1]+2
    
    for idx, region in region_boundary.iterrows():
        ax.text(x,y, f'{region.acronym}: {region["name"]}')
        y += 1
    

def plot_firing_rate_regions(df_cell, depth_col='depth_mm', group_method='consecutive'):
    # plot firing rate of brain regions with different depth
    df_cell = df_cell.copy()
    df_cell['depth_group'], dv_bins = pd.cut(df_cell[depth_col],30, retbins=True)
    region_boundary = get_region_boundary(df_cell, depth_col,group_method)
    region_boundary = assign_region_layer(region_boundary)
    # display(region_boundary)
    
    plt.figure(figsize=(8,max(len(region_boundary)*0.6,12)),dpi=200)
    ax = sns.barplot(df_cell, y='depth_group', x='firingRate')
    
    #set a consistent max rate so that figures from different sessions are comparable
    max_rate = ax.get_xlim()[1]
    ax.set_xlim([0,max(40,max_rate)])
    
    ax = add_regions(ax, region_boundary, dv_bins)
    ax.set(ylabel=depth_col, xlabel='Firing Rate (Hz)')
    
    draw_region_legend(ax, region_boundary)
    return ax
    
def get_session_date(session_id):
    if type(session_id) is str:
    # only return the date of the session
        return '-'.join(session_id.split('-')[:-1])
    
def ccfmm2ccf(ap_mm, dv_mm, ml_mm, bregma=[5400, 0, 5700], atlas_resolution=0.001):
    # covert back the ap_mm etc. back to the original CCF coordinate
    ap = (-ap_mm/atlas_resolution+bregma[0])
    dv = (dv_mm/atlas_resolution+bregma[1])
    ml = (ml_mm/atlas_resolution+bregma[2])
    
    return ap,dv,ml

def ccf2ccfmm(ap, dv, ml, bregma=[5400, 0, 5700]):
    # convert breg centered coordinate back to the original CCF coordinate
    # unit for bregma is in um
    ap_mm = -(ap-bregma[0])/1000
    dv_mm = (dv-bregma[1])/1000
    ml_mm = (ml-bregma[2])/1000
    return (ap_mm, dv_mm, ml_mm)

def get_extent(mask, axis='ap'):
    """
    This is a function for calculating the extent of a mask, axis can be AP, ML, or DV

    Args:
        mask (numpy.ndarray): A three-dimensional array where non-zero values represent the region of interest.

    Returns:
        tuple: A tuple containing two elements - the start point (anterior extent) and the end point (posterior extent) of the mask along the AP axis.
    """

    if axis=='AP':
        extent = mask.max(1).max(1) #collapse to AP axis only
    elif axis=='ML':
        extent = mask.max(0).max(0)
    
    idx = np.where(extent!=0)[0] # find the beginning and end point of mask
    start,end = idx[0], idx[-1]
    return (start, end)


def draw_brain_regions(region_names, bg_atlas, ax=None, draw_coord_range=None,
                       plane='transverse', cmap='tab20', alpha=1, hemisphere_only=True):
    # plot specified brain regions
    
    ap_coords, dv_coords, ml_coords = [np.arange(bg_atlas.shape[i])*bg_atlas.resolution[i] for i in range(3)]
    ap_mm_coords, dv_mm_coords, ml_mm_coords = ccf2ccfmm(ap_coords, dv_coords, ml_coords)
    
    if type(region_names) is not list:
        region_names = [region_names]

    #get the region mask
    mask = None
    mask_idx = 1
    for name in region_names:
        region_mask = bg_atlas.get_structure_mask(name)
        if mask is None:
            mask = region_mask
        else:
            mask += (region_mask + mask_idx)
            mask_idx += 1
            

    if draw_coord_range is None:
        if plane == 'transverse':
            start,end = get_extent(mask, axis='AP')
            draw_coord_range = ap_mm_coords[start], ap_mm_coords[end]
        elif plane == 'sagittal':
            start,end = get_extent(mask, axis='ML')
            draw_coord_range = ml_mm_coords[start], ml_mm_coords[end]

    # search for the specified range
    if plane == 'transverse':
        start_idx = np.searchsorted(-ap_mm_coords,-draw_coord_range[0]) # searchsort only works on ascending array, but ap is descending
        end_idx = np.searchsorted(-ap_mm_coords, -draw_coord_range[1])
        if end_idx <= start_idx:
            end_idx = start_idx +1
        
        img = mask[start_idx:end_idx,:,:].max(0)
        
    elif plane =='sagittal':
        start_idx = np.searchsorted(ml_mm_coords, draw_coord_range[0]) # searchsort only works on ascending array, but ap is descending
        end_idx = np.searchsorted(ml_mm_coords, draw_coord_range[1])
        if end_idx <= start_idx:
            end_idx = start_idx +1
            
        img = mask[:,:,start_idx:end_idx].max(2)
        
        
    if hemisphere_only and plane =='transverse':
        img = img[:,:img.shape[1]//2]
    
    if plane == 'transverse':
        extent = [ml_mm_coords[0], 0,dv_mm_coords[-1], 0]
    elif plane =='sagittal':
        extent = [ap_mm_coords[0], ap_mm_coords[-1],dv_mm_coords[-1], 0]
        
        
    if ax is not None:
        ax2plot = ax
    else:
        ax2plot = plt
    
    if plane == 'transverse':
        ax2plot.imshow(img,cmap=cmap, extent=extent, alpha=alpha)
    elif plane == 'sagittal':
        ax2plot.imshow(img.T,cmap=cmap, extent=extent, alpha=alpha)


    return ax, mask, extent

def load_ccf_data(base_directory):
    
    # The annoatation volume is a 3D volume with an index at each voxel
    # it present the row of in the structure_tree_safe_2017 table
    # the template volume contains the reference image of the brain (voxel with each element the brightness of the pixel)
    # the atlas is 1-based, its index corresponds to the row of the structure_tree_safe_2017 table, so 1 is the first row 
    atlas = np.load(base_directory / 'annotation_volume_10um_by_index.npy', mmap_mode='r')
    structure_tree = pd.read_csv(base_directory / 'structure_tree_safe_2017.csv')
    return atlas, structure_tree


def get_region_boundaries(coords,  atlas, structure_tree):
    # the trajectory is between the starting and end point of the coords
    # the end point is not the ending of the probe. Neuropixel probe is about 4000um long if only the tip is used
    trajectory_depth = np.linalg.norm(np.diff(coords, axis=0)) * 10 # axis unit is in 10um, so we convert it back to 1um
    n_coords = int(trajectory_depth) # sample every 1um
    trajector_depth_list = np.linspace(trajectory_depth,0, n_coords)

    coords_sampled = np.zeros((n_coords, 3)).astype(int)
    for i in range(3):
        coords_sampled[:, i] = np.linspace(coords[0, i], coords[1, i], n_coords).astype(int)

    # get the annotation index from the coordinates
    annot = np.zeros((coords_sampled.shape[0],))
    for i in range(coords_sampled.shape[0]):
        annot[i] = atlas[coords_sampled[i, 0], coords_sampled[i, 1], coords_sampled[i, 2]] - 1  # convert it to zero based

    # Find out the boundary regions
    # Find the boundary of the region, where the annot index changes
    regions_bins = [0, *(np.nonzero(np.diff(annot) != 0)[0] + 1), annot.shape[0]-1]  # idx of annot that denotes the boundaries
    region_boundaries = [regions_bins[:-2], regions_bins[1:-1]]  # the beginning and end of the region

    trajectory_areas = structure_tree.loc[annot[region_boundaries[0]]]
    trajectory_areas['depth_start'] = trajector_depth_list[region_boundaries[0]]
    trajectory_areas['depth_end'] = trajector_depth_list[region_boundaries[1]]
    # Note: the actual may be a few microns different from the AP_historylogy results due to
    # slightly different sampling point for the trajectory 
    
    return trajectory_areas

def shift_trajectory_depth(coords, shift_depth, length=np.inf, axis_resolution=10):
    # Calculate the direction vector
    # coords is [start, end] in axis coordinate
    # shift_depth is in um, positive is deeper into the brain
    # if length=np.inf, then by default it is from the surface of the brain to the base
    # otherwise the end point is at the base, and the starting point is calculated back from the length
    
    coords = coords.copy()
    direction_vector = coords[1] - coords[0]
    
    # Normalize the direction vector
    direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)
    # print(np.linalg.norm(coords[1]-coords[0])*axis_resolution)
    # if length is set, compute the correct starting point of the probe 
    if length !=np.inf:
        #calculate the starting point of the coordinate, and then shift by the shift_depth
        coords[0] = coords[1] - direction_vector_normalized * length/axis_resolution
    
    # print(coords)
    # Apply the shift
    coords_shifted = np.zeros_like(coords)

    coords_shifted[0] = coords[0] + shift_depth/axis_resolution * direction_vector_normalized #start piont    
    coords_shifted[1] = coords[1] + shift_depth/axis_resolution * direction_vector_normalized #end point
    
    return coords_shifted
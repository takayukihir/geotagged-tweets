import numpy as np
import pandas as pd
from cycler import cycler
    
def center(df, keyword='target'):
    ratio = df['tweetcount_'+keyword] / df['tweetcount_all']
    if 'density_' + keyword in df.columns:
        density = df['density_'+keyword]
    else:
        density = df['tweetcount_'+keyword] / df['area']
    center_idx = density[ratio > 0.01].idxmax()
    center_latlon = df.loc[center_idx][['latitude', 'longitude']].to_list()
    return center_latlon

def distance_from_center(df, keyword='target'):
    '''
    Returns the distance of each grid cell from the center, defined as the cell where 
    the keyword occurs most frequently among the cells with more than 1% occurence ratio.
    '''
    import geopy.distance
    center_latlon = center(df, keyword=keyword)
    distance = pd.Series(zip(df['latitude'], df['longitude'])).apply(
        lambda x: geopy.distance.great_circle(center_latlon, x).km).to_numpy()
    return distance

def create_bins(array, numbins, log=False, integer=False):
    '''
    Creates suitable bins for array
    Parameters
    ----------
    array: data array which needs to be binned
    numbins (int): number of bins
    log (bool): returns logarithmic bins if True, linear bins if False
    integer (bool): data array contains only integer (discrete) values if True. Only relevant for log bins
    '''
    if numbins < 2:
        raise ValueError('Number of bins must be at least 2')
    if log:
        minval = np.min(array)
        if minval < 0:
            raise ValueError('Data array contains negative values. Log bins cannot be created')
        maxval = np.max(array)
        if integer:
            # first few bins are spaced linearly to include all integer values
            # the number of integer bins is determined so that all the log bin widths are larger than 2
            kmin = 0 if minval > 0 else 1    
            for k in range(kmin, numbins):
                if np.log(maxval/(minval+k)) / (numbins-k) > np.log((minval+k+2) / (minval+k)):
                    numintbins = k
                    break
                if k == numbins - 1:
                    raise ValueError('Number of bins is too large for this data')

            logbin_edges = np.geomspace(minval+numintbins, np.max(array)+1, num=numbins-numintbins+1)
            logbin_reps = np.sqrt(logbin_edges[:-1] * logbin_edges[1:])
            intbin_edges = np.arange(minval, minval+numintbins)
            bin_edges = np.concatenate((intbin_edges, logbin_edges))
            bin_reps = np.concatenate((intbin_edges, logbin_reps))
        else:
            nonzero_array = np.array(array)
            nonzero_array = nonzero_array[nonzero_array > 0]
            nonzero_min = np.min(nonzero_array)
            bin_edges = np.geomspace(nonzero_min, maxval, num=numbins+1)
            bin_reps = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        bin_edges = np.linspace(np.min(array), np.max(array), num=numbins+1)
        bin_reps = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_edges, bin_reps

def plot_single_histogram(ax, array, 
                          numbins=30, xscale='linear', yscale='linear', discrete=False, **kwargs):
    xlog = True if xscale=='log' else False
    bins, bin_reps = create_bins(array, numbins, xlog, discrete)
    hist, _ = np.histogram(array, bins, density=True)
    ax.plot(bin_reps, hist, **kwargs)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

def plot_many_histograms(ax, arrays, labels=[],
                         numbins=30, xscale='linear', yscale='linear', discrete=False, **kwargs):
    flattened = [val for array in arrays for val in array]
    xlog = True if xscale=='log' else False
    bins, bin_reps = create_bins(flattened, numbins, xlog, discrete)

    if not labels:
        for array in arrays:
            hist, _ = np.histogram(array, bins, density=True)
            ax.plot(bin_reps, hist, **kwargs)
    elif len(labels)==len(arrays):
        for array, label in zip(arrays, labels):
            hist, _ = np.histogram(array, bins, density=True)
            ax.plot(bin_reps, hist, label=label, **kwargs)
    else:
        raise ValueError('Length of labels does not match the number of arrays')
    
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

def set_style_cycler(ax, colors, markers):
    color_cycle = cycler(color=colors)
    marker_cycle = cycler(marker=markers)
    style_cycle = color_cycle * len(marker_cycle) + marker_cycle * len(color_cycle)
    ax.set_prop_cycle(style_cycle)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import cartopy.crs as ccrs
import cartopy.geodesic as cgeod
import shapely.geometry as sgeom

def dms_to_dd(dms):
    '''Convert the degrees, minutes, seconds notation to decimal degrees'''
    return dms[0] + dms[1] / 60 + dms[2] / 3600

class MapVisualizer:
    '''
    Class for visualizing data on a map.
    '''
    def __init__(self, fig, extent, ax=None, projection=None, resolution='50m', **kwargs):
        '''
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure object to draw on
        extent : list
            List or tuple of the form [lon_min, lon_max, lat_min, lat_max]
        projection : cartopy.crs.Projection
            Projection to use
        ax : matplotlib.axes.Axes, optional
            Axes object to draw on, by default None
        position : tuple, optional
            Position of the axes in the figure, by default (1, 1, 1)
        resolution : str, optional
            Resolution of the coastlines, by default '50m'
        '''
        self.fig = fig
        if ax is None:
            position=(1, 1, 1)
            self.ax = self.fig.add_subplot(*position, projection=projection)
        else:
            self.ax = ax
        self.ax.set_extent(extent)
        self.ax.coastlines(resolution=resolution, **kwargs)
        
    def visualize(self, data_lon, data_lat, data_val, cmap=None, norm=None, marker=',', size=0.1, **kwargs):
        '''
        Visualize data on the map.

        Parameters
        ----------
        data_lon : array-like
            Longitude values
        data_lat : array-like
            Latitude values
        data_val : array-like
            Data values
        cmap : matplotlib.colors.Colormap, optional
            Colormap to use, by default None
        norm : matplotlib.colors.Normalize, optional
            Normalization to use, by default None
        marker : str, optional
            Marker to use, by default ','
        size : float, optional
            Marker size, by default 0.1
        '''
        if cmap is None:
            cmap = plt.get_cmap('GnBu', 8)
        if norm is None:
            norm = mplcolors.Normalize(vmin=np.min(data_val), vmax=np.max(data_val))
        self.mappable = self.ax.scatter(data_lon, data_lat, c=data_val, cmap=cmap, norm=norm, 
                                        edgecolors='none', transform=ccrs.PlateCarree(), 
                                        marker=marker, s=size, **kwargs)
        return self.mappable
    
    def add_colorbar(self, cblabel='', orientation='horizontal'):
        '''
        Add a colorbar to the map.

        Parameters
        ----------
        cblabel : str, optional
            Colorbar label, by default ''
        orientation : str, optional
            Colorbar orientation, by default 'horizontal'
        '''
        if orientation == 'vertical':
            cax = self.ax.inset_axes([0.85, 0.03, 0.04, 0.33])
            cb = self.fig.colorbar(self.mappable, cax=cax, orientation=orientation)
            cb.set_label(cblabel, loc='top', rotation=0, va='bottom')
        elif orientation == 'horizontal':
            cax = self.ax.inset_axes([0.51, 0.095, 0.33, 0.04])
            cb = self.fig.colorbar(self.mappable, cax=cax, orientation=orientation)
            cb.set_label(cblabel, labelpad=1)
        else:
            raise ValueError('orientation must be either "vertical" or "horizontal"')
        return cb
    
    def add_text(self, text, x=0.05, y=0.93, fontsize='large'):
        '''
        Add text to the map.

        Parameters
        ----------
        text : str
            Text to add
        x, y : float, optional
            Position of the text, by default (0.05, 0.93)
        fontsize : str, optional
            Font size, by default 'large'
        '''
        txt = self.ax.text(x, y, text, fontsize=fontsize, transform=self.ax.transAxes)
        return txt
    
    def add_circle(self, lon, lat, radius, **kwargs):
        '''
        Add a circle to the map.

        Parameters
        ----------
        lon : float
            Longitude of the center of the circle
        lat : float
            Latitude of the center of the circle
        radius : float
            Radius of the circle
        color : str, optional
            Color of the circle, by default 'k'
        linewidth : float, optional
            Line width of the circle, by default 1
        linestyle : str, optional
            Line style of the circle, by default '-'
        '''
        circle_points = cgeod.Geodesic().circle(lon=lon, lat=lat, radius=radius, n_samples=200, endpoint=False)
        circle = sgeom.Polygon(circle_points)
        geom = self.ax.add_geometries((circle,), crs=ccrs.PlateCarree(), **kwargs)
        return geom
    
    def add_geometry(self, geometry, **kwargs):
        '''
        Add a geometry to the map.

        Parameters
        ----------
        geometry : shapely.geometry.Geometry
            Geometry to add
        '''
        geom = self.ax.add_geometries((geometry,), crs=ccrs.PlateCarree(), **kwargs)
        return geom

'''
Scale bar code
Copy pasted from https://stackoverflow.com/a/50674451
'''


def _axes_to_lonlat(ax, coords):
    """(lon, lat) from axes coordinates."""
    display = ax.transAxes.transform(coords)
    data = ax.transData.inverted().transform(display)
    lonlat = ccrs.PlateCarree().transform_point(*data, ax.projection)

    return lonlat


def _upper_bound(start, direction, distance, dist_func):
    """A point farther than distance from start, in the given direction.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        direction  Nonzero (2, 1)-shaped array, a direction vector.
        distance:  Positive distance to go past.
        dist_func: A two-argument function which returns distance.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    if distance <= 0:
        raise ValueError(f"Minimum distance is not positive: {distance}")

    if np.linalg.norm(direction) == 0:
        raise ValueError("Direction vector must not be zero.")

    # Exponential search until the distance between start and end is
    # greater than the given limit.
    length = 0.1
    end = start + length * direction

    while dist_func(start, end) < distance:
        length *= 2
        end = start + length * direction

    return end


def _distance_along_line(start, end, distance, dist_func, tol):
    """Point at a distance from start on the segment from start to end.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        end:       Outer bound on point's location.
        distance:  Positive distance to travel.
        dist_func: Two-argument function which returns distance.
        tol:       Relative error in distance to allow.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    initial_distance = dist_func(start, end)
    if initial_distance < distance:
        raise ValueError(f"End is closer to start ({initial_distance}) than "
                         f"given distance ({distance}).")

    if tol <= 0:
        raise ValueError(f"Tolerance is not positive: {tol}")

    # Binary search for a point at the given distance.
    left = start
    right = end

    while not np.isclose(dist_func(start, right), distance, rtol=tol):
        midpoint = (left + right) / 2

        # If midpoint is too close, search in second half.
        if dist_func(start, midpoint) < distance:
            left = midpoint
        # Otherwise the midpoint is too far, so search in first half.
        else:
            right = midpoint

    return right


def _point_along_line(ax, start, distance, angle=0, tol=0.01):
    """Point at a given distance from start at a given angle.

    Args:
        ax:       CartoPy axes.
        start:    Starting point for the line in axes coordinates.
        distance: Positive physical distance to travel.
        angle:    Anti-clockwise angle for the bar, in radians. Default: 0
        tol:      Relative error in distance to allow. Default: 0.01

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    # Direction vector of the line in axes coordinates.
    direction = np.array([np.cos(angle), np.sin(angle)])

    geodesic = cgeod.Geodesic()

    # Physical distance between points.
    def dist_func(a_axes, b_axes):
        a_phys = _axes_to_lonlat(ax, a_axes)
        b_phys = _axes_to_lonlat(ax, b_axes)

        # Geodesic().inverse returns a NumPy MemoryView like [[distance,
        # start azimuth, end azimuth]].
        # return geodesic.inverse(a_phys, b_phys).base[0, 0]
        return geodesic.inverse(a_phys, b_phys)[0, 0]

    end = _upper_bound(start, direction, distance, dist_func)

    return _distance_along_line(start, end, distance, dist_func, tol)


def scale_bar(ax, location, length, metres_per_unit=1000, unit_name='km',
              tol=0.01, angle=0, color='black', linewidth=3, text_offset=0.005,
              ha='center', va='bottom', plot_kwargs=None, text_kwargs=None,
              **kwargs):
    """Add a scale bar to CartoPy axes.

    For angles between 0 and 90 the text and line may be plotted at
    slightly different angles for unknown reasons. To work around this,
    override the 'rotation' keyword argument with text_kwargs.

    Args:
        ax:              CartoPy axes.
        location:        Position of left-side of bar in axes coordinates.
        length:          Geodesic length of the scale bar.
        metres_per_unit: Number of metres in the given unit. Default: 1000
        unit_name:       Name of the given unit. Default: 'km'
        tol:             Allowed relative error in length of bar. Default: 0.01
        angle:           Anti-clockwise rotation of the bar.
        color:           Color of the bar and text. Default: 'black'
        linewidth:       Same argument as for plot.
        text_offset:     Perpendicular offset for text in axes coordinates.
                         Default: 0.005
        ha:              Horizontal alignment. Default: 'center'
        va:              Vertical alignment. Default: 'bottom'
        **plot_kwargs:   Keyword arguments for plot, overridden by **kwargs.
        **text_kwargs:   Keyword arguments for text, overridden by **kwargs.
        **kwargs:        Keyword arguments for both plot and text.
    """
    # Setup kwargs, update plot_kwargs and text_kwargs.
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}

    plot_kwargs = {'linewidth': linewidth, 'color': color, **plot_kwargs,
                   **kwargs}
    text_kwargs = {'ha': ha, 'va': va, 'rotation': angle, 'color': color,
                   **text_kwargs, **kwargs}

    # Convert all units and types.
    location = np.asarray(location)  # For vector addition.
    length_metres = length * metres_per_unit
    angle_rad = angle * np.pi / 180

    # End-point of bar.
    end = _point_along_line(ax, location, length_metres, angle=angle_rad,
                            tol=tol)

    # Coordinates are currently in axes coordinates, so use transAxes to
    # put into data coordinates. *zip(a, b) produces a list of x-coords,
    # then a list of y-coords.
    ax.plot(*zip(location, end), transform=ax.transAxes, **plot_kwargs)

    # Push text away from bar in the perpendicular direction.
    midpoint = (location + end) / 2
    offset = text_offset * np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    text_location = midpoint + offset

    # 'rotation' keyword argument is in text_kwargs.
    ax.text(*text_location, f"{length} {unit_name}", rotation_mode='anchor',
            transform=ax.transAxes, **text_kwargs)
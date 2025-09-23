import numpy as np

import matplotlib.patches as mpatches
import cartopy.crs as ccrs

def add_north(ax, labelsize=18, loc_x=0.95, loc_y=0.95, width=0.04, height=0.13, pad=0.14):
    """
    Add a north arrow to a map.

    Parameters:
    ax : matplotlib.axes.Axes
        The axes to which the north arrow will be added.
    labelsize : int, optional
        The font size of the 'N' label. Default is 18.
    loc_x : float, optional
        The x-location of the arrow's base as a fraction of the axes width. Default is 0.95.
    loc_y : float, optional
        The y-location of the arrow's base as a fraction of the axes height. Default is 0.95.
    width : float, optional
        The width of the arrow as a fraction of the axes width. Default is 0.04.
    height : float, optional
        The height of the arrow as a fraction of the axes height. Default is 0.13.
    pad : float, optional
        The padding between the arrow and the 'N' label as a fraction of the axes height. Default is 0.14.

    Returns:
    None
    """
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ylen = maxy - miny
    xlen = maxx - minx

    # Calculate the positions for the north arrow
    left = [minx + xlen * (loc_x - width * 0.5), miny + ylen * (loc_y - pad)]
    right = [minx + xlen * (loc_x + width * 0.5), miny + ylen * (loc_y - pad)]
    top = [minx + xlen * loc_x, miny + ylen * (loc_y - pad + height)]
    center = [minx + xlen * loc_x, left[1] + (top[1] - left[1]) * 0.4]

    # Create the north arrow as a polygon
    triangle = mpatches.Polygon([left, top, right, center], color='k')
    ax.add_patch(triangle)

    # Add the 'N' label
    ax.text(s='N',
            x=minx + xlen * loc_x,
            y=miny + ylen * (loc_y - pad + height),
            fontsize=labelsize,
            horizontalalignment='center',
            verticalalignment='bottom')
    


def add_scalebar(ax, y=34.5, x=73.5, length_km=500, lw=3, size=10, lat_range=(10, 60)):
    """
    Add a scale bar to the specified axes.

    Parameters:
    ax : matplotlib.axes.Axes
        The axes to which the scale bar will be added.
    y : float
        The latitude position of the scale bar.
    x : float
        The starting longitude position of the scale bar.
    length_km : float
        The total length of the scale bar in kilometers.
    lw : float
        The line width of the scale bar.
    size : int
        The font size of the scale bar labels.
    lat_range : tuple
        The latitude range (min_lat, max_lat) to calculate the average km per degree.

    Returns:
    None
    """

    # Calculate the average latitude
    avg_lat = np.mean(lat_range)
    
    # Calculate km per degree based on the average latitude
    km_per_degree = 111.32 * np.cos(np.radians(avg_lat))
    
    # Convert length from km to degrees
    length_degree = length_km / km_per_degree
    
    # Main scale bar line
    ax.hlines(y=y, xmin=x, xmax=x+length_degree, 
              colors='black', lw=lw, transform=ccrs.PlateCarree())
    
    # Ticks (at both ends and the middle)
    for pos in [0, 0.5, 1]:
        ax.vlines(x=x + pos*length_degree, ymin=y-0.15, ymax=y+0.15, 
                  colors='black', lw=lw-1, transform=ccrs.PlateCarree())
    
    # Text annotations
    ax.text(x, y + 0.2, '0', ha='center', va='bottom', 
            fontsize=size, fontname='Times New Roman', transform=ccrs.PlateCarree())
    ax.text(x + length_degree/2, y + 0.2, f'{int(length_km/2)}',
            ha='center', va='bottom', fontsize=size, fontname='Times New Roman', transform=ccrs.PlateCarree())
    ax.text(x + length_degree, y + 0.2, f'{length_km} km', ha='center', 
            va='bottom', fontsize=size, fontname='Times New Roman', transform=ccrs.PlateCarree())

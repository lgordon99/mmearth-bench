import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.lines import Line2D
from shapely.geometry import Point
from shapely.ops import unary_union

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']

def make_sparse_data():
    fig = plt.figure(figsize=(9, 2))

    # New colors
    land_color1 = '#417e46'  # Dark green
    land_color2 = '#58ba47'  # Light green
    ocean_color = '#1f97d4'  # Blue

    # Create map with Equal Earth projection
    ax = fig.add_subplot(111, projection=ccrs.EqualEarth())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.set_adjustable('datalim')
    ax.spines['geo'].set_visible(False)

    # Load land polygons
    land_shp = shpreader.natural_earth(resolution='110m',
                                        category='physical',
                                        name='land')
    land_geoms = list(shpreader.Reader(land_shp).geometries())

    def point_on_land(lon, lat, land_geoms):
        """Check if a point is on land"""
        point = Point(lon, lat)
        for geom in land_geoms:
            if geom.contains(point):
                return True
        return False

    def generate_land_points(n_points, land_geoms, max_attempts=10000):
        """Generate random points that are on land"""
        points_lon = []
        points_lat = []
        attempts = 0

        while len(points_lon) < n_points and attempts < max_attempts:
            lon = np.random.uniform(-180, 180)
            lat = np.random.uniform(-60, 75)

            if point_on_land(lon, lat, land_geoms):
                points_lon.append(lon)
                points_lat.append(lat)

            attempts += 1

        return np.array(points_lon), np.array(points_lat)

    def generate_ocean_points(n_points, land_geoms, max_attempts=10000):
        """Generate random points that are in ocean"""
        points_lon = []
        points_lat = []
        attempts = 0

        while len(points_lon) < n_points and attempts < max_attempts:
            lon = np.random.uniform(-180, 180)
            lat = np.random.uniform(-60, 75)

            if not point_on_land(lon, lat, land_geoms):
                points_lon.append(lon)
                points_lat.append(lat)

            attempts += 1

        return np.array(points_lon), np.array(points_lat)

    # Generate sparse sampling points in 4:3:2 ratio
    np.random.seed(42)
    n_dark_green = 40  # 4 parts
    n_light_green = 30  # 3 parts
    n_blue = 20  # 2 parts

    # Land points (dark green) - 4 parts
    sparse_lon_land1, sparse_lat_land1 = generate_land_points(n_dark_green, land_geoms)
    ax.scatter(sparse_lon_land1, sparse_lat_land1, s=100, alpha=0.8, color=land_color1,
            marker='o', linewidths=1.5,
            transform=ccrs.PlateCarree(), zorder=3, label='Climate Action')

    # Land points (light green) - 3 parts
    sparse_lon_land2, sparse_lat_land2 = generate_land_points(n_light_green, land_geoms)
    ax.scatter(sparse_lon_land2, sparse_lat_land2, s=100, alpha=0.8, color=land_color2,
            marker='o', linewidths=1.5,
            transform=ccrs.PlateCarree(), zorder=3, label='Life on Land')

    # Ocean points (blue) - 2 parts
    sparse_lon_ocean, sparse_lat_ocean = generate_ocean_points(n_blue, land_geoms)
    ax.scatter(sparse_lon_ocean, sparse_lat_ocean, s=100, alpha=0.8, color=ocean_color,
            marker='o', linewidths=1.5,
            transform=ccrs.PlateCarree(), zorder=3, label='Life Below Water')

    # Create custom legend handles with full opacity and no horizontal lines
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=land_color1,
               markersize=10, markeredgecolor=land_color1, markeredgewidth=1.5,
               alpha=1.0, label='Climate Action', linestyle='none'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=land_color2,
               markersize=10, markeredgecolor=land_color2, markeredgewidth=1.5,
               alpha=1.0, label='Life on Land', linestyle='none'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ocean_color,
               markersize=10, markeredgecolor=ocean_color, markeredgewidth=1.5,
               alpha=1.0, label='Life Below Water', linestyle='none')
    ]

    # Add legend with column title - positioned outside the axes on the left
    legend = ax.legend(handles=legend_handles, title='SDGs', loc='center left', bbox_to_anchor=(-0.3, 0.5),
                       framealpha=0, frameon=False, fontsize=16, handletextpad=0.2, handlelength=1.0)
    legend.get_title().set_fontsize(16)

    # Adjust padding to minimize width - just enough space for legend on left, minimal on right
    plt.subplots_adjust(left=0.32, right=0.98, top=1, bottom=0.02)
    plt.savefig('teaser_subfigures/sparse_data.svg', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)

    # Save version with white text
    for text in legend.get_texts():
        text.set_color('white')
    legend.get_title().set_color('white')

    plt.savefig('teaser_subfigures/sparse_data_white.svg', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)

def make_globe():
    # Get viridis colors
    viridis = plt.get_cmap('viridis')
    africa_color = viridis(0.75)
    other_color = viridis(0.0)

    # Create figure with transparent background
    fig = plt.figure(figsize=(10, 10), facecolor='none')
    ax = plt.axes(projection=ccrs.Orthographic(central_longitude=-20, central_latitude=0))

    # Set ocean/water to light blue
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', zorder=0)

    # Add all land in viridis(0)
    ax.add_feature(cfeature.LAND, facecolor=other_color, edgecolor='none', zorder=1)

    # Merge all African countries into one geometry
    shpfilename = shpreader.natural_earth(resolution='110m',
                                        category='cultural',
                                        name='admin_0_countries')

    african_geometries = []
    for country in shpreader.Reader(shpfilename).records():
        if country.attributes['CONTINENT'] == 'Africa':
            african_geometries.append(country.geometry)

    # Merge all African geometries into one
    africa_merged = unary_union(african_geometries)

    # Add merged Africa in viridis(0.75) with no edges
    ax.add_geometries([africa_merged], ccrs.PlateCarree(),
                    facecolor=africa_color,
                    edgecolor='none',
                    zorder=2)

    # Add globe outline
    ax.spines['geo'].set_edgecolor('black')
    ax.spines['geo'].set_linewidth(2)

    plt.savefig('teaser_subfigures/globe.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)

def make_histogram():
    # Get viridis colors
    viridis = plt.get_cmap('viridis')
    source_color = mcolors.to_hex(viridis(0.0))
    target_color = mcolors.to_hex(viridis(0.75))

    # Generate distributions
    # Source domain: lower mean and std
    source_mean = 3
    source_std = 2
    source_data = np.random.normal(source_mean, source_std, 1000)

    # Target domain: higher mean and std
    target_mean = 1
    target_std = 1
    target_data = np.random.normal(target_mean, target_std, 1000)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 2))

    # Compute common bins based on the range of both datasets
    # Ensure 0 is exactly at a bin edge so no bar gets cut off
    all_data = np.concatenate([source_data, target_data])
    data_max = all_data.max()

    # Create bins starting from 0 and extending to cover all data
    # This ensures 0 is exactly at a bin edge
    n_bins = 30
    bin_width = data_max / n_bins
    bins = np.arange(0, data_max + bin_width, bin_width)

    # Plot histograms with same bins - use stepfilled to create non-overlapping filled outlines
    ax.hist(source_data, bins=bins, color=source_color, label='Source domain', edgecolor=source_color, histtype='stepfilled', linewidth=2)
    ax.hist(target_data, bins=bins, color=target_color, label='Target domain', edgecolor=target_color, histtype='stepfilled', linewidth=2)

    # Set x-axis to go from 0 to 20
    ax.set_xlim(0, 12)

    # Labels and legend
    ax.set_xlabel('Label', fontsize=18)
    ax.set_ylabel('Frequency', fontsize=18)
    ax.legend(fontsize=18, frameon=False)

    # Remove tick labels and tick marks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)

    # Remove box outline (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig('teaser_subfigures/distribution_shift.svg', dpi=300, bbox_inches='tight', transparent=True)

    # Save version with white text
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color('white')
    plt.savefig('teaser_subfigures/distribution_shift_white.svg', dpi=300, bbox_inches='tight', transparent=True)

def make_multimodal_visualization():
    # Use red and blue colors
    class1_color = 'red'
    class2_color = 'yellow'
    decision_boundary_color = 'orange'

    # Generate data with spherical decision boundary
    # Classes only separable in 3D (distance from origin)
    np.random.seed(42)
    n_points = 100

    # Scale factor for z-axis to make sphere squished vertically
    z_scale = 0.4  # Scale down z-coordinates to make it shorter vertically

    # Class 1: Inside sphere (close to origin)
    theta1 = np.random.uniform(0, 2*np.pi, n_points)
    phi1 = np.random.uniform(0, np.pi, n_points)
    r1 = np.random.uniform(0, 0.85, n_points)  # Radius < 0.7

    class1_x = r1 * np.sin(phi1) * np.cos(theta1)
    class1_y = r1 * np.sin(phi1) * np.sin(theta1)
    class1_z = r1 * np.cos(phi1) * z_scale  # Scale z-coordinates
    class1_points = np.column_stack([class1_x, class1_y, class1_z])

    # Class 2: Outside sphere (far from origin)
    theta2 = np.random.uniform(0, 2*np.pi, n_points)
    phi2 = np.random.uniform(0, np.pi, n_points)
    r2 = np.random.uniform(0.95, 1.56, n_points)  # Radius > 0.8

    class2_x = r2 * np.sin(phi2) * np.cos(theta2)
    class2_y = r2 * np.sin(phi2) * np.sin(theta2)
    class2_z = r2 * np.cos(phi2) * z_scale  # Scale z-coordinates
    class2_points = np.column_stack([class2_x, class2_y, class2_z])

    # Create 3D plot with wider, shorter figure
    fig = plt.figure(figsize=(11, 5))  # Wider and shorter
    ax = fig.add_subplot(111, projection='3d')

    # Plot the two classes
    scatter1 = ax.scatter(class1_points[:, 0], class1_points[:, 1], class1_points[:, 2],
            c=class1_color, s=50, alpha=0.6, label='Class 1', edgecolors=class1_color, linewidth=0.5)
    scatter2 = ax.scatter(class2_points[:, 0], class2_points[:, 1], class2_points[:, 2],
            c=class2_color, s=50, alpha=0.6, label='Class 2', edgecolors=class2_color, linewidth=0.5)

    # Add decision boundary sphere with green color (no label here)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = 0.9 * np.outer(np.cos(u), np.sin(v))
    y_sphere = 0.9 * np.outer(np.sin(u), np.sin(v))
    z_sphere = 0.9 * np.outer(np.ones(np.size(u)), np.cos(v)) * z_scale  # Scale z-coordinates
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color=decision_boundary_color)

    # Set axis limits - adjust zlim to match scaled z-coordinates
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-0.65, 0.65])  # Adjusted to match scaled z-coordinates

    # Set box aspect to make the plot appear wider (x, y, z relative sizes)
    ax.set_box_aspect([3.5, 3.5, 2])  # Wider x and y relative to z

    # Remove gray background - make panes transparent
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Remove tick marks but keep grid lines
    ax.tick_params(axis='x', which='both', length=0, pad=0, width=0)
    ax.tick_params(axis='y', which='both', length=0, pad=0, width=0)
    ax.tick_params(axis='z', which='both', length=0, pad=0, width=0)

    # Remove only the black axes lines at the edges (keep gray gridlines)
    ax.xaxis.line.set_linewidth(0)
    ax.yaxis.line.set_linewidth(0)
    ax.zaxis.line.set_linewidth(0)
    ax.xaxis.line.set_visible(False)
    ax.yaxis.line.set_visible(False)
    ax.zaxis.line.set_visible(False)

    # Remove any spines or additional black lines
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo['tick']['outward_factor'] = 0
        axis._axinfo['tick']['inward_factor'] = 0
        # Make sure tick lines are not visible
        for line in axis.get_ticklines():
            line.set_visible(False)

    # Create custom legend with solid green (no alpha)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=class1_color,
            markersize=10, alpha=1, markeredgecolor=class1_color, markeredgewidth=0.5, label='Class 1', linestyle='none'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=class2_color,
            markersize=10, alpha=1, markeredgecolor=class2_color, markeredgewidth=0.5, label='Class 2', linestyle='none'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=decision_boundary_color,
            markersize=12, alpha=0.6, markeredgecolor=decision_boundary_color, label='Decision\nboundary', linestyle='none')
    ]

    ax.legend(handles=legend_elements,
              fontsize=26,
              loc='center right',
              bbox_to_anchor=(1.6, 0.5),
              ncol=1,
              framealpha=0.9,
              markerscale=2,
              handletextpad=0.1,
              columnspacing=0.5,
              frameon=False)
    ax.view_init(elev=20, azim=45) # sets viewing angle
    # Remove all margins
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    # Set figure margins to zero
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    plt.savefig('teaser_subfigures/3d_modalities_spherical.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0, facecolor='none')

    # Save version with white text
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color('white')
    plt.savefig('teaser_subfigures/3d_modalities_spherical_white.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0, facecolor='none')

if __name__ == '__main__':
    make_sparse_data()
    make_globe()
    make_histogram()
    make_multimodal_visualization()

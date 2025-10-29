import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
from PIL import Image
from shapely.geometry import Point
from shapely.ops import unary_union

def make_limited_data():
    fig = plt.figure(figsize=(12, 6))

    # New colors
    land_color1 = '#417e46'  # Dark green
    land_color2 = '#58ba47'  # Light green
    ocean_color = '#1f97d4'  # Blue

    # Create map
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.set_extent([-180, 180, -60, 85], crs=ccrs.PlateCarree())
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
            transform=ccrs.PlateCarree(), zorder=3)

    # Land points (light green) - 3 parts
    sparse_lon_land2, sparse_lat_land2 = generate_land_points(n_light_green, land_geoms)
    ax.scatter(sparse_lon_land2, sparse_lat_land2, s=100, alpha=0.8, color=land_color2,
            marker='o', linewidths=1.5,
            transform=ccrs.PlateCarree(), zorder=3)

    # Ocean points (blue) - 2 parts
    sparse_lon_ocean, sparse_lat_ocean = generate_ocean_points(n_blue, land_geoms)
    ax.scatter(sparse_lon_ocean, sparse_lat_ocean, s=100, alpha=0.8, color=ocean_color,
            marker='o', linewidths=1.5,
            transform=ccrs.PlateCarree(), zorder=3)

    # Remove padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig('sparse_data.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig('sparse_data.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig('sparse_data.svg', dpi=300, bbox_inches='tight', pad_inches=0)

def make_globe():
    # Get viridis colors
    viridis = plt.get_cmap('viridis')
    africa_color = viridis(0.75)
    other_color = viridis(0.0)

    # Create figure with transparent background
    fig = plt.figure(figsize=(10, 10), facecolor='none')
    ax = plt.axes(projection=ccrs.Orthographic(central_longitude=-20, central_latitude=0))

    # Set ocean/water to white
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)

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

    plt.savefig('globe.svg', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)

def make_histogram():
    # Get viridis colors
    viridis = plt.get_cmap('viridis')
    source_color = viridis(0.0)
    target_color = viridis(0.75)

    # Generate distributions
    # Source domain: lower mean and std
    source_mean = 5
    source_std = 1.5
    source_data = np.random.normal(source_mean, source_std, 1000)

    # Target domain: higher mean and std
    target_mean = 10
    target_std = 2.5
    target_data = np.random.normal(target_mean, target_std, 1000)

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 3))

    # Compute common bins based on the range of both datasets
    all_data = np.concatenate([source_data, target_data])
    bins = np.linspace(all_data.min(), all_data.max(), 31)  # 31 edges = 30 bins

    # Plot histograms with same bins
    ax.hist(source_data, bins=bins, alpha=0.7, color=source_color, label='Source domain', edgecolor='black')
    ax.hist(target_data, bins=bins, alpha=0.7, color=target_color, label='Target domain', edgecolor='black')

    # Labels and legend
    ax.set_xlabel('Label', fontsize=18)
    ax.set_ylabel('Frequency', fontsize=18)
    ax.legend(fontsize=18)

    # Remove tick labels and tick marks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)

    plt.tight_layout()
    plt.savefig('distribution_shift.svg', dpi=300, bbox_inches='tight')

def make_multimodal_visualization():
    # Use red and blue colors
    class1_color = 'blue'
    class2_color = 'red'

    # Generate data with spherical decision boundary
    # Classes only separable in 3D (distance from origin)
    np.random.seed(42)
    n_points = 300

    # Class 1: Inside sphere (close to origin)
    theta1 = np.random.uniform(0, 2*np.pi, n_points)
    phi1 = np.random.uniform(0, np.pi, n_points)
    r1 = np.random.uniform(0, 0.7, n_points)  # Radius < 0.7

    class1_x = r1 * np.sin(phi1) * np.cos(theta1)
    class1_y = r1 * np.sin(phi1) * np.sin(theta1)
    class1_z = r1 * np.cos(phi1)
    class1_points = np.column_stack([class1_x, class1_y, class1_z])

    # Class 2: Outside sphere (far from origin)
    theta2 = np.random.uniform(0, 2*np.pi, n_points)
    phi2 = np.random.uniform(0, np.pi, n_points)
    r2 = np.random.uniform(0.8, 1.5, n_points)  # Radius > 0.8

    class2_x = r2 * np.sin(phi2) * np.cos(theta2)
    class2_y = r2 * np.sin(phi2) * np.sin(theta2)
    class2_z = r2 * np.cos(phi2)
    class2_points = np.column_stack([class2_x, class2_y, class2_z])

    # Create 3D plot
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the two classes
    scatter1 = ax.scatter(class1_points[:, 0], class1_points[:, 1], class1_points[:, 2],
            c=class1_color, s=50, alpha=0.6, label='Class 1', edgecolors='black', linewidth=0.5)
    scatter2 = ax.scatter(class2_points[:, 0], class2_points[:, 1], class2_points[:, 2],
            c=class2_color, s=50, alpha=0.6, label='Class 2', edgecolors='black', linewidth=0.5)

    # Add decision boundary sphere with green color (no label here)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = 0.75 * np.outer(np.cos(u), np.sin(v))
    y_sphere = 0.75 * np.outer(np.sin(u), np.sin(v))
    z_sphere = 0.75 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='green')

    # Set tight axis limits to compress empty space
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Remove tick marks but keep grid lines
    ax.tick_params(axis='x', which='both', length=0, pad=0)
    ax.tick_params(axis='y', which='both', length=0, pad=0)
    ax.tick_params(axis='z', which='both', length=0, pad=0)

    # Set axis labels with labelpad=2
    ax.set_xlabel('Modality X', fontsize=22, labelpad=2)
    ax.set_ylabel('Modality Y', fontsize=22, labelpad=2)
    ax.set_zlabel('Modality Z', fontsize=22, labelpad=2)

    # Create custom legend with solid green (no alpha)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=class1_color,
            markersize=10, alpha=1, markeredgecolor='black', markeredgewidth=0.5, label='Class 1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=class2_color,
            markersize=10, alpha=1, markeredgecolor='black', markeredgewidth=0.5, label='Class 2'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green',
            markersize=12, alpha=0.6, markeredgecolor='none', label='Decision boundary')
    ]

    ax.legend(handles=legend_elements, fontsize=22, loc='upper left',
            bbox_to_anchor=(-0.045, 0.89), framealpha=0.9, markerscale=2, handletextpad=0.2)

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    # Manual spacing
    plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.08)

    # Save
    temp_filename = '3d_modalities_spherical_full.png'
    plt.savefig(temp_filename, dpi=300, transparent=True)
    plt.savefig(temp_filename.replace('.png', '.svg'), dpi=300, transparent=True)

    # Crop the image to remove excess whitespace
    img = Image.open(temp_filename)
    width, height = img.size

    # Crop margins: 10% from left, top, right; 15% from bottom
    crop_left = int(width * 0.15)    # Remove 15% from left
    crop_top = int(height * 0.15)    # Remove 15% from top
    crop_right = int(width * 0.90)   # Remove 10% from right
    crop_bottom = int(height * 0.85) # Remove 15% from bottom

    img_cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))

    # Save cropped version
    final_filename = '3d_modalities_spherical.png'
    img_cropped.save(final_filename)

    # Delete the full version
    os.remove(temp_filename)

if __name__ == '__main__':
    make_limited_data()
    make_globe()
    make_histogram()
    make_multimodal_visualization()

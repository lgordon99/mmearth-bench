import ee
import geemap
from osgeo import gdal
import rasterio
from rasterio.features import shapes
import xml.etree.ElementTree as ET
import numpy as np

ee.Initialize()

ecoRegions = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017')

# Color updates
colorUpdates = [
    {'ECO_ID': 204, 'COLOR': '#B3493B'},
    {'ECO_ID': 245, 'COLOR': '#267400'},
    {'ECO_ID': 259, 'COLOR': '#004600'},
    {'ECO_ID': 286, 'COLOR': '#82F178'},
    {'ECO_ID': 316, 'COLOR': '#E600AA'},
    {'ECO_ID': 453, 'COLOR': '#5AA500'},
    {'ECO_ID': 317, 'COLOR': '#FDA87F'},
    {'ECO_ID': 763, 'COLOR': '#A93800'},
]

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

# Get all ecoregion features to build color mapping
print("Fetching ecoregion info in batches...")

# Get unique ECO_IDs and their default colors
# We'll fetch in smaller batches to avoid timeout
eco_colors = {}

# First, add all the default colors from the COLOR property
# We'll do this by getting the distinct ECO_IDs first
eco_id_list = ecoRegions.aggregate_array('ECO_ID').distinct().getInfo()
print(f"Found {len(eco_id_list)} ecoregions")

# Get colors in batches
batch_size = 100
for i in range(0, len(eco_id_list), batch_size):
    batch_ids = eco_id_list[i:i+batch_size]
    print(f"Fetching colors for ecoregions {i} to {min(i+batch_size, len(eco_id_list))}...")

    batch_features = ecoRegions.filter(ee.Filter.inList('ECO_ID', batch_ids)).select(['ECO_ID', 'COLOR']).limit(batch_size).getInfo()

    for feature in batch_features['features']:
        eco_id = feature['properties']['ECO_ID']
        color = feature['properties'].get('COLOR', '#808080')
        eco_colors[eco_id] = color

print(f"Loaded {len(eco_colors)} ecoregion colors")

# Apply color updates
for update in colorUpdates:
    eco_colors[update['ECO_ID']] = update['COLOR']

# Create RGB value lists
unique_ecoregions = sorted(eco_colors.keys())
red_values = [hex_to_rgb(eco_colors[eco_id])[0] for eco_id in unique_ecoregions]
green_values = [hex_to_rgb(eco_colors[eco_id])[1] for eco_id in unique_ecoregions]
blue_values = [hex_to_rgb(eco_colors[eco_id])[2] for eco_id in unique_ecoregions]

print("Creating RGB image with remap...")
empty = ee.Image().byte()
painted_ecoregions = empty.paint(ecoRegions, 'ECO_ID')

# Remap to RGB colors
band_r = painted_ecoregions.remap(unique_ecoregions, red_values).rename('vis-red')
band_g = painted_ecoregions.remap(unique_ecoregions, green_values).rename('vis-green')
band_b = painted_ecoregions.remap(unique_ecoregions, blue_values).rename('vis-blue')
image_rgb = ee.Image([band_r, band_g, band_b])

# Blue ocean background (100, 150, 200)
ocean_color = ee.Image([100, 150, 200]).rename(['vis-red', 'vis-green', 'vis-blue'])
final_export = image_rgb.unmask(ocean_color).toByte()

temp_file_initial = 'ecoregions_temp_4326_initial.tif'
temp_file = 'ecoregions_temp_4326.tif'

print("Exporting from Earth Engine...")
geemap.ee_export_image(
    final_export,
    filename=temp_file_initial,
    scale=10000,
    region=ee.Geometry.Rectangle([-180, -89, 180, 89], 'EPSG:4326', False),
    crs='EPSG:4326',
    file_per_band=False
)

# Remove nodata value from the exported file
print("Removing nodata value from source file...")
with rasterio.open(temp_file_initial) as src:
    profile = src.meta.copy()
    profile.update(nodata=None)
    data = src.read()

    with rasterio.open(temp_file, 'w', **profile) as dst:
        dst.write(data)

print(f"✓ Created {temp_file} without nodata value")

output_file = 'ecoregions_equal_earth_8857_color.tif'

print("Reprojecting to EPSG:8857...")
gdal.Warp(
    output_file,
    temp_file,
    dstSRS='EPSG:8857',
    xRes=10000,
    yRes=10000,
    resampleAlg='near',
    srcNodata=None,
    dstNodata=None,
    creationOptions=['COMPRESS=LZW']
)

input_file = 'ecoregions_equal_earth_8857_color.tif'
svg_file = 'ecoregions_equal_earth_color.svg'

print("Vectorizing...")

with rasterio.open(input_file) as src:
    image_r = src.read(1)
    image_g = src.read(2)
    image_b = src.read(3)
    bounds = src.bounds

    width = bounds.right - bounds.left
    height = bounds.top - bounds.bottom

    # Create SVG
    svg = ET.Element('svg', {
        'width': '1200',
        'height': '600',
        'viewBox': f'0 0 {int(width)} {int(height)}',
        'xmlns': 'http://www.w3.org/2000/svg'
    })

    print("Creating mask for non-black pixels...")
    non_black_mask = ~((image_r == 0) & (image_g == 0) & (image_b == 0))
    valid_mask = non_black_mask
    print(f"Valid pixels: {valid_mask.sum()} out of {valid_mask.size} total pixels (including ocean)")

    # Create a label image where each unique RGB gets a unique ID
    print("Creating unique IDs for each color...")
    unique_colors = {}
    color_id_image = np.zeros_like(image_r, dtype=np.int32)

    next_id = 1
    for row in range(image_r.shape[0]):
        for col in range(image_r.shape[1]):
            if not valid_mask[row, col]:
                continue

            r = int(image_r[row, col])
            g = int(image_g[row, col])
            b = int(image_b[row, col])
            color_tuple = (r, g, b)

            if color_tuple not in unique_colors:
                unique_colors[color_tuple] = next_id
                next_id += 1

            color_id_image[row, col] = unique_colors[color_tuple]

    print(f"Found {len(unique_colors)} unique colors")

    # Reverse mapping: ID -> RGB
    id_to_color = {v: k for k, v in unique_colors.items()}

    print("Extracting shapes by color ID...")

    # Vectorize the color ID image
    for geom, color_id in shapes(color_id_image.astype(np.int16), transform=src.transform, connectivity=8):
        if color_id == 0:  # Skip black border areas
            continue

        # Get the RGB color for this ID
        if color_id not in id_to_color:
            continue

        r, g, b = id_to_color[color_id]
        color = f'rgb({r},{g},{b})'

        coords = geom['coordinates']
        geom_type = geom['type']

        # Collect all path parts for this feature
        all_path_parts = []

        if geom_type == 'Polygon':
            rings = coords
        elif geom_type == 'MultiPolygon':
            rings = [r for poly in coords for r in poly]
        else:
            continue

        for ring in rings:
            path_parts = []
            for j, (x, y) in enumerate(ring):
                x_svg = x - bounds.left
                y_svg = bounds.top - y
                path_parts.append(f'{"M" if j == 0 else "L"} {x_svg:.2f},{y_svg:.2f}')
            all_path_parts.append(' '.join(path_parts) + ' Z')

        if not all_path_parts:
            continue

        path_data = ' '.join(all_path_parts)

        ET.SubElement(svg, 'path', {
            'd': path_data,
            'fill': color,
            'stroke': 'none',
            'fill-rule': 'evenodd'
        })

    tree = ET.ElementTree(svg)
    ET.indent(tree, space='  ')
    tree.write(svg_file, encoding='unicode', xml_declaration=True)

print(f"✓ SVG created: {svg_file}")

import ee
import geemap
from osgeo import gdal
import rasterio
from rasterio.features import shapes
import xml.etree.ElementTree as ET
import numpy as np

ee.Initialize()

ecoRegions = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017')

# Define distinct colors for each biome - avoiding blues for ocean distinction
biome_colors = {
    1:  '#004600',  # Tropical & Subtropical Moist Broadleaf Forests - dark green
    2:  '#8B7355',  # Tropical & Subtropical Dry Broadleaf Forests - brown
    3:  '#556B2F',  # Tropical & Subtropical Coniferous Forests - olive
    4:  '#006400',  # Temperate Broadleaf & Mixed Forests - dark green
    5:  '#2E8B57',  # Temperate Conifer Forests - sea green
    6:  '#228B22',  # Boreal Forests/Taiga - forest green (changed from blue)
    7:  '#DAA520',  # Tropical & Subtropical Grasslands, Savannas & Shrublands - goldenrod
    8:  '#FFD700',  # Temperate Grasslands, Savannas & Shrublands - gold
    9:  '#9ACD32',  # Flooded Grasslands & Savannas - yellow green (changed from blue)
    10: '#CD853F',  # Montane Grasslands & Shrublands - peru
    11: '#D3D3D3',  # Tundra - light gray (changed from light blue)
    12: '#D2691E',  # Mediterranean Forests, Woodlands & Scrub - chocolate
    13: '#F4A460',  # Deserts & Xeric Shrublands - sandy brown
    14: '#FF00FF'   # Mangroves - magenta
}

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

unique_biomes = sorted(biome_colors.keys())
red_values = [hex_to_rgb(biome_colors[b])[0] for b in unique_biomes]
green_values = [hex_to_rgb(biome_colors[b])[1] for b in unique_biomes]
blue_values = [hex_to_rgb(biome_colors[b])[2] for b in unique_biomes]

empty = ee.Image().byte()
painted_biomes = empty.paint(ecoRegions, 'BIOME_NUM')

# Create RGB image
band_r = painted_biomes.remap(unique_biomes, red_values).rename('vis-red')
band_g = painted_biomes.remap(unique_biomes, green_values).rename('vis-green')
band_b = painted_biomes.remap(unique_biomes, blue_values).rename('vis-blue')
image_rgb = ee.Image([band_r, band_g, band_b])

# Blue ocean background (100, 150, 200)
ocean_color = ee.Image([100, 150, 200]).rename(['vis-red', 'vis-green', 'vis-blue'])
final_export = image_rgb.unmask(ocean_color).toByte()

temp_file_initial = 'biomes_temp_4326_initial.tif'
temp_file = 'biomes_temp_4326.tif'

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

output_file = 'biomes_equal_earth_8857_color.tif'

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

# print(f"✓ Done! Color biomes with blue oceans")

input_file = 'biomes_equal_earth_8857_color.tif'
svg_file = 'biomes_equal_earth_color.svg'

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
    # Ocean is now (100, 150, 200)
    non_black_mask = ~((image_r == 0) & (image_g == 0) & (image_b == 0))
    valid_mask = non_black_mask
    print(f"Valid pixels: {valid_mask.sum()} out of {valid_mask.size} total pixels (including ocean)")
    print(f"Black pixels: {non_black_mask.size - non_black_mask.sum()}")

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
    # Use connectivity=8 to ensure all regions are captured (including diagonal connections)
    for geom, color_id in shapes(color_id_image.astype(np.int16), transform=src.transform, connectivity=8):
        if color_id == 0:  # Skip only black border areas (ocean and biomes have non-zero IDs)
            continue

        # Get the RGB color for this ID
        if color_id not in id_to_color:
            continue

        r, g, b = id_to_color[color_id]
        color = f'rgb({r},{g},{b})'

        coords = geom['coordinates']
        geom_type = geom['type']

        # Collect all path parts for this feature (including holes) into one path definition
        all_path_parts = []

        if geom_type == 'Polygon':
            rings = coords
        elif geom_type == 'MultiPolygon':
            # Flatten list of polygons into list of rings for the path
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
            'fill-rule': 'evenodd'  # Important for handling holes correctly
        })

    tree = ET.ElementTree(svg)
    ET.indent(tree, space='  ')
    tree.write(svg_file, encoding='unicode', xml_declaration=True)

print(f"✓ SVG created: {svg_file}")

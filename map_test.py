from rasterio.warp import transform_bounds
from shapely.geometry import box
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import rasterio

m = folium.Map(location=[0, 0], zoom_start=3, tiles=None)
folium.TileLayer('cartodbpositron', name='World map', control=True, show=True).add_to(m)
folium.TileLayer(tiles='', name='Clear', control=True, attr='No attribution needed', show=False).add_to(m)

tasks = {'species': {'title': 'Species', 'color': 'red'},
         'soil_nitrogen': {'title': 'Soil nitrogen', 'color': 'blue'},
         'soil_organic_carbon': {'title': 'Soil organic carbon', 'color': 'brown'},
         'soil_pH': {'title': 'Soil pH', 'color': 'purple'}}

layers = {}

# Create a feature group for each visualization type
solid_group = folium.FeatureGroup(name="Solid colors")
imagery_group = folium.FeatureGroup(name="Sentinel-2")

for task, properties in tasks.items():
    title = properties['title']
    color = properties['color']
    gdf = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')
    datadir = f'{task}/data'
    task_layer = folium.FeatureGroup(name=title)

    # Extract bounding boxes and process TIFFs as before
    for tiff_name in os.listdir(datadir):
        try:
            with rasterio.open(f'{datadir}/{tiff_name}') as tiff:
                bounds = tiff.bounds
                crs = tiff.crs
                rgb = tiff.read([4,3,2]).astype(float)
                task_value = tiff.tags()[task]

            if crs != 'EPSG:4326':
                bounds = transform_bounds(crs, 'EPSG:4326', *bounds)

            # Process Sentinel-2 imagery
            for i in range(3):
                rgb[i] = (rgb[i] - rgb[i].min()) / (rgb[i].max() - rgb[i].min())

            rgb = np.stack(rgb, axis=-1)
            img_path = f'temp/{task}_{tiff_name}.png'
            os.makedirs('temp', exist_ok=True)
            plt.imsave(img_path, rgb)

            bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
            gdf = pd.concat([gdf, gpd.GeoDataFrame([{
                'geometry': bbox,
                task: task_value,
                'img_path': img_path,
                'bounds': [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
            }], crs=gdf.crs)], ignore_index=True)
        except:
            continue

    # Add to both visualization groups
    for _, row in gdf.iterrows():
        # Add to solid colors group
        folium.GeoJson(
            data=row['geometry'].__geo_interface__,
            style_function=lambda x, color=color: {
                'fillColor': color,
                'color': color,
                'weight': 1,
                'fillOpacity': 0.5
            },
            tooltip=folium.Tooltip(f'{title}: {row[task]}')
        ).add_to(solid_group)

        # Add to imagery group
        folium.raster_layers.ImageOverlay(
            image=row['img_path'],
            bounds=row['bounds'],
            opacity=0.7,
            interactive=True,
            cross_origin=True
        ).add_to(imagery_group)

    task_layer.add_to(m)
    layers[title] = color

# Add visualization groups to map
solid_group.add_to(m)
imagery_group.add_to(m)

javascript = """
<script>
document.addEventListener("DOMContentLoaded", function() {
    var layerColors = %s;

    // Function to create section headers
    function createHeader(text) {
        var header = document.createElement('div');
        header.innerHTML = '<strong>' + text + '</strong>';
        header.style.marginBottom = '2px';
        header.style.marginTop = '5px';
        return header;
    }

    var control = document.querySelector('.leaflet-control-layers-list');

    // Add Background header
    var baseMapsHeader = createHeader('Background');
    control.insertBefore(baseMapsHeader, control.children[0]);

    // Add Tasks header
    var tasksHeader = createHeader('Tasks');
    var overlaysIndex = Array.from(control.children).findIndex(el => el.classList.contains('leaflet-control-layers-overlays'));
    control.insertBefore(tasksHeader, control.children[overlaysIndex]);

    // Add Visualization header
    var vizHeader = createHeader('Visualization');
    control.appendChild(vizHeader);

    // Color the task labels
    var labels = document.querySelectorAll('.leaflet-control-layers label span');
    labels.forEach(function(label) {
        var text = label.innerText.trim();
        if (layerColors[text]) {
            label.style.color = layerColors[text];
        }
    });
});
</script>
""" % layers

m.get_root().html.add_child(folium.Element(javascript))
map_html = m.get_root().render()

# read the HTML template
with open('template.html', 'r') as f:
    template_html = f.read()

# replace the placeholder with the Folium map HTML
embedded_html = template_html.replace('{{ folium_map }}', map_html)

# save the combined HTML to a new file
with open('index.html', 'w') as f:
    f.write(embedded_html)

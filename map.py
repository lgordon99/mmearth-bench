'''
map.py
'''

# ============================================== IMPORTS ============================================== #

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
layers = {}
# tasks = {'species': {'title': 'Species', 'color': 'red'},
#          'soil_nitrogen': {'title': 'Soil nitrogen', 'color': 'blue'},
#          'soil_organic_carbon': {'title': 'Soil organic carbon', 'color': 'brown'},
#          'soil_pH': {'title': 'Soil pH', 'color': 'purple'}}
tasks = {'species': {'title': 'Species', 'color': 'red'}}
for task, properties in tasks.items():
    title = properties['title']
    color = properties['color']
    gdf = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')
    datadir = f'{task}/data'
    task_layer = folium.FeatureGroup(name=title)

    # extract bounding boxes from TIFFs
    for tiff_name in os.listdir(datadir):
        with rasterio.open(f'{datadir}/{tiff_name}') as tiff:
            bounds = tiff.bounds
            crs = tiff.crs
            rgb = tiff.read([4,3,2]).astype(float) # R, G, B
            task_value = tiff.tags()[task]

        if crs != 'EPSG:4326':
            bounds = transform_bounds(crs, 'EPSG:4326', *bounds)

        for i in range(3):  # normalize each band to the [0,1] range
            rgb[i] = (rgb[i] - rgb[i].min()) / (rgb[i].max() - rgb[i].min())

        rgb = np.stack(rgb, axis=-1) # (H, W, 3)
        os.makedirs('temp', exist_ok=True)
        img_path = f'temp/{task}_{tiff_name}.png'
        plt.imsave(img_path, rgb)

        folium.raster_layers.ImageOverlay(name=tiff_name,
                                            image=img_path,
                                            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                                            opacity=0.7,
                                            interactive=True,
                                            cross_origin=True).add_to(task_layer)

        bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
        gdf = pd.concat([gdf, gpd.GeoDataFrame([{'geometry': bbox, task: task_value}], crs=gdf.crs)], ignore_index=True)
        # except:
        #     print('except')
        #     continue

    # for _, row in gdf.iterrows():
    #     folium.GeoJson(data=row['geometry'].__geo_interface__,
    #                    style_function=lambda x, color=color: {'fillColor': color,
    #                                                           'color': color,
    #                                                           'weight': 1,
    #                                                           'fillOpacity': 0.5},
    #                    name=title,
    #                    tooltip=folium.Tooltip(f'{title}: {row[task]}')).add_to(task_layer)

    task_layer.add_to(m)
    layers[title] = color

folium.LayerControl(collapsed=False).add_to(m)

javascript = """
<script>
document.addEventListener("DOMContentLoaded", function() {
    // Define the layer colors
    var layerColors = %s;

    // Find all layer control labels
    var labels = document.querySelectorAll('.leaflet-control-layers label span');

    // Apply colors to the corresponding labels
    labels.forEach(function(label) {
        var text = label.innerText.trim(); // Get the label text

        if (layerColors[text]) {
            label.style.color = layerColors[text]; // Apply color
            // label.style.fontWeight = "bold"; // Optional: make it bold
        }
    });

    var control = document.querySelector('.leaflet-control-layers-list');

    // Create header for Base Maps
    var baseMapsHeader = document.createElement('div');
    baseMapsHeader.innerHTML = '<strong>Background</strong>';
    baseMapsHeader.style.marginBottom = '2px';
    baseMapsHeader.style.marginTop = '1px';
    control.insertBefore(baseMapsHeader, control.children[0]);

    // Create header for Overlays
    var overlaysHeader = document.createElement('div');
    overlaysHeader.innerHTML = '<strong>Tasks</strong>';
    overlaysHeader.style.marginTop = '5px';
    overlaysHeader.style.marginBottom = '2px';
    var overlaysIndex = Array.from(control.children).findIndex(el => el.classList.contains('leaflet-control-layers-overlays'));
    control.insertBefore(overlaysHeader, control.children[overlaysIndex]);
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

# web map tile service
# KU server
# leaderboard

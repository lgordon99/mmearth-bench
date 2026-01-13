"""Testing geospatial clustering and polygon creation algorithm."""

import numpy as np
from shapely.geometry import Point, Polygon
import json
import h5py
import utils

def make_clusters(points, target_size):
    """
    Recursively partition 2D space to create balanced groups.
    Returns both clusters and their bounding rectangles.
    """

    clusters = {}
    cluster_id = 0

    def partition_recursive(points_subset, bounds):
        """
        Recursively partition points into balanced groups.
        bounds: [min_x, max_x, min_y, max_y]
        """
        nonlocal cluster_id # allows cluster_id to be accessed and modified in the outer scope
        n = len(points_subset)

        # Base case: if points fit in one group (within tolerance)
        if n <= target_size * 1.2:  # allows 20% tolerance
            clusters[cluster_id] = points_subset # adds points to cluster
            cluster_id += 1 # increments cluster id
            return # stops recursion

        needed_groups = max(1, (n + target_size - 1) // target_size) # calculates how many groups we need

        # Decide split direction based on aspect ratio
        min_x, max_x, min_y, max_y = bounds
        width = max_x - min_x
        height = max_y - min_y

        # Split along longer dimension
        if width > height:
            axis = 1 # splits on x
        else:
            axis = 2  # splits on y

        # Sort points along chosen axis
        sorted_indices = np.argsort(points_subset[:, axis])
        sorted_points = points_subset[sorted_indices]

        # Calculate optimal split point
        left_groups = needed_groups // 2
        split_idx = min(left_groups * target_size, n - target_size)
        split_idx = max(target_size, split_idx)

        # Find the actual split value (between split_idx-1 and split_idx)
        split_value = (sorted_points[split_idx - 1, axis] + sorted_points[split_idx, axis]) / 2

        # Split the points
        left_points = sorted_points[:split_idx]
        right_points = sorted_points[split_idx:]

        # Create new bounds for each partition
        if axis == 1:  # Split on x
            left_bounds = [min_x, split_value, min_y, max_y]
            right_bounds = [split_value, max_x, min_y, max_y]
        else:  # Split on y
            left_bounds = [min_x, max_x, min_y, split_value]
            right_bounds = [min_x, max_x, split_value, max_y]

        # Recursively partition each half
        partition_recursive(left_points, left_bounds)
        partition_recursive(right_points, right_bounds)

    # get initial bounds of the points
    min_x, min_y = points[:, 1:].min(axis=0)
    max_x, max_y = points[:, 1:].max(axis=0)

    # Add small padding
    padding_x = (max_x - min_x) * 0.05
    padding_y = (max_y - min_y) * 0.05

    initial_bounds = [min_x - padding_x, max_x + padding_x,
                        min_y - padding_y, max_y + padding_y]

    partition_recursive(points, initial_bounds)

    return clusters

def make_cluster_polygons(clusters):
    """
    Create bounding box polygons directly from the points in each cluster.
    """
    polygons = {}

    for cluster_id, points in clusters.items():
        points_array = np.array(points)

        # Get actual bounding box of the points
        min_x = points_array[:, 1].min()
        max_x = points_array[:, 1].max()
        min_y = points_array[:, 2].min()
        max_y = points_array[:, 2].max()

        # Create rectangle polygon (no padding needed)
        polygons[cluster_id] = [[min_x, min_y],
                                [max_x, min_y],
                                [max_x, max_y],
                                [min_x, max_y],
                                [min_x, min_y]]

    return polygons

def save_cluster_polygons_as_geojson(clusters, polygons):
    """
    Convert clusters and polygons to GeoJSON format.
    """
    features = []

    for cluster_id in sorted(clusters.keys()):
        points = clusters[cluster_id]
        polygon_coordinates = polygons[cluster_id]
        points_list = [[int(idx), float(lon), float(lat)] for idx, lon, lat in points] # converts points to list of floats
        polygon_coordinates_clean = [[float(x), float(y)] for x, y in polygon_coordinates] # converts polygon coordinates to list of floats

        feature = {
            "type": "Feature",
            "properties": {
                "cluster_id": int(cluster_id),
                "point_count": len(points),
                "points": points_list
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon_coordinates_clean]
            }
        }
        features.append(feature)

    output_file = 'clusters.geojson'

    with open(output_file, 'w') as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)

    print(f"GeoJSON saved to: {output_file}")

def check(clusters, polygons):
    # Verify containment
    print("\nVerifying point containment:")
    total_inside = 0
    total_points = 0

    for cluster_id, points in clusters.items():
        if cluster_id not in polygons:
            print(f"Cluster {cluster_id}: No polygon created")
            continue

        poly_coordinates = polygons[cluster_id]
        poly = Polygon(poly_coordinates[:-1])

        inside_count = 0
        for point in points:
            pt = Point(point[1:])
            if poly.contains(pt) or poly.distance(pt) < 1e-9:
                inside_count += 1

        total_inside += inside_count
        total_points += len(points)
        print(f"Cluster {cluster_id}: {inside_count}/{len(points)} points inside polygon")

    print(f"\nTotal: {total_inside}/{total_points} points contained ({100*total_inside/total_points:.1f}%)")

    # Verify non-overlapping
    print("\nVerifying non-overlapping:")
    cluster_ids = sorted(polygons.keys())
    overlaps_found = False

    for i, id1 in enumerate(cluster_ids):
        poly1 = Polygon(polygons[id1][:-1])

        for id2 in cluster_ids[i+1:]:
            poly2 = Polygon(polygons[id2][:-1])

            if poly1.intersects(poly2):
                intersection = poly1.intersection(poly2)
                # Allow tiny numerical overlaps (shared edges)
                if intersection.geom_type == 'Polygon' and intersection.area > 1e-9:
                    print(f"WARNING: Clusters {id1} and {id2} overlap (area: {intersection.area:.6f})")
                    overlaps_found = True

    if not overlaps_found:
        print("✓ No overlaps detected (edges may touch)!")

def main():
    data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']

    with h5py.File(f'{data_dir_path}/soil_nitrogen/soil_nitrogen.h5', 'r') as h5_file:
        geolocation = h5_file['geolocation'][:]
        train_indices = utils.read_json(f'{data_dir_path}/soil_nitrogen/soil_nitrogen_split_data.json')['train_100%_indices']
        coordinates = geolocation[train_indices]

    coordinates = np.column_stack((train_indices, coordinates))

    print(f"Total points: {len(coordinates)}")
    clusters = make_clusters(coordinates, target_size=16) # creates clusters of target size 16
    print(f"Created {len(clusters)} clusters")
    cluster_sizes = sorted([len(points) for points in clusters.values()])
    print(f"Cluster sizes: {cluster_sizes}")
    print(f"Min: {min(cluster_sizes)}, Max: {max(cluster_sizes)}, Avg: {np.mean(cluster_sizes):.2f}")

    polygons = make_cluster_polygons(clusters) # converts to bounding box polygons based on actual points
    save_cluster_polygons_as_geojson(clusters, polygons) # converts to GeoJSON
    check(clusters, polygons) # verifies containment and non-overlapping

if __name__ == "__main__":
    main()

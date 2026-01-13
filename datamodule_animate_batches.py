'''
datamodule.py
'''

# ============================================== IMPORTS ============================================== #

import os
os.environ['DATA_DIR_PATH'] = '/n/gajos_lab/Lab/luciagordon/mmearth-bench'

from dataset import MMEarthBenchDataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import BatchSampler, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

# ============================================== CLASSES ============================================== #

class GeographicBatchSampler(BatchSampler):
    def __init__(self, dataset, split, batch_size):
        self.indices = np.array(dataset.split_data[f'{split}_indices'])
        coordinates = dataset.geolocation[self.indices]
        points = np.column_stack((self.indices, coordinates))
        self.batch_size = batch_size
        self.batches = {}
        batch_id = 0
        split_events = []

        def partition_recursive(points_subset, bounds):
            """
            Recursively partition points into balanced groups.
            bounds: [min_x, max_x, min_y, max_y]
            """
            nonlocal batch_id # allows batch_id to be accessed and modified in the outer scope
            n = len(points_subset) # number of points remaining to be batched

            # Base case: if points fit in one group (within tolerance)
            if n < 2 * batch_size:  # allows 20% tolerance
                self.batches[batch_id] = points_subset # adds points to cluster
                split_events.append({
                'type': 'batch',
                'bounds': bounds,
                'id': batch_id
            })
                print(f'Created batch {batch_id} with {n} tiles')
                batch_id += 1 # increments cluster id
                return # stops recursion

            num_batches = max(1, (n + batch_size - 1) // batch_size) # calculates how many more batches we need

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
            num_batches_on_left = num_batches // 2 # calculates how many batches to put on the left
            split_idx = min(num_batches_on_left * batch_size, n - batch_size)
            split_idx = max(batch_size, split_idx)

            # Find the actual split value (between split_idx-1 and split_idx)
            split_value = (sorted_points[split_idx - 1, axis] + sorted_points[split_idx, axis]) / 2

            # Split the points
            left_points = sorted_points[:split_idx]
            right_points = sorted_points[split_idx:]

            # Create new bounds for each partition
            if axis == 1:  # Split on x
                left_bounds = [min_x, split_value, min_y, max_y]
                right_bounds = [split_value, max_x, min_y, max_y]
                line = ((split_value, bounds[2]), (split_value, bounds[3]))
            else:  # Split on y
                left_bounds = [min_x, max_x, min_y, split_value]
                right_bounds = [min_x, max_x, split_value, max_y]
                line = ((bounds[0], split_value), (bounds[1], split_value))

            split_events.append({
            'type': 'split',
            'line': line,
            'bounds': bounds
        })

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

        # Calculate tight bounding boxes for each batch
        batch_bounds = {}  # batch_id -> (initial_bounds, tight_bounds)
        for batch_id, batch_points in self.batches.items():
            points_array = np.array(batch_points)
            # Find the initial bounds from split_events
            initial_b = None
            for event in split_events:
                if event['type'] == 'batch' and event['id'] == batch_id:
                    initial_b = event['bounds']
                    break

            # Calculate tight bounding box
            min_x = points_array[:, 1].min()
            max_x = points_array[:, 1].max()
            min_y = points_array[:, 2].min()
            max_y = points_array[:, 2].max()
            tight_bounds = [min_x, max_x, min_y, max_y]

            if initial_b:
                batch_bounds[batch_id] = (initial_b, tight_bounds)

        print("Generating animation...")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        fig.suptitle('TTT-MMR-Geo Batches for Biomass Geographic Test', fontsize=16, fontweight='bold', y=0.92)
        ax.scatter(points[:, 1], points[:, 2], s=1, c='black', alpha=0.5)
        ax.set_xlim(initial_bounds[0], initial_bounds[1])
        ax.set_ylim(initial_bounds[2], initial_bounds[3])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Add batch number text (initially hidden)
        batch_text = ax.text(0.85, 0.95, 'Batch: ', transform=ax.transAxes,
                            fontsize=14, verticalalignment='top', horizontalalignment='left')

        lines = []
        patches = []
        patch_batch_ids = []  # Track which batch_id each patch belongs to
        processed_events = []  # Track which events have been processed

        num_morph_frames = 120  # Number of frames for morphing animation (smoother)
        num_hold_frames = 90   # Hold final result for ~3 seconds at 30fps
        pause_frames_per_batch = 30  # Pause for 1 second at 30fps for each of first 10 batches
        pause_frames_per_early_split = 30  # Pause for 1 second for splits before batch 10
        initial_pause_frames = 30 # Pause at start before first split
        pause_frames_after_last_batch = 30  # Pause after last batch before morph

        def update(frame):
            if frame < initial_pause_frames:
                return lines + patches + [batch_text]

            frame -= initial_pause_frames

            # Map frame to split_event index accounting for pauses
            current_frame = 0
            batch_count = 0
            split_event_index = 0

            # Find which split_event we're at
            for i, event in enumerate(split_events):
                duration = 1
                if event['type'] == 'batch':
                    if batch_count < 10:
                        duration += pause_frames_per_batch
                    batch_count += 1
                else:
                    if batch_count < 10:
                        duration += pause_frames_per_early_split

                # Add pause after the very last event
                if i == len(split_events) - 1:
                    duration += pause_frames_after_last_batch

                if frame >= current_frame and frame < current_frame + duration:
                    split_event_index = i
                    break
                current_frame += duration
            else:
                # Past all split events
                split_event_index = len(split_events)

            # Zoom logic
            # if batch_count < 10 and split_event_index < len(split_events):
            #     bounds = split_events[split_event_index]['bounds']
            #     w = bounds[1] - bounds[0]
            #     h = bounds[3] - bounds[2]
            #     pad_x = w * 0.5
            #     pad_y = h * 0.5
            #     # Ensure we don't zoom out further than initial bounds
            #     ax.set_xlim(max(initial_bounds[0], bounds[0] - pad_x), min(initial_bounds[1], bounds[1] + pad_x))
            #     ax.set_ylim(max(initial_bounds[2], bounds[2] - pad_y), min(initial_bounds[3], bounds[3] + pad_y))
            # else:
            #     ax.set_xlim(initial_bounds[0], initial_bounds[1])
            #     ax.set_ylim(initial_bounds[2], initial_bounds[3])

            # Just keep default zoom level
            ax.set_xlim(initial_bounds[0], initial_bounds[1])
            ax.set_ylim(initial_bounds[2], initial_bounds[3])

            # Adjust frame count for phase calculations
            num_split_frames = current_frame

            # Phase 1: Show splits only (no rectangles yet)
            if split_event_index < len(split_events):
                batch_text.set_visible(True)
                event = split_events[split_event_index]

                if event['type'] == 'split':
                    # Only draw the line once when we first see this event
                    if split_event_index not in processed_events:
                        (x1, y1), (x2, y2) = event['line']
                        # New lines start red
                        line, = ax.plot([x1, x2], [y1, y2], color='red', lw=3.0, solid_capstyle='butt')
                        lines.append(line)
                        processed_events.append(split_event_index)

                # Update line colors: all previous lines gray, latest line red
                if len(lines) > 0:
                    lines[-1].set_color('red')
                    lines[-1].set_linewidth(3.0)
                    for line in lines[:-1]:
                        line.set_color('gray')
                        line.set_linewidth(1.0)

                if event['type'] == 'batch':
                    # Store batch info and create red rectangle immediately
                    if split_event_index not in processed_events:
                        patch_batch_ids.append(event['id'])
                        processed_events.append(split_event_index)

                        if event['id'] in batch_bounds:
                            initial_b, _ = batch_bounds[event['id']]
                            rect = Rectangle((initial_b[0], initial_b[2]),
                                              initial_b[1]-initial_b[0],
                                              initial_b[3]-initial_b[2],
                                              fill=False, edgecolor='blue', lw=1.0, zorder=10)
                            ax.add_patch(rect)
                            patches.append(rect)

                    # Update batch text and make it visible
                    batch_text.set_text(f'Batch: {event["id"] + 1}')
                else:
                    if len(patch_batch_ids) < 10:
                        batch_text.set_text('Batch: ')

            # Phase 2: Morph lines into rectangles, then morph to tight bounding boxes
            elif frame < num_split_frames + num_morph_frames:
                batch_text.set_visible(False)
                # Hide lines during morph phase
                for line in lines:
                    line.set_alpha(0.0)

                morph_progress = (frame - num_split_frames) / num_morph_frames
                # Use easing function for smooth transition
                t = morph_progress * morph_progress * (3 - 2 * morph_progress)  # smoothstep

                # Morph rectangles from initial to tight bounds
                for i, batch_id in enumerate(patch_batch_ids):
                    if batch_id in batch_bounds:
                        initial_b, tight_b = batch_bounds[batch_id]
                        # Interpolate between initial and tight bounds
                        x = initial_b[0] + (tight_b[0] - initial_b[0]) * t
                        y = initial_b[2] + (tight_b[2] - initial_b[2]) * t
                        width = (initial_b[1] - initial_b[0]) + ((tight_b[1] - tight_b[0]) - (initial_b[1] - initial_b[0])) * t
                        height = (initial_b[3] - initial_b[2]) + ((tight_b[3] - tight_b[2]) - (initial_b[3] - initial_b[2])) * t

                        # Update rectangle
                        patches[i].set_xy((x, y))
                        patches[i].set_width(width)
                        patches[i].set_height(height)

            # Phase 3: Hold final result (no changes, just display)
            # Make sure lines are fully transparent
            else:
                batch_text.set_visible(False)
                if len(patches) == 0:
                    # Create patches if somehow not created yet
                    for batch_id in patch_batch_ids:
                        if batch_id in batch_bounds:
                            _, tight_b = batch_bounds[batch_id]
                            rect = Rectangle((tight_b[0], tight_b[2]),
                                           tight_b[1]-tight_b[0],
                                           tight_b[3]-tight_b[2],
                                           fill=False, edgecolor='blue', lw=1.0, zorder=10)
                            ax.add_patch(rect)
                            patches.append(rect)
                for line in lines:
                    line.set_alpha(0.0)

            return lines + patches + [batch_text]

        # Calculate total frames including pauses for early splits and first 10 batches
        num_paused_batches = 0
        num_early_splits = 0
        batch_count = 0

        for event in split_events:
            if event['type'] == 'batch':
                if batch_count < 10:
                    num_paused_batches += 1
                batch_count += 1
            elif event['type'] == 'split':
                if batch_count < 10:
                    num_early_splits += 1

        total_frames = (len(split_events) +
                       (num_paused_batches * pause_frames_per_batch) +
                       (num_early_splits * pause_frames_per_early_split) +
                       pause_frames_after_last_batch +
                       num_morph_frames + num_hold_frames + initial_pause_frames)
        ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)
        writer = animation.PillowWriter(fps=30)
        ani.save(f'partition_{split}.gif', writer=writer)
        print("Animation saved.")

        # polygons = {}

        # for batch_id, points in self.batches.items():
        #     points_array = np.array(points)

        #     # Get bounding box of the points
        #     min_x = points_array[:, 1].min()
        #     max_x = points_array[:, 1].max()
        #     min_y = points_array[:, 2].min()
        #     max_y = points_array[:, 2].max()

        #     # Create rectangle polygon
        #     polygons[batch_id] = [[min_x, min_y],
        #                           [max_x, min_y],
        #                           [max_x, max_y],
        #                           [min_x, max_y],
        #                           [min_x, min_y]]

        # features = []

        # for batch_id in sorted(self.batches.keys()):
        #     points = self.batches[batch_id]
        #     polygon_coordinates = polygons[batch_id]
        #     points_list = [[int(idx), float(lon), float(lat)] for idx, lon, lat in points] # converts points to list of floats
        #     polygon_coordinates_clean = [[float(x), float(y)] for x, y in polygon_coordinates] # converts polygon coordinates to list of floats

        #     feature = {
        #         "type": "Feature",
        #         "properties": {
        #             "batch_id": int(batch_id),
        #             "point_count": len(points),
        #             "points": points_list
        #         },
        #         "geometry": {
        #             "type": "Polygon",
        #             "coordinates": [polygon_coordinates_clean]
        #         }
        #     }
        #     features.append(feature)

        # output_file = f'/n/home01/luciagordon/mmearth-bench/{split}_batches.geojson'

        # with open(output_file, 'w') as f:
        #     import json
        #     json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)

        # print(f"GeoJSON saved to: {output_file}")
        # exit()
    def __iter__(self):
        batches = [batch[:, 0].astype(int) for batch in self.batches.values()]

        return iter(batches)

    def __len__(self):
        return len(self.batches)

class DataModule(LightningDataModule):
    def __init__(self, task, architecture, adaptation_mode, train_percent, batch_size, num_workers, seed):
        super().__init__()

        self.task = task
        self.adaptation_mode = adaptation_mode
        self.train_percent = train_percent
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.dataset = MMEarthBenchDataset(task, architecture, adaptation_mode, train_percent)

    def test_dataloader(self):
        # random_test_geographic_sampler = GeographicBatchSampler(dataset=self.dataset, split='random_test', batch_size=self.batch_size)
        # random_test_dataloader = DataLoader(self.dataset, batch_sampler=random_test_geographic_sampler, num_workers=self.num_workers, pin_memory=True)

        geographic_test_geographic_sampler = GeographicBatchSampler(dataset=self.dataset, split='geographic_test', batch_size=self.batch_size)
        geographic_test_dataloader = DataLoader(self.dataset, batch_sampler=geographic_test_geographic_sampler, num_workers=self.num_workers, pin_memory=True)

        # return [random_test_dataloader, geographic_test_dataloader]
        return geographic_test_dataloader

datamodule = DataModule(task='biomass', architecture='ConvNeXtV2A', adaptation_mode='JT-TTT-Geo', train_percent=100, batch_size=8, num_workers=1, seed=42)
test_dataloaders = datamodule.test_dataloader()

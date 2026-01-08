import os
import re
import subprocess
import sys
import utils

data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
tasks = ['biomass', 'soil_nitrogen', 'soil_organic_carbon', 'soil_pH', 'species']

# Create output directories
os.makedirs('summary_stats_tex', exist_ok=True)
os.makedirs('summary_stats_PDF', exist_ok=True)

def parse_check_h5_output(task):
    """Parse the check_h5.out file for a given task and extract modality statistics."""
    output_file = f'{data_dir_path}/{task}/output-files/check_h5.out'

    if not os.path.exists(output_file):
        # File doesn't exist yet - return empty dict to leave entries blank in table
        return {}

    stats = {}
    current_modality = None

    with open(output_file, 'r') as f:
        content = f.read()
        lines = content.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check if this is a modality name (format: "ModalityName: dtype")
            # Only treat lines as modality declarations if they have a valid dtype after the colon
            valid_dtypes = ['float32', 'float64', 'int32', 'int64', 'object', 'uint8', 'uint16', 'uint32']
            if ':' in line and not line.startswith('NaN') and not line.startswith('Min') and not line.startswith('Max'):
                parts = line.split(':', 1)  # Split only on first colon
                if len(parts) == 2:
                    modality = parts[0].strip()
                    dtype_part = parts[1].strip()
                    # Only treat as modality if it looks like a dtype declaration
                    # Check if dtype_part starts with a valid dtype (may have extra whitespace)
                    is_valid_dtype = any(dtype_part.startswith(dt) for dt in valid_dtypes)
                    # Skip non-modality keys (case-insensitive check)
                    skip_keys = ['crs', 'id', 'missing_modalities', 'sentinel2_date', 'sentinel2_system_index',
                                'transform']

                    # If this is the task name (target variable), track it but don't process its stats for the table
                    if modality.lower() == task.lower():
                        # Reset current_modality so subsequent Min/Max don't get assigned to wrong modality
                        current_modality = None
                    elif is_valid_dtype and modality.lower() not in [k.lower() for k in skip_keys]:
                        # Keep original capitalization from file
                        current_modality = modality
                        stats[current_modality] = {'nan_pct': None, 'min': None, 'max': None}

            # Check for NaN percentage
            if line.startswith('NaN pixels ='):
                nan_match = re.search(r'NaN pixels = ([\d.]+)%', line)
                if nan_match and current_modality:
                    stats[current_modality]['nan_pct'] = float(nan_match.group(1))

            # Check for Min/Max (could be on same line or separate lines)
            # Only process if we have a current modality set
            if current_modality:
                if 'Min:' in line and 'Max:' in line:
                    minmax_match = re.search(r'Min: ([\d.eE+-]+), Max: ([\d.eE+-]+)', line)
                    if minmax_match:
                        stats[current_modality]['min'] = float(minmax_match.group(1))
                        stats[current_modality]['max'] = float(minmax_match.group(2))
                elif line.startswith('Min:') and stats[current_modality]['min'] is None:
                    min_match = re.search(r'Min: ([\d.eE+-]+)', line)
                    if min_match:
                        # Check next line for Max
                        if i + 1 < len(lines) and 'Max:' in lines[i + 1]:
                            max_match = re.search(r'Max: ([\d.eE+-]+)', lines[i + 1])
                            if max_match:
                                stats[current_modality]['min'] = float(min_match.group(1))
                                stats[current_modality]['max'] = float(max_match.group(1))
                        else:
                            # Min without Max on next line - might be on same line or later
                            # Store min for now, will look for max later
                            stats[current_modality]['min'] = float(min_match.group(1))
                elif line.startswith('Max:') and stats[current_modality]['max'] is None and stats[current_modality]['min'] is not None:
                    # Max on its own line (Min was on previous line)
                    max_match = re.search(r'Max: ([\d.eE+-]+)', line)
                    if max_match:
                        stats[current_modality]['max'] = float(max_match.group(1))

            # Handle special cases for multi-band modalities (case-insensitive check)
            if current_modality and current_modality.lower() == 'aster_gdem':
                if 'Elevation band' in line:
                    # Next line should have Min/Max for elevation
                    if i + 1 < len(lines):
                        elev_match = re.search(r'Min: ([\d.eE+-]+), Max: ([\d.eE+-]+)', lines[i + 1])
                        if elev_match:
                            stats[current_modality + '_elevation'] = {
                                'nan_pct': stats[current_modality]['nan_pct'],
                                'min': float(elev_match.group(1)),
                                'max': float(elev_match.group(2))
                            }
                elif 'Slope band' in line:
                    if i + 1 < len(lines):
                        slope_match = re.search(r'Min: ([\d.eE+-]+), Max: ([\d.eE+-]+)', lines[i + 1])
                        if slope_match:
                            stats[current_modality + '_slope'] = {
                                'nan_pct': stats[current_modality]['nan_pct'],
                                'min': float(slope_match.group(1)),
                                'max': float(slope_match.group(2))
                            }

            if current_modality and current_modality.lower() == 'eth_gch':
                if 'Height band' in line:
                    if i + 1 < len(lines):
                        height_match = re.search(r'Min: ([\d.eE+-]+), Max: ([\d.eE+-]+)', lines[i + 1])
                        if height_match:
                            stats[current_modality + '_height'] = {
                                'nan_pct': stats[current_modality]['nan_pct'],
                                'min': float(height_match.group(1)),
                                'max': float(height_match.group(2))
                            }
                elif 'Uncertainty band' in line:
                    if i + 1 < len(lines):
                        unc_match = re.search(r'Min: ([\d.eE+-]+), Max: ([\d.eE+-]+)', lines[i + 1])
                        if unc_match:
                            stats[current_modality + '_uncertainty'] = {
                                'nan_pct': stats[current_modality]['nan_pct'],
                                'min': float(unc_match.group(1)),
                                'max': float(unc_match.group(2))
                            }

            i += 1

    return stats

def format_value(value, is_percentage=False):
    """Format a value for LaTeX display."""
    if value is None:
        return '--'
    if is_percentage:
        if value == 0.0:
            return '0.0'
        # Use scientific notation for percentages
        formatted = f'{value:.2e}'
        # Remove e+00 or e-00 if present (convert to regular decimal)
        if formatted.endswith('e+00') or formatted.endswith('e-00'):
            # Extract the number part and convert to float, then format as decimal
            num_part = formatted.split('e')[0]
            return f'{float(num_part):.2f}'
        return formatted
    else:
        # Format regular numbers, use scientific notation if very small or very large
        if abs(value) < 0.01 or abs(value) >= 10000:
            formatted = f'{value:.2e}'
            # Remove e+00 or e-00 if present (convert to regular decimal)
            if formatted.endswith('e+00') or formatted.endswith('e-00'):
                # Extract the number part and convert to float, then format as decimal
                num_part = formatted.split('e')[0]
                return f'{float(num_part):.2f}'
            return formatted
        else:
            return f'{value:.2f}'

def parse_task_summary(task):
    """Parse the {task}_summary.txt file for a given task and extract tile counts per split.

    Returns a dictionary mapping split names to tile counts.
    """
    summary_file = f'{data_dir_path}/{task}/output-files/{task}_summary.txt'

    if not os.path.exists(summary_file):
        return {}

    split_counts = {}

    with open(summary_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format is: "{count} {split_name} tiles"
            # Examples:
            # - "10665 train 100% tiles"
            # - "5332 train 50% tiles"
            # - "533 train 5% tiles"
            # - "2286 val tiles"
            # - "2286 random test tiles"
            # - "3156 geographic test tiles"

            tiles_match = re.search(r'^(\d+)\s+(.+?)\s+tiles$', line)
            if tiles_match:
                count = int(tiles_match.group(1))
                split_name = tiles_match.group(2).strip()

                # Normalize split names to match expected format
                split_name_lower = split_name.lower()
                if split_name_lower == 'train 100%':
                    split_counts['Train 100%'] = count
                elif split_name_lower == 'train 50%':
                    split_counts['Train 50%'] = count
                elif split_name_lower == 'train 5%':
                    split_counts['Train 5%'] = count
                elif split_name_lower == 'val':
                    split_counts['Validation'] = count
                elif split_name_lower == 'random test':
                    split_counts['Random test'] = count
                elif split_name_lower == 'geographic test':
                    split_counts['Geographic test'] = count
                else:
                    # Keep original for any unknown splits
                    split_counts[split_name] = count

    return split_counts

def create_tiles_per_split_table():
    """Create a LaTeX table with number of tiles per split by task."""
    # Collect tile counts for all tasks
    all_split_counts = {}
    for task in tasks:
        split_counts = parse_task_summary(task)
        all_split_counts[task] = split_counts

    # Get all unique split names across all tasks
    all_splits = set()
    for task_counts in all_split_counts.values():
        all_splits.update(task_counts.keys())

    # Define preferred split order
    preferred_split_order = [
        'Train 100%',
        'Train 50%',
        'Train 5%',
        'Validation',
        'Random test',
        'Geographic test'
    ]

    # Build split_order using actual split names from all_splits
    split_order = []
    all_splits_lower_map = {s.lower(): s for s in all_splits}

    # Add splits in preferred order
    for preferred in preferred_split_order:
        if preferred.lower() in all_splits_lower_map:
            split_order.append(all_splits_lower_map[preferred.lower()])

    # Add any missing splits to the end
    for split in sorted(all_splits):
        if split not in split_order:
            split_order.append(split)

    # Build LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{\\textbf{Number of tiles per split by task.}}")
    latex_lines.append("\\label{tab:tiles_per_split}")
    latex_lines.append("\\resizebox{\\linewidth}{!}{%")
    latex_lines.append("\\begin{tabular}{l" + "c" * len(tasks) + "}")
    latex_lines.append("\\toprule")

    # Header row
    task_header_parts = []
    for task in tasks:
        # Handle special cases first
        if task == 'soil_nitrogen':
            task_name = 'Soil N'
        elif task == 'soil_organic_carbon':
            task_name = 'Soil OC'
        elif task == 'soil_pH':
            task_name = 'Soil pH'
        else:
            task_name = task.replace('_', ' ').title()
        task_header_parts.append(f"\\textbf{{{task_name}}}")
    header = "\\textbf{Split} & " + " & ".join(task_header_parts) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")

    # Data rows
    for split in split_order:
        # Escape % for LaTeX (% is a comment character in LaTeX)
        split_escaped = split.replace('%', '\\%')
        row_parts = [split_escaped]
        for task in tasks:
            task_counts = all_split_counts[task]
            if split in task_counts:
                row_parts.append(str(task_counts[split]))
            else:
                row_parts.append("--")
        latex_lines.append(" & ".join(row_parts) + " \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}")
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)

def parse_task_summary_stats(task):
    """Parse the {task}_summary.txt file for a given task and extract mean and std per split.

    Returns a dictionary mapping split names to (mean, std) tuples.
    """
    summary_file = f'{data_dir_path}/{task}/output-files/{task}_summary.txt'

    if not os.path.exists(summary_file):
        return {}

    split_stats = {}

    with open(summary_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Match patterns like "Mean of train 100% values: 62.51"
            mean_match = re.search(r'Mean of (.+?) values:\s*([\d.]+)', line, re.IGNORECASE)
            if mean_match:
                split_name = mean_match.group(1).strip()
                mean_val = float(mean_match.group(2))

                # Normalize split names
                split_name_lower = split_name.lower()
                if split_name_lower == 'train 100%':
                    normalized_split = 'Train 100%'
                elif split_name_lower == 'train 50%':
                    normalized_split = 'Train 50%'
                elif split_name_lower == 'train 5%':
                    normalized_split = 'Train 5%'
                elif split_name_lower == 'val':
                    normalized_split = 'Validation'
                elif split_name_lower == 'random test':
                    normalized_split = 'Random test'
                elif split_name_lower == 'geographic test':
                    normalized_split = 'Geographic test'
                else:
                    normalized_split = split_name

                if normalized_split not in split_stats:
                    split_stats[normalized_split] = {'mean': None, 'std': None}
                split_stats[normalized_split]['mean'] = mean_val

            # Match patterns like "STD of train 100% values: 99.24"
            std_match = re.search(r'STD of (.+?) values:\s*([\d.]+)', line, re.IGNORECASE)
            if std_match:
                split_name = std_match.group(1).strip()
                std_val = float(std_match.group(2))

                # Normalize split names (same logic as above)
                split_name_lower = split_name.lower()
                if split_name_lower == 'train 100%':
                    normalized_split = 'Train 100%'
                elif split_name_lower == 'train 50%':
                    normalized_split = 'Train 50%'
                elif split_name_lower == 'train 5%':
                    normalized_split = 'Train 5%'
                elif split_name_lower == 'val':
                    normalized_split = 'Validation'
                elif split_name_lower == 'random test':
                    normalized_split = 'Random test'
                elif split_name_lower == 'geographic test':
                    normalized_split = 'Geographic test'
                else:
                    normalized_split = split_name

                if normalized_split not in split_stats:
                    split_stats[normalized_split] = {'mean': None, 'std': None}
                split_stats[normalized_split]['std'] = std_val

    # Convert to (mean, std) tuples
    result = {}
    for split, stats in split_stats.items():
        if stats['mean'] is not None and stats['std'] is not None:
            result[split] = (stats['mean'], stats['std'])

    return result

def create_split_stats_table():
    """Create a LaTeX table with mean ± std statistics for regression tasks by split."""
    # Only include regression tasks (exclude species)
    regression_tasks = ['biomass', 'soil_nitrogen', 'soil_organic_carbon', 'soil_pH']

    # Collect statistics for all tasks
    all_split_stats = {}
    for task in regression_tasks:
        all_split_stats[task] = parse_task_summary_stats(task)

    # Get all unique splits and order them
    all_splits = set()
    for task_stats in all_split_stats.values():
        all_splits.update(task_stats.keys())

    preferred_split_order = ['Train 100%', 'Train 50%', 'Train 5%', 'Validation', 'Random test', 'Geographic test']
    split_order = []
    all_splits_lower_map = {s.lower(): s for s in all_splits}

    for preferred in preferred_split_order:
        if preferred.lower() in all_splits_lower_map:
            split_order.append(all_splits_lower_map[preferred.lower()])

    for split in sorted(all_splits):
        if split not in split_order:
            split_order.append(split)

    # Build LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{\\textbf{Split statistics for regression tasks.} Values are mean $\\pm$ standard deviation.}")
    latex_lines.append("\\label{tab:split_stats}")
    latex_lines.append("\\resizebox{\\linewidth}{!}{%")
    latex_lines.append("\\begin{tabular}{l" + "c" * len(regression_tasks) + "}")
    latex_lines.append("\\toprule")

    # Header row
    task_header_parts = []
    for task in regression_tasks:
        if task == 'soil_nitrogen':
            task_name = 'Soil N'
        elif task == 'soil_organic_carbon':
            task_name = 'Soil OC'
        elif task == 'soil_pH':
            task_name = 'Soil pH'
        else:
            task_name = task.replace('_', ' ').title()
        task_header_parts.append(f"\\textbf{{{task_name}}}")
    header = "\\textbf{Split} & " + " & ".join(task_header_parts) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")

    # Data rows
    for split in split_order:
        # Escape % for LaTeX
        split_escaped = split.replace('%', '\\%')
        row_parts = [split_escaped]
        for task in regression_tasks:
            task_stats = all_split_stats[task]
            if split in task_stats:
                mean, std = task_stats[split]
                # Format with 2 decimal places in math mode
                row_parts.append(f"${mean:.2f} \\pm {std:.2f}$")
            else:
                row_parts.append("--")
        latex_lines.append(" & ".join(row_parts) + " \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}")
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)

def parse_task_summary_ranges(task):
    """Parse the {task}_summary.txt file for a given task and extract min and max per split.

    Returns a dictionary mapping split names to (min, max) tuples.
    """
    summary_file = f'{data_dir_path}/{task}/output-files/{task}_summary.txt'

    if not os.path.exists(summary_file):
        return {}

    split_ranges = {}

    with open(summary_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Match patterns like "Min of train 100% values: 0.0"
            min_match = re.search(r'Min of (.+?) values:\s*([\d.]+)', line, re.IGNORECASE)
            if min_match:
                split_name = min_match.group(1).strip()
                min_val = float(min_match.group(2))

                # Normalize split names
                split_name_lower = split_name.lower()
                if split_name_lower == 'train 100%':
                    normalized_split = 'Train 100%'
                elif split_name_lower == 'train 50%':
                    normalized_split = 'Train 50%'
                elif split_name_lower == 'train 5%':
                    normalized_split = 'Train 5%'
                elif split_name_lower == 'val':
                    normalized_split = 'Validation'
                elif split_name_lower == 'random test':
                    normalized_split = 'Random test'
                elif split_name_lower == 'geographic test':
                    normalized_split = 'Geographic test'
                else:
                    normalized_split = split_name

                if normalized_split not in split_ranges:
                    split_ranges[normalized_split] = {'min': None, 'max': None}
                split_ranges[normalized_split]['min'] = min_val

            # Match patterns like "Max of train 100% values: 1991.21"
            max_match = re.search(r'Max of (.+?) values:\s*([\d.]+)', line, re.IGNORECASE)
            if max_match:
                split_name = max_match.group(1).strip()
                max_val = float(max_match.group(2))

                # Normalize split names (same logic as above)
                split_name_lower = split_name.lower()
                if split_name_lower == 'train 100%':
                    normalized_split = 'Train 100%'
                elif split_name_lower == 'train 50%':
                    normalized_split = 'Train 50%'
                elif split_name_lower == 'train 5%':
                    normalized_split = 'Train 5%'
                elif split_name_lower == 'val':
                    normalized_split = 'Validation'
                elif split_name_lower == 'random test':
                    normalized_split = 'Random test'
                elif split_name_lower == 'geographic test':
                    normalized_split = 'Geographic test'
                else:
                    normalized_split = split_name

                if normalized_split not in split_ranges:
                    split_ranges[normalized_split] = {'min': None, 'max': None}
                split_ranges[normalized_split]['max'] = max_val

    # Convert to (min, max) tuples
    result = {}
    for split, ranges in split_ranges.items():
        if ranges['min'] is not None and ranges['max'] is not None:
            result[split] = (ranges['min'], ranges['max'])

    return result

def create_split_ranges_table():
    """Create a LaTeX table with min/max ranges for regression tasks by split."""
    # Only include regression tasks (exclude species)
    regression_tasks = ['biomass', 'soil_nitrogen', 'soil_organic_carbon', 'soil_pH']

    # Collect ranges for all tasks
    all_split_ranges = {}
    for task in regression_tasks:
        all_split_ranges[task] = parse_task_summary_ranges(task)

    # Get all unique splits and order them
    all_splits = set()
    for task_ranges in all_split_ranges.values():
        all_splits.update(task_ranges.keys())

    preferred_split_order = ['Train 100%', 'Train 50%', 'Train 5%', 'Validation', 'Random test', 'Geographic test']
    split_order = []
    all_splits_lower_map = {s.lower(): s for s in all_splits}

    for preferred in preferred_split_order:
        if preferred.lower() in all_splits_lower_map:
            split_order.append(all_splits_lower_map[preferred.lower()])

    for split in sorted(all_splits):
        if split not in split_order:
            split_order.append(split)

    # Build LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{\\textbf{Split ranges for regression tasks.} Values are [min, max].}")
    latex_lines.append("\\label{tab:split_ranges}")
    latex_lines.append("\\resizebox{\\linewidth}{!}{%")
    latex_lines.append("\\begin{tabular}{l" + "c" * len(regression_tasks) + "}")
    latex_lines.append("\\toprule")

    # Header row
    task_header_parts = []
    for task in regression_tasks:
        if task == 'soil_nitrogen':
            task_name = 'Soil N'
        elif task == 'soil_organic_carbon':
            task_name = 'Soil OC'
        elif task == 'soil_pH':
            task_name = 'Soil pH'
        else:
            task_name = task.replace('_', ' ').title()
        task_header_parts.append(f"\\textbf{{{task_name}}}")
    header = "\\textbf{Split} & " + " & ".join(task_header_parts) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")

    # Data rows
    for split in split_order:
        # Escape % for LaTeX
        split_escaped = split.replace('%', '\\%')
        row_parts = [split_escaped]
        for task in regression_tasks:
            task_ranges = all_split_ranges[task]
            if split in task_ranges:
                min_val, max_val = task_ranges[split]
                # Format with 2 decimal places
                row_parts.append(f"[{min_val:.2f}, {max_val:.2f}]")
            else:
                row_parts.append("--")
        latex_lines.append(" & ".join(row_parts) + " \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}")
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)

def create_modality_stats_table():
    """Create a LaTeX table with modality statistics across all tasks."""
    # Collect all statistics
    all_stats = {}
    for task in tasks:
        all_stats[task] = parse_check_h5_output(task)

    # Get all unique modalities across all tasks
    all_modalities = set()
    for task_stats in all_stats.values():
        all_modalities.update(task_stats.keys())

    # Define preferred modality order (case-insensitive matching)
    preferred_order_lower = [
        'sentinel2', 'sentinel1', 'aster_gdem_elevation', 'aster_gdem_slope',
        'eth_gch_height', 'eth_gch_uncertainty', 'dynamicworld', 'esa_worldcover',
        'precipitation', 'temperature', 'geolocation', 'geolocation_encoding', 'month_encoding',
        'biome', 'ecoregion',
        'msk_cldprb', 's2cloudless', 'scl'
    ]

    # Parent modalities to skip in main list
    parent_modalities_lower = ['aster_gdem', 'eth_gch']

    # Build modality_order using actual modality names from all_modalities
    modality_order = []
    all_modalities_lower_map = {m.lower(): m for m in all_modalities}

    # Add modalities in preferred order
    for preferred in preferred_order_lower:
        if preferred in all_modalities_lower_map:
            modality_order.append(all_modalities_lower_map[preferred])

    # Add any missing modalities to the end (skip parent modalities)
    for mod in sorted(all_modalities):
        if mod not in modality_order and mod.lower() not in parent_modalities_lower:
            modality_order.append(mod)

    # Build LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}")
    latex_lines.append("\\centering")
    latex_lines.append("\\resizebox{\\linewidth}{!}{%")
    latex_lines.append("\\begin{tabular}{l" + "cc" * len(tasks) + "}")
    latex_lines.append("\\toprule")

    # First header row: task names spanning 2 columns each
    task_header_parts = []
    for task in tasks:
        task_name = task.replace('_', ' ').title().replace('Ph', 'pH')
        task_header_parts.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{task_name}}}}}")
    task_header = "Modality & " + " & ".join(task_header_parts) + " \\\\"
    latex_lines.append(task_header)
    # Generate cmidrule for each task (each spans 2 columns)
    cmidrules = []
    for i in range(len(tasks)):
        start_col = 2 + 2 * i
        end_col = 3 + 2 * i
        cmidrules.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
    latex_lines.append("".join(cmidrules))

    # Second header row: column labels
    column_header_parts = []
    for task in tasks:
        column_header_parts.append("\\textbf{NaN pixel \\%}")
        column_header_parts.append("\\textbf{[Min, Max]}")
    column_header = " & " + " & ".join(column_header_parts) + " \\\\"
    latex_lines.append(column_header)
    latex_lines.append("\\midrule")

    # Data rows
    for modality in modality_order:
        # Replace underscores with spaces
        formatted_name = modality.replace('_', ' ')
        # Capitalize first letter if it's not already capitalized
        if formatted_name and formatted_name[0].islower():
            formatted_name = formatted_name[0].upper() + formatted_name[1:]
        row_parts = [formatted_name]
        for task in tasks:
            task_stats = all_stats[task]
            if modality in task_stats:
                stats = task_stats[modality]
                # If min/max exist but NaN % is missing, set it to 0
                if stats['nan_pct'] is None and (stats['min'] is not None or stats['max'] is not None):
                    stats['nan_pct'] = 0.0
                nan_pct = format_value(stats['nan_pct'], is_percentage=True)
                min_val = format_value(stats['min'])
                max_val = format_value(stats['max'])
                row_parts.append(nan_pct + "\\%")
                row_parts.append(f"[{min_val}, {max_val}]")
            else:
                row_parts.append("--")
                row_parts.append("--")
        latex_lines.append(" & ".join(row_parts) + " \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}")
    latex_lines.append("\\caption{Modality statistics (NaN percentage and [min, max] range) across all tasks}")
    latex_lines.append("\\label{tab:modality_stats}")
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)

def compile_latex(tex_file):
    """Compile a LaTeX file to PDF using pdflatex."""
    base_name = os.path.splitext(os.path.basename(tex_file))[0]
    tex_path = os.path.abspath(tex_file)
    tex_dir = os.path.dirname(tex_path)
    tex_filename = os.path.basename(tex_path)

    # Read the content of the original .tex file
    with open(tex_path, 'r') as f:
        original_content = f.read()

    # Check if the document already has a preamble
    if not original_content.strip().startswith('\\documentclass'):
        # Wrap the content in a full LaTeX document structure
        modified_content = original_content.replace('\\begin{table*}', '\\begin{table}').replace('\\end{table*}', '\\end{table}')
        wrapped_content = """\\documentclass{article}
\\usepackage{booktabs}
\\usepackage{graphicx}
\\usepackage{float}
\\usepackage[margin=1in]{geometry}
\\begin{document}
""" + modified_content + """
\\end{document}
"""
        # Write the wrapped content to a temporary .tex file for compilation
        temp_tex_file = os.path.join(tex_dir, f'{base_name}_wrapped.tex')
        with open(temp_tex_file, 'w') as f:
            f.write(wrapped_content)
        compile_target_file = temp_tex_file
    else:
        compile_target_file = tex_path

    # Run pdflatex in the tex file's directory
    result = subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', os.path.basename(compile_target_file)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tex_dir
    )
    # Run again for references
    subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', os.path.basename(compile_target_file)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tex_dir
    )

    # Move PDF to summary_stats_PDF and clean up auxiliary files
    pdf_source = os.path.join(tex_dir, f'{os.path.splitext(os.path.basename(compile_target_file))[0]}.pdf')
    pdf_dest = f'summary_stats_PDF/{base_name}.pdf'

    if os.path.exists(pdf_source):
        import shutil
        shutil.move(pdf_source, pdf_dest)
        # Clean up auxiliary files from tex directory
        for ext in ['.aux', '.log', '.out']:
            aux_file = os.path.join(tex_dir, f'{os.path.splitext(os.path.basename(compile_target_file))[0]}{ext}')
            if os.path.exists(aux_file):
                os.remove(aux_file)
        # Remove the temporary wrapped file if it was created
        if compile_target_file != tex_path and os.path.exists(compile_target_file):
            os.remove(compile_target_file)
        print(f"Compiled {tex_file} to {pdf_dest}")
    else:
        print(f"Warning: PDF not created for {tex_file}. Check LaTeX errors.")
        if result.stderr:
            print(f"Error output: {result.stderr[:500]}")

# Table generator registry
# Each entry maps a table name to a dict with:
#   - 'generator': function that returns LaTeX table string
#   - 'tex_file': path to save the .tex file
TABLE_GENERATORS = {
    'modality_stats': {
        'generator': create_modality_stats_table,
        'tex_file': 'summary_stats_tex/modality_stats.tex'
    },
    'tiles_per_split': {
        'generator': create_tiles_per_split_table,
        'tex_file': 'summary_stats_tex/tiles_per_split.tex'
    },
    'split_stats': {
        'generator': create_split_stats_table,
        'tex_file': 'summary_stats_tex/split_stats.tex'
    },
    'split_ranges': {
        'generator': create_split_ranges_table,
        'tex_file': 'summary_stats_tex/split_ranges.tex'
    }
    # Add more table types here:
    # 'another_table': {
    #     'generator': create_another_table,
    #     'tex_file': 'summary_stats_tex/another_table.tex'
    # }
}

def generate_table(table_name, compile_pdf=True):
    """Generate a specific summary statistics table.

    Args:
        table_name: Name of the table to generate (must be in TABLE_GENERATORS)
        compile_pdf: Whether to compile the LaTeX to PDF (default: True)
    """
    if table_name not in TABLE_GENERATORS:
        available = ', '.join(TABLE_GENERATORS.keys())
        raise ValueError(f"Unknown table name '{table_name}'. Available tables: {available}")

    table_config = TABLE_GENERATORS[table_name]
    generator = table_config['generator']
    tex_file = table_config['tex_file']

    # Generate LaTeX table
    latex_table = generator()

    # Save to .tex file
    with open(tex_file, 'w') as f:
        f.write(latex_table)

    print(f"Saved LaTeX table to {tex_file}")

    # Compile to PDF if requested
    if compile_pdf:
        compile_latex(tex_file)

    return tex_file

def generate_all_tables(compile_pdf=True):
    """Generate all registered summary statistics tables.

    Args:
        compile_pdf: Whether to compile each LaTeX to PDF (default: True)
    """
    for table_name in TABLE_GENERATORS.keys():
        print(f"\nGenerating {table_name} table...")
        generate_table(table_name, compile_pdf=compile_pdf)

if __name__ == '__main__':
    # Check command line arguments
    if len(sys.argv) > 1:
        # Generate specific table(s)
        for table_name in sys.argv[1:]:
            if table_name == 'all':
                generate_all_tables()
            else:
                generate_table(table_name)
    else:
        # Default: generate modality_stats table
        generate_table('modality_stats')

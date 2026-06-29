from matplotlib.ticker import FixedLocator
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import average_precision_score
import subprocess
import torch
import utils
import wandb
import re
import requests

entity = utils.read_yaml('config-user.yml')['entity']
project = utils.read_yaml('config-user.yml')['project']
tasks = ['biomass', 'soil_nitrogen', 'soil_organic_carbon', 'soil_pH', 'species']

# Create results directories if they don't exist
os.makedirs('results_figures', exist_ok=True)
os.makedirs('results_tex', exist_ok=True)
os.makedirs('results_PDF', exist_ok=True)
os.makedirs('results_cache/wandb/predictions', exist_ok=True)
os.makedirs('results_cache/wandb/runs', exist_ok=True)

RESIDUALS_WANDB_TAG = 'residuals_42'
WANDB_CACHE_DIR = 'results_cache/wandb'

def _sanitize_cache_key(value):
    return re.sub(r'[^\w.-]+', '_', value.strip('_'))

def _predictions_cache_path(tag, run_name_prefix, split, flatten=False):
    cache_dir = os.path.join(WANDB_CACHE_DIR, 'predictions', tag)
    os.makedirs(cache_dir, exist_ok=True)
    suffix = '_flat' if flatten else ''
    return os.path.join(cache_dir, f'{_sanitize_cache_key(run_name_prefix)}_{split}{suffix}.npz')

def _runs_cache_path(tag):
    cache_dir = os.path.join(WANDB_CACHE_DIR, 'runs')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'{_sanitize_cache_key(entity)}_{_sanitize_cache_key(project)}_{tag}.json')

def get_wandb_runs_by_tag(tag):
    cache_file = _runs_cache_path(tag)
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    runs = list(wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': [tag]}}))
    run_infos = [{'id': run.id, 'name': run.name} for run in runs]
    with open(cache_file, 'w') as f:
        json.dump(run_infos, f)
    return run_infos

def _find_run_for_prefix(run_infos, run_name_prefix):
    matching = [info for info in run_infos if info['name'].startswith(run_name_prefix)]
    return matching[0] if matching else None

def load_predictions_targets(run_name_prefix, split, tag=RESIDUALS_WANDB_TAG, flatten=False, run_infos=None):
    """Load predictions/targets arrays from local cache or a W&B artifact."""
    cache_file = _predictions_cache_path(tag, run_name_prefix, split, flatten)
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return {key: data[key] for key in data.files}

    if run_infos is None:
        run_infos = get_wandb_runs_by_tag(tag)
    run_info = _find_run_for_prefix(run_infos, run_name_prefix)
    if run_info is None:
        return None

    run = wandb.Api().run(f'{entity}/{project}/{run_info["id"]}')
    for artifact in run.logged_artifacts():
        if artifact.name.startswith(f'predictions_targets_{split}.pt'):
            pt = torch.load(os.path.join(artifact.download(), f'predictions_targets_{split}.pt'))
            arrays = {
                'predictions_JT': pt['predictions_JT'].float().numpy(),
                'predictions_TTT': pt['predictions_TTT'].float().numpy(),
                'targets': pt['targets'].float().numpy(),
            }
            if flatten:
                arrays = {key: value.flatten() for key, value in arrays.items()}
            np.savez_compressed(cache_file, **arrays)
            return arrays
    return None

def _load_residuals_predictions_data(task, models, splits, tag=RESIDUALS_WANDB_TAG, flatten=False):
    run_infos = get_wandb_runs_by_tag(tag)
    data = {}
    for split in splits:
        data[split] = {}
        for model in models:
            names = ['_'.join([task, model, 'JT-TTT', str(100)]) + '_',
                     '_'.join([task, model, 'JT-TTT-Geo', str(100)]) + '_']
            data[split][model] = {}
            for name in names:
                data[split][model][name] = {}
                predictions = load_predictions_targets(name, split, tag=tag, flatten=flatten, run_infos=run_infos)
                if predictions is None:
                    continue
                data[split][model][name].update(predictions)
    return data

def compile_latex(tex_file):
    """Compile a LaTeX file to PDF using pdflatex and save PDF to results_PDF folder."""
    try:
        # Get the base name for output files
        base_name = os.path.splitext(os.path.basename(tex_file))[0]
        tex_path = os.path.abspath(tex_file)
        tex_dir = os.path.dirname(tex_path)

        # Read the original table content
        with open(tex_path, 'r') as f:
            table_content = f.read()

        # Check if file already has document structure
        has_document = '\\documentclass' in table_content or '\\begin{document}' in table_content

        # Create a wrapped version for compilation if needed
        if not has_document:
            # Replace table* with table for single-column documents
            # (table* is for two-column layouts)
            modified_content = table_content.replace('\\begin{table*}', '\\begin{table}').replace('\\end{table*}', '\\end{table}')

            # Wrap table content in a proper LaTeX document
            wrapped_content = """\\documentclass{article}
\\usepackage{booktabs}
\\usepackage{graphicx}
\\usepackage{float}
\\usepackage[table,dvipsnames,rgb]{xcolor}
\\usepackage{geometry}
\\geometry{a4paper, margin=1in}
\\begin{document}
""" + modified_content + """
\\end{document}
"""
            # Write wrapped version to a temporary file
            wrapped_file = os.path.join(tex_dir, f'{base_name}_wrapped.tex')
            with open(wrapped_file, 'w') as f:
                f.write(wrapped_content)
            tex_filename = f'{base_name}_wrapped.tex'
        else:
            tex_filename = os.path.basename(tex_path)

        # Run pdflatex in the tex file's directory (may need to run twice for references)
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', tex_filename],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=tex_dir
        )
        # Run again to resolve references
        subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', tex_filename],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=tex_dir
        )

        # Move PDF to results_PDF and clean up auxiliary files
        if not has_document:
            pdf_source = os.path.join(tex_dir, f'{base_name}_wrapped.pdf')
            # Clean up wrapped tex file
            if os.path.exists(wrapped_file):
                os.remove(wrapped_file)
        else:
            pdf_source = os.path.join(tex_dir, f'{base_name}.pdf')
        pdf_dest = f'results_PDF/{base_name}.pdf'

        if os.path.exists(pdf_source):
            import shutil
            shutil.move(pdf_source, pdf_dest)
            # Clean up auxiliary files from tex directory
            for ext in ['.aux', '.log', '.out']:
                if not has_document:
                    aux_file = os.path.join(tex_dir, f'{base_name}_wrapped{ext}')
                else:
                    aux_file = os.path.join(tex_dir, f'{base_name}{ext}')
                if os.path.exists(aux_file):
                    os.remove(aux_file)
            print(f"Compiled {tex_file} to {pdf_dest}")
        else:
            print(f"Warning: PDF not created for {tex_file}. Check LaTeX errors.")
            if result.stderr:
                print(f"Error output: {result.stderr[:1000]}")
            if result.stdout:
                # Print last 500 chars of stdout for debugging
                print(f"LaTeX output (last 500 chars): {result.stdout[-500:]}")
    except FileNotFoundError:
        print(f"Warning: pdflatex not found. Install LaTeX to compile {tex_file} to PDF")
    except subprocess.TimeoutExpired:
        print(f"Warning: LaTeX compilation timed out for {tex_file}")
    except Exception as e:
        print(f"Warning: Failed to compile {tex_file}: {e}")
        import traceback
        traceback.print_exc()

architectures_plots = ['ConvNeXtV2A', 'ScaleMAE', 'DINOv3Web', 'DINOv3Sat', 'SatlasNet', 'MPMAE', 'TerraMind', 'CopernicusFM', 'Galileo', 'ConvNeXtV2AMM']

# Include S2 versions, with S2 above non-S2
architectures_tables = ['ConvNeXtV2A', 'ScaleMAE', 'DINOv3Web', 'DINOv3Sat', 'SatlasNet', 'MPMAE',
                        'TerraMindS2', 'TerraMind', 'CopernicusFMS2', 'CopernicusFM', 'GalileoS2', 'Galileo', 'ConvNeXtV2AMM']

# Create a consistent color mapping for all architectures
ARCHITECTURE_COLORS = {}
colors_list = plt.cm.tab10(np.linspace(0, 1, len(architectures_plots)))

for i, arch in enumerate(architectures_plots):
    ARCHITECTURE_COLORS[arch] = colors_list[i]

# Display-name mapping for plots/tables (keep 'Satlas' for wandb lookups)
def display_arch_name(name: str) -> str:
    if name == 'ScaleMAE':
        return 'Scale-MAE'
    elif name == 'DINOv3Web':
        return 'DINOv3 Web'
    elif name == 'DINOv3Sat':
        return 'DINOv3 Sat'
    elif name == 'CopernicusFM':
        return 'Copernicus-FM'
    elif name =='CopernicusFMS2':
        return 'Copernicus-FM S2'
    elif name == 'TerraMindS2':
        return 'TerraMind S2'
    elif name == 'GalileoS2':
        return 'Galileo S2'
    elif name == 'ConvNeXtV2AMM':
        return 'ConvNeXtV2A-MM'
    else:
        return name

# Font size configuration
LEGEND_FONTSIZE = 8
AXIS_LABEL_FONTSIZE = 8
COL_WIDTH = 4.8
MARKER_SIZE = 2
LINE_WIDTH = 1.0

# Set font to Times for all figures
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']

def tabulate_results_task(task, adaptation_mode):
    splits = ['Random', 'Geographic']
    tags = ['chi_41', 'chi_42', 'chi_43']
    seeds = [41, 42, 43]

    if adaptation_mode == 'LP':
        mode_name = 'linear probing'
    else:
        mode_name = 'finetuning'

    train_percents = [5, 50, 100]
    metric = 'R2' if task != 'species' else 'mAP'

    # Load all runs with all tags in a single API call
    all_runs_list = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': tags}})
    # Filter by tag in memory
    all_runs = {tag: [r for r in all_runs_list if tag in r.tags] for tag in tags}

    # Collect data for each seed and split
    all_data = {}

    for tag, seed in zip(tags, seeds):
        runs = all_runs[tag]
        data = {split: {architecture: {train_percent: np.nan for train_percent in train_percents} for architecture in architectures_tables} for split in splits}

        for architecture in architectures_tables:
            for train_percent in train_percents:
                name = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
                run = next((run for run in runs if run.name.startswith(name)), None)
                if run:
                    for split in splits:
                        split_metric_name = f'{split} test {metric}'
                        test_metric = run.summary_metrics.get(split_metric_name, np.nan)
                        data[split][architecture][train_percent] = test_metric

        all_data[seed] = data

    # Create DataFrames for each seed and split
    df_disps = {}
    masks = {}
    display_decimals = 2

    for seed in seeds:
        df_disps[seed] = {}
        masks[seed] = {}
        for split in splits:
            # Create DataFrame with architectures as rows and train_percents as columns
            df = pd.DataFrame.from_dict(all_data[seed][split], orient='index')
            # Apply display name mapping for row index
            df.index = [display_arch_name(idx) for idx in df.index]
            df.index.name = 'Architecture'
            df.columns.name = 'Train %'
            # Round for display
            df_disp = df.round(display_decimals)
            df_disps[seed][split] = df_disp

            # --- highlight best per column (after rounding) ---
            mask = pd.DataFrame(False, index=df_disp.index, columns=df_disp.columns, dtype=bool)

            for col in df_disp.columns:
                best = df_disp[col].max(skipna=True)
                eq = df_disp[col].eq(best).fillna(False)
                mask[col] = eq

            masks[seed][split] = mask

    # Format each DataFrame
    df_fmts = {}

    for seed in seeds:
        df_fmts[seed] = {}
        for split in splits:
            df_disp = df_disps[seed][split]
            mask = masks[seed][split]
            # Format from the rounded values
            df_fmt = df_disp.apply(lambda col: col.map(lambda x: '--' if pd.isna(x) else f'{x:.{display_decimals}f}'))
            bold = '\\textbf{' + df_fmt + '}'
            df_fmt = df_fmt.where(~mask, bold)
            df_fmts[seed][split] = df_fmt

    # Combine DataFrames: first by split (Random and Geographic), then by seed
    # Structure: Random (Seed 41, Seed 42, Seed 43) | Geographic (Seed 41, Seed 42, Seed 43)
    combined_dfs = []

    for split in splits:
        split_dfs = []
        for seed in seeds:
            df_fmt = df_fmts[seed][split].copy()
            # Rename columns to include seed number
            df_fmt.columns = [f'{col} (Seed {seed})' for col in df_fmt.columns]
            split_dfs.append(df_fmt)
        # Concatenate seeds for this split
        split_combined = pd.concat(split_dfs, axis=1)
        combined_dfs.append(split_combined)

    # Concatenate splits horizontally (Random on left, Geographic on right)
    combined_df = pd.concat(combined_dfs, axis=1)

    # Create LaTeX table
    cols_per_seed = len(train_percents)
    cols_per_split = cols_per_seed * len(seeds)
    caption_metric = r"R$^2$" if task != 'species' else "mAP"

    # Build split header with vertical line between Random and Geographic
    split_header_parts = []
    for i, split in enumerate(splits):
        if i < len(splits) - 1:
            # Not the last split: include vertical line on the right
            split_header_parts.append(f'\\multicolumn{{{cols_per_split}}}{{c|}}{{\\textbf{{{split}}}}}')
        else:
            # Last split: no vertical line on the right
            split_header_parts.append(f'\\multicolumn{{{cols_per_split}}}{{c}}{{\\textbf{{{split}}}}}')
    split_header = ' & '.join([''] + split_header_parts) + r' \\'

    # Build seed header for each split
    seed_header_parts = []
    for split in splits:
        for i, seed in enumerate(seeds):
            if i < len(seeds) - 1:
                # Not the last seed: include vertical line on the right
                seed_header_parts.append(f'\\multicolumn{{{cols_per_seed}}}{{c|}}{{\\textbf{{Seed {seed}}}}}')
            else:
                # Last seed: no vertical line on the right (unless it's the last split)
                if split == splits[-1]:
                    seed_header_parts.append(f'\\multicolumn{{{cols_per_seed}}}{{c}}{{\\textbf{{Seed {seed}}}}}')
                else:
                    seed_header_parts.append(f'\\multicolumn{{{cols_per_seed}}}{{c|}}{{\\textbf{{Seed {seed}}}}}')
    seed_header = ' & '.join([''] + seed_header_parts) + r' \\'

    # Training percentages header
    train_header = ' & '.join([''] + [f'\\textbf{{{c}\\%}}' for split in splits for seed in seeds for c in train_percents]) + r' \\'

    # Column format: l| (Model) | rrr|rrr|rrr (Random: Seeds 41,42,43) | rrr|rrr|rrr (Geographic: Seeds 41,42,43)
    # Build format parts: 'l|' for model, then for each split, 'r'*cols_per_seed for each seed
    format_parts = ['l|']
    for split_idx, split in enumerate(splits):
        for seed_idx, seed in enumerate(seeds):
            format_parts.append('r' * cols_per_seed)
            # Add | between seeds (but not after the last seed of the last split)
            if seed_idx < len(seeds) - 1 or split_idx < len(splits) - 1:
                format_parts.append('|')
    column_format = ''.join(format_parts)

    latex = combined_df.to_latex(index=True, header=False, index_names=False, escape=False, column_format=column_format)
    latex = latex.replace('\\toprule', '\\toprule\n' + split_header + '\n' + seed_header + '\n' + train_header + '\n\\midrule', 1)
    latex = ("\\begin{table*}\n\\centering\n" +
            f"\\caption{{{task.replace('_', ' ').capitalize().replace('ph', 'pH')} {mode_name} test {caption_metric}}}\n" +
            "\\resizebox{\\linewidth}{!}{%\n" +
            latex +
            "\n}\n" +
            f"\\label{{tab:{adaptation_mode}-{task}}}\n" +
            "\\end{table*}\n")

    return latex

def tabulate_iteration_numbers():
    """Tabulate the iteration number used in TTT-MMR for each task for each model
    averaged across the three seeds (41, 42, 43), using the chi_ tags.
    Extracts the number from WandB console logs: "Running TTT for X iteration(s)".
    """
    tags = ['chi_41', 'chi_42', 'chi_43']
    adaptation_mode = 'JT-TTT'

    pattern = re.compile(r"Running TTT for (\d+) iteration\(s\)")

    # Initialize results storage: arch -> task -> list of iteration counts
    results = {arch: {task: [] for task in tasks} for arch in architectures_plots}

    # Load all runs with the relevant tags and adaptation mode
    print("Loading runs from WandB...")
    all_runs_list = wandb.Api().runs(f'{entity}/{project}', filters={
        'tags': {'$in': tags},
        'config.adaptation_mode': adaptation_mode
    })

    for run in all_runs_list:
        cfg = run.config
        task = cfg.get('task')
        arch = cfg.get('architecture')

        if task not in tasks or arch not in architectures_plots:
            continue

        try:
            # Get the output.log file from WandB
            log_file = run.file("output.log")
            if log_file:
                # Use requests to get the content without creating a local file
                response = requests.get(log_file.url)
                if response.status_code == 200:
                    content = response.text
                    # Search for the pattern in content
                    matches = pattern.findall(content)
                    if matches:
                        # Use the last match as requested
                        last_iter = int(matches[-1])
                        results[arch][task].append(last_iter)
        except Exception as e:
            print(f"Warning: Could not read WandB logs for run {run.name}: {e}")

    # Compute averages across seeds
    avg_data = {arch: {task: np.nan for task in tasks} for arch in architectures_plots}
    for arch in architectures_plots:
        for task in tasks:
            if results[arch][task]:
                avg_data[arch][task] = np.mean(results[arch][task])

    # Create DataFrame for tabulation
    df = pd.DataFrame.from_dict(avg_data, orient='index')
    # Use display names for models
    df.index = [display_arch_name(idx) for idx in df.index]
    # Use pretty names for tasks
    df.columns = [task.replace('_', ' ').capitalize().replace('ph', 'pH') for task in tasks]

    # Generate LaTeX table
    latex = df.to_latex(index=True,
                        header=True,
                        index_names=False,
                        escape=False,
                        na_rep='--',
                        float_format="%.2f",
                        column_format='l' + 'c' * len(tasks))

    caption = "Average iteration number used in TTT-MMR for each task and model, averaged across seeds 41, 42, and 43."

    latex_table = (
        "\\begin{table*}[ht]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        f"{latex}\n"
        "}\n"
        "\\label{tab:ttt_iteration_numbers}\n"
        "\\end{table*}\n"
    )

    tex_file = 'results_tex/iteration_numbers.tex'
    with open(tex_file, 'w') as f:
        f.write(latex_table)

    print(f"Tabulated iteration numbers to {tex_file}")
    compile_latex(tex_file)

def tabulate_results(adaptation_mode):
    """Generate combined LaTeX table for all tasks with both Random and Geographic splits.

    Args:
        adaptation_mode: 'FT', 'LP', etc.
    """
    tex_file = f'results_tex/results_{adaptation_mode}.tex'
    with open(tex_file, 'w') as file:
        file.write('\n'.join([tabulate_results_task(task, adaptation_mode) for task in tasks]))
    compile_latex(tex_file)

def tabulate_TTT_results_task(task):
    adaptation_modes = ['FT', 'JT', 'JT-TTT', 'JT-TTT-Geo']
    splits = ['Random', 'Geographic']
    tags = ['chi_41', 'chi_42', 'chi_43']
    seeds = [41, 42, 43]
    metric = 'R2' if task != 'species' else 'mAP'
    display_name_mapping = {'JT-TTT': 'TTT-MMR', 'JT-TTT-Geo': 'TTT-MMR-Geo'}

    # Load all runs with all tags in a single API call
    all_runs_list = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': tags}})
    # Filter by tag in memory
    all_runs = {tag: [r for r in all_runs_list if tag in r.tags] for tag in tags}

    # Collect data for each seed
    all_data = {}

    for tag, seed in zip(tags, seeds):
        runs = all_runs[tag]
        data = {split: {mode: {architecture: np.nan for architecture in architectures_plots} for mode in adaptation_modes} for split in splits}

        for architecture in architectures_plots:
            for adaptation_mode in adaptation_modes:
                name = '_'.join([task, architecture, adaptation_mode, str(100)]) + '_' # uses 100% train percent
                run = next((run for run in runs if run.name.startswith(name)), None)
                if run:
                    data['Random'][adaptation_mode][architecture] = run.summary_metrics.get(f'Random test {metric}', np.nan)
                    data['Geographic'][adaptation_mode][architecture] = run.summary_metrics.get(f'Geographic test {metric}', np.nan)

        all_data[seed] = data

    def _one_seed_table(seed):
        # Create DataFrames for Random and Geographic splits
        df_random = pd.DataFrame.from_dict(all_data[seed]['Random'], orient='index')[architectures_plots].reindex(adaptation_modes).T
        df_geographic = pd.DataFrame.from_dict(all_data[seed]['Geographic'], orient='index')[architectures_plots].reindex(adaptation_modes).T

        # Apply display name mapping to index
        df_random.index = [display_arch_name(idx) for idx in df_random.index]
        df_geographic.index = [display_arch_name(idx) for idx in df_geographic.index]
        df_random.index.name = 'Model'
        df_geographic.index.name = 'Model'

        # Format each DataFrame
        formatted_dfs = {}

        for split_name, df in [('Random', df_random), ('Geographic', df_geographic)]:
            formatted_df = df.copy()

            # --- bold highest per column (after rounding to 2 decimals), robust to strings ---
            for col in df.columns:
                # numeric view for max
                col_numeric = pd.to_numeric(df[col], errors='coerce')
                if col_numeric.dropna().empty:
                    formatted_df[col] = "--"
                    continue
                col_max_r2 = col_numeric.round(2).max()

                def _fmt_cell(x):
                    x_num = pd.to_numeric(x, errors='coerce')
                    if pd.isna(x_num):
                        return "--"
                    s = f"{x_num:.2f}"
                    return f"\\textbf{{{s}}}" if round(float(x_num), 2) == col_max_r2 else s

                formatted_df[col] = df[col].apply(_fmt_cell)

            # --- underline row-wise max (tie-friendly), robust to strings ---
            for idx in df.index:
                row_num = pd.to_numeric(df.loc[idx], errors='coerce')
                if row_num.dropna().empty:
                    continue
                row_max = row_num.round(2).max()
                for mode in adaptation_modes:
                    x_num = pd.to_numeric(df.at[idx, mode], errors='coerce')
                    if pd.isna(x_num):
                        continue
                    if round(float(x_num), 2) == row_max:
                        formatted_df.at[idx, mode] = f"\\underline{{{formatted_df.at[idx, mode]}}}"

            formatted_dfs[split_name] = formatted_df

        # Combine Random and Geographic horizontally (side by side)
        df_random_fmt = formatted_dfs['Random'].copy()
        df_geographic_fmt = formatted_dfs['Geographic'].copy()

        # Concatenate horizontally (columns are already the adaptation modes)
        combined_df = pd.concat([df_random_fmt, df_geographic_fmt], axis=1)

        # Create LaTeX table
        num_adaptation_modes = len(adaptation_modes)
        cols_per_split = num_adaptation_modes
        cols = combined_df.columns.tolist()

        # Build group header: Split headers with vertical line between Random and Geographic
        split_header_parts = []
        for i, split in enumerate(splits):
            if i < len(splits) - 1:
                # Not the last split: include vertical line on the right
                split_header_parts.append(f'\\multicolumn{{{cols_per_split}}}{{c|}}{{\\textbf{{{split}}}}}')
            else:
                # Last split: no vertical line on the right
                split_header_parts.append(f'\\multicolumn{{{cols_per_split}}}{{c}}{{\\textbf{{{split}}}}}')
        split_header = ' & '.join([''] + split_header_parts) + r' \\'

        # Second header row: Adaptation modes for each split (with display name mapping)
        mode_header_parts = []
        for split in splits:
            display_modes = [display_name_mapping.get(mode, mode) for mode in adaptation_modes]
            mode_header_parts.extend(display_modes)
        mode_header = '\\textbf{Model} & ' + ' & '.join(mode_header_parts) + r' \\'

        # Column format: l (Model) | cccc (Random) | cccc (Geographic)
        column_format = 'l|' + ('c' * cols_per_split) + '|' + ('c' * cols_per_split)
        body = combined_df.to_latex(index=True,
                                     header=False,
                                     index_names=False,
                                     escape=False,
                                     na_rep='--',
                                     column_format=column_format,
                                     multicolumn=False,
                                     multirow=False)

        caption_metric = r"R$^2$" if metric == 'R2' else "mAP"
        # Replace \toprule\n with our custom headers
        body = body.replace("\\toprule\n", "\\toprule\n" + split_header + "\n" + mode_header + "\n\\midrule", 1)

        latex = ("\\begin{table}[H]\n\\centering\n" +
                 f"\\caption{{{task.replace('_', ' ').capitalize().replace('ph', 'pH')} test {caption_metric} by architecture, adaptation mode, and split (Seed {seed})}}\n" +
                 f"\\label{{tab:{task}_{metric}_seed{seed}}}\n" +
                 "\\resizebox{\\linewidth}{!}{%\n" +
                 body +
                 "\n}\n" +
                 "\\end{table}\n")
        return latex

    # Generate one table per seed
    latex_tables = []
    for seed in seeds:
        latex_tables.append(_one_seed_table(seed))

    return "\n".join(latex_tables)

def tabulate_TTT_results():
    tasks = ['biomass', 'soil_nitrogen', 'soil_organic_carbon', 'soil_pH', 'species']
    latex = '\n'.join([tabulate_TTT_results_task(task) for task in tasks])

    tex_file = 'results_tex/results_TTT.tex'
    with open(tex_file, 'w') as file:
        file.write(latex)
    compile_latex(tex_file)

def tabulate_TTT_modality_excluded():
    """Create a modality-exclusion table for CopernicusFM and DINOv3Sat JT-TTT runs.

    Rows are modalities listed in task_modalities.json.
    Columns are tasks x splits (Random, Geographic).
    Uses R2 for regression tasks and mAP for species.
    """
    task_list = ['biomass', 'soil_nitrogen', 'soil_organic_carbon', 'soil_pH', 'species']
    split_list = ['Random', 'Geographic']
    modalities = utils.read_json('task_modalities.json')
    architectures = ['CopernicusFM', 'DINOv3Sat']
    adaptation_mode = 'JT-TTT'
    train_percent = 100

    all_runs = list(wandb.Api().runs(f'{entity}/{project}'))
    all_latex = []

    for architecture in architectures:
        # Collect metrics per modality, task, and split.
        data = {
            modality: {task: {split: np.nan for split in split_list} for task in task_list}
            for modality in modalities
        }

        for modality in modalities:
            modality_tag = f'excluded_{modality}'
            for task in task_list:
                metric = 'R2' if task != 'species' else 'mAP'
                prefix = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
                matching_runs = [
                    run for run in all_runs
                    if modality_tag in run.tags and run.name.startswith(prefix)
                ]

                for split in split_list:
                    metric_name = f'{split} test {metric}'
                    values = [
                        run.summary_metrics.get(metric_name, np.nan)
                        for run in matching_runs
                    ]
                    numeric_values = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
                    if not numeric_values.empty:
                        data[modality][task][split] = float(numeric_values.mean())

        # Build a fixed-width row matrix (11 columns total):
        # excluded modality + (5 tasks x 2 splits).
        columns = [(task, split) for task in task_list for split in split_list]

        def _latex_escape(text):
            return str(text).replace('_', r'\_')

        def _task_label(task_name):
            return task_name.replace('_', ' ').capitalize().replace('ph', 'pH')

        # Header row 1: Architecture name spanning ten columns.
        display_name = display_arch_name(architecture)
        top_header = f'\\textbf{{Excluded modality}} & \\multicolumn{{10}}{{c}}{{\\textbf{{{display_name}}}}} \\\\'

        # Header row 2: each task spans Random + Geographic.
        task_headers = [
            f'\\multicolumn{{2}}{{c}}{{\\textbf{{{_task_label(task)}}}}}'
            for task in task_list
        ]
        mid_header = ' & '.join([''] + task_headers) + r' \\'

        # Header row 1: Random/Geographic for each task.
        split_headers = []
        for _ in task_list:
            split_headers.extend([r'\textbf{Random}', r'\textbf{Geographic}'])
        low_header = r'\textbf{Excluded modality} & ' + ' & '.join(split_headers) + r' \\'

        # Column format: one left column + ten centered columns grouped by task.
        column_format = 'l|' + '|'.join(['cc'] * len(task_list))
        body_rows = []
        for modality in modalities:
            row_values = [_latex_escape(modality)]
            for task, split in columns:
                value = data[modality][task][split]
                row_values.append('--' if pd.isna(value) else f'{value:.3f}')
            body_rows.append(' & '.join(row_values) + r' \\')

        table_body = (
            f"\\begin{{tabular}}{{{column_format}}}\n"
            "\\toprule\n"
            + top_header + "\n"
            + mid_header + "\n"
            + low_header + "\n"
            + "\\midrule\n"
            + "\n".join(body_rows) + "\n"
            + "\\bottomrule\n"
            + "\\end{tabular}"
        )

        latex = (
            "\\begin{table*}[ht]\n"
            "\\centering\n"
            f"\\caption{{\\textbf{{{display_name} test performance when excluding one modality at TTT time.}} "
            "Rows denote the excluded modality (tagged as excluded\\_MODALITY). "
            "Columns report Random and Geographic test performance for each task "
            "(R$^2$ for regression tasks and mAP for species).}\n"
            "\\resizebox{\\linewidth}{!}{%\n"
            f"{table_body}\n"
            "}\n"
            f"\\label{{tab:ttt_modality_excluded_{architecture.lower()}}}\n"
            "\\end{table*}\n"
        )
        all_latex.append(latex)

    final_latex = "\n\n".join(all_latex)

    tex_file = 'results_tex/tabulate_TTT_modality_excluded.tex'
    with open(tex_file, 'w') as file:
        file.write(final_latex)
    compile_latex(tex_file)


def plot_TTT_modality_excluded():
    task_list = ['biomass', 'soil_nitrogen', 'soil_organic_carbon', 'soil_pH', 'species']
    split_list = ['Random', 'Geographic']
    modalities = utils.read_json('task_modalities.json')
    architectures = ['CopernicusFM', 'DINOv3Sat']
    train_percent = 100

    all_runs = list(wandb.Api().runs(f'{entity}/{project}'))

    for architecture in architectures:
        # Collect baseline performance per task and split (base JT from chi_42).
        baseline_perf = {split: {task: np.nan for task in task_list} for split in split_list}

        for split in split_list:
            for task in task_list:
                metric = 'R2' if task != 'species' else 'mAP'
                metric_name = f'{split} test {metric}'

                jt_prefix = '_'.join([task, architecture, 'JT', str(train_percent)]) + '_'
                jt_runs = [
                    run for run in all_runs
                    if 'chi_42' in run.tags and run.name.startswith(jt_prefix)
                ]

                if jt_runs:
                    jt_value = jt_runs[0].summary_metrics.get(metric_name, np.nan)
                    if not np.isnan(jt_value):
                        baseline_perf[split][task] = float(jt_value)

        # Collect delta performance per modality, task, and split.
        delta_by_task_split = {split: {task: [] for task in task_list} for split in split_list}

        columns = ['None'] + modalities

        for modality in columns:
            if modality == 'None':
                modality_tag = None
            else:
                modality_tag = f'excluded_{modality}'

            for task in task_list:
                metric = 'R2' if task != 'species' else 'mAP'
                ttt_prefix = '_'.join([task, architecture, 'JT-TTT', str(train_percent)]) + '_'

                if modality_tag is None:
                    ttt_runs = [
                        run for run in all_runs
                        if 'chi_42' in run.tags and not any(tag.startswith('excluded_') for tag in run.tags) and run.name.startswith(ttt_prefix)
                    ]
                else:
                    ttt_runs = [
                        run for run in all_runs
                        if modality_tag in run.tags and run.name.startswith(ttt_prefix)
                    ]

                for split in split_list:
                    perfs = []
                    metric_name = f'{split} test {metric}'

                    for ttt_run in ttt_runs:
                        ttt_value = ttt_run.summary_metrics.get(metric_name, np.nan)
                        if not np.isnan(ttt_value):
                            perfs.append(ttt_value)

                    perf = float(np.mean(perfs)) if perfs else np.nan
                    baseline = baseline_perf[split][task]

                    if not pd.isna(perf) and not pd.isna(baseline):
                        delta_by_task_split[split][task].append(perf - baseline)
                    else:
                        delta_by_task_split[split][task].append(np.nan)

        def _modality_xtick_label(m):
            if m == 'geolocation_encoding':
                return 'geolocation'
            if m == 'month_encoding':
                return 'month'
            return m.replace('_', ' ')

        # Plot
        for split in split_list:
            fig, ax = plt.subplots(figsize=(6, 2.1))
            x = np.arange(len(columns))
            task_colors = ['green', 'blue', 'brown', 'purple', 'mediumvioletred']

            for color, task in zip(task_colors, task_list):
                y = np.array(delta_by_task_split[split][task], dtype=float)
                ax.plot(
                    x, y,
                    marker='o',
                    linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE + 1,
                    color=color,
                    label=task.replace('_', ' ').capitalize().replace('nitrogen', 'N').replace('organic carbon', 'OC').replace('ph', 'pH')
                )

            ax.axhline(y=0, color='black', linewidth=LINE_WIDTH)

            ax.set_xticks(x)
            ax.set_xticklabels(
                [_modality_xtick_label(m) for m in columns],
                fontsize=AXIS_LABEL_FONTSIZE,
                rotation=20,
                ha='right',
            )
            ax.set_xlabel('Excluded modality', fontsize=AXIS_LABEL_FONTSIZE)

            ax.set_ylabel('Δ Performance', fontsize=AXIS_LABEL_FONTSIZE)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='y', labelsize=AXIS_LABEL_FONTSIZE, pad=3, labelrotation=0)
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=len(task_list), fontsize=LEGEND_FONTSIZE, frameon=False, handletextpad=0.3, columnspacing=1.0)
            plt.subplots_adjust(left=0.12, right=0.98, top=0.78, bottom=0.32)

            plt.savefig(f'results_figures/TTT_modality_excluded_plot_{architecture.lower()}_{split.lower()}.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(f'results_figures/TTT_modality_excluded_plot_{architecture.lower()}_{split.lower()}.png', dpi=300, bbox_inches='tight')
            plt.close()

def compare_JT_TTT_reconstruction_quality():
    """Tabulate **JT-TTT minus JT** per-modality reconstruction deltas for ``architectures_plots``.

    Modalities come from ``task_modalities.json``. W&B metric keys match ``model.py``:

    - Continuous: ``{Random|Geographic} test <modality> R2``
    - Categorical: ``{Random|Geographic} test <modality> accuracy``

    Categorical modalities match ``model.py`` (``DynamicWorld``, ``ESA_WorldCover``, ``biome``,
    ``ecoregion``).

    One LaTeX ``table*`` **per benchmark task** (no averaging across tasks): each cell is
    JT-TTT $-$ JT for that task and model only.

    JT runs use tag ``chi_42``; JT-TTT runs use ``residuals_42``. JT-TTT runs with ``excluded_*``
    tags are skipped.

    Train fraction is 100.

    Writes ``results_tex/reconstruction_quality_jt_ttt_minus_jt.tex`` (several ``table*`` environments)
    and runs ``compile_latex`` for ``results_PDF/reconstruction_quality_jt_ttt_minus_jt.pdf``.

    Returns ``dict[task, DataFrame]`` with task-specific delta tables for inspection.
    """
    train_percent = 100
    tag_jt = 'chi_42'
    tag_jt_ttt = 'residuals_42'
    splits = ['Random', 'Geographic']
    modalities = utils.read_json('task_modalities.json')
    categorical_modalities = frozenset({'DynamicWorld', 'ESA_WorldCover', 'biome', 'ecoregion'})

    def _metric_key(split, modality):
        suffix = 'accuracy' if modality in categorical_modalities else 'R2'
        return f'{split} test {modality} {suffix}'

    def _column_label(split, modality):
        kind = 'acc' if modality in categorical_modalities else 'R²'
        return f'Δ ({kind}) {split} — {modality}'

    def _task_caption_label(task_name):
        return task_name.replace('_', ' ').capitalize().replace('ph', 'pH')

    all_runs_list = wandb.Api().runs(
        f'{entity}/{project}',
        filters={'tags': {'$in': [tag_jt, tag_jt_ttt]}},
    )
    runs_jt = [r for r in all_runs_list if tag_jt in r.tags]
    runs_jt_ttt = [r for r in all_runs_list if tag_jt_ttt in r.tags]

    def _standard_jt_ttt_run(run):
        return not any(t.startswith('excluded_') for t in getattr(run, 'tags', ()) or [])

    def _get_run_metric(pool, task, architecture, adaptation_mode, split, modality):
        prefix = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
        run = next((r for r in pool if r.name.startswith(prefix)), None)
        if run is None:
            return np.nan
        if adaptation_mode == 'JT-TTT' and not _standard_jt_ttt_run(run):
            return np.nan
        key = _metric_key(split, modality)
        v = run.summary_metrics.get(key, np.nan)
        if v is None:
            return np.nan
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    metric_columns = [_column_label(sp, m) for sp in splits for m in modalities]
    n_modalities = len(modalities)

    def _modality_header(modality):
        if '_' in modality:
            return r'\shortstack{' + r' \\ '.join(modality.split('_')) + '}'
        return modality

    def _fmt_cell(x):
        return '--' if pd.isna(x) else f'{float(x):.4f}'

    modality_headers = [_modality_header(m) for m in modalities]
    row1 = ''.join(['c'] * n_modalities)
    column_format = f'l|{row1}|{row1}'
    split_rand = '\\multicolumn{%d}{c|}{\\textbf{Random} ($\\Delta$ TTT-MMR$-$JT)}' % n_modalities
    split_geo = '\\multicolumn{%d}{c}{\\textbf{Geographic} ($\\Delta$ TTT-MMR$-$JT)}' % n_modalities
    header_top = '& ' + split_rand + ' & ' + split_geo + r' \\'
    sub_header = (
        '\\textbf{Model} & '
        + ' & '.join(modality_headers)
        + ' & '
        + ' & '.join(modality_headers)
        + r' \\'
    )

    latex_blocks = []
    dfs_by_task = {}

    for task in tasks:
        rows_out = []
        for architecture in architectures_plots:
            row_data = {'Model': architecture}
            for split in splits:
                for modality in modalities:
                    jt_v = _get_run_metric(runs_jt, task, architecture, 'JT', split, modality)
                    ttt_v = _get_run_metric(runs_jt_ttt, task, architecture, 'JT-TTT', split, modality)
                    col = _column_label(split, modality)
                    if np.isnan(jt_v) or np.isnan(ttt_v):
                        row_data[col] = np.nan
                    else:
                        row_data[col] = float(ttt_v - jt_v)
            rows_out.append(row_data)

        df = pd.DataFrame(rows_out)
        df = df[['Model'] + metric_columns]
        dfs_by_task[task] = df

        body_rows = []
        for _, row in df.iterrows():
            arch = row['Model']
            cells = [_fmt_cell(row[c]) for c in metric_columns]
            body_rows.append(display_arch_name(arch) + ' & ' + ' & '.join(cells) + r' \\')

        tabular = (
            f'\\begin{{tabular}}{{{column_format}}}\n'
            '\\toprule\n'
            + header_top + '\n'
            + sub_header + '\n'
            '\\midrule\n'
            + '\n'.join(body_rows) + '\n'
            '\\bottomrule\n'
            '\\end{tabular}'
        )

        task_tex = _task_caption_label(task).replace('&', r'\&')
        label_safe = task.replace('_', '')
        latex_blocks.append(
            '\\begin{table*}[ht]\n'
            '\\centering\n'
            f'\\caption{{\\textbf{{{task_tex} per-modality reconstruction improvement for TTT-MMR over joint training (TTT-MMR $-$ JT).}} '
            r'R$^2$ for continuous modalities, accuracy for '
            r'\texttt{DynamicWorld}, \texttt{ESA\_WorldCover}, \texttt{biome}, \texttt{ecoregion}. '
            '}\n'
            '\\resizebox{\\linewidth}{!}{%\n'
            + tabular + '\n'
            '}\n'
            f'\\label{{tab:reconstruction_jt_ttt_minus_jt_{label_safe}}}\n'
            '\\end{table*}\n'
            '\\vspace{2\\baselineskip}\n'
        )

        print(f'\n=== {task} ===')
        print(df.to_string(index=False))

    latex = ''.join(latex_blocks)

    tex_path = 'results_tex/reconstruction_quality_jt_ttt_minus_jt.tex'
    with open(tex_path, 'w') as f:
        f.write(latex)
    print(f"\nWrote {tex_path}")
    compile_latex(tex_path)
    return dfs_by_task

def tabulate_JT_reconstruction_quality():
    """Tabulate raw **JT** per-modality reconstruction numbers for ``architectures_plots`` averaged over seeds 41, 42, 43.

    Similar to compare_JT_TTT_reconstruction_quality() but showing raw numbers instead of deltas,
    and averaging over three seeds.
    """
    train_percent = 100
    tags_jt = ['chi_41', 'chi_42', 'chi_43']
    splits = ['Random', 'Geographic']
    modalities = utils.read_json('task_modalities.json')
    categorical_modalities = frozenset({'DynamicWorld', 'ESA_WorldCover', 'biome', 'ecoregion'})

    def _metric_key(split, modality):
        suffix = 'accuracy' if modality in categorical_modalities else 'R2'
        return f'{split} test {modality} {suffix}'

    def _column_label(split, modality):
        kind = 'acc' if modality in categorical_modalities else 'R²'
        return f'{kind} {split} -- {modality}'

    def _task_caption_label(task_name):
        return task_name.replace('_', ' ').capitalize().replace('ph', 'pH')

    all_runs_list = wandb.Api().runs(
        f'{entity}/{project}',
        filters={'tags': {'$in': tags_jt}},
    )
    # Group runs by tag
    runs_by_tag = {tag: [r for r in all_runs_list if tag in r.tags] for tag in tags_jt}

    metric_columns = [_column_label(sp, m) for sp in splits for m in modalities]
    n_modalities = len(modalities)

    def _modality_header(modality):
        if '_' in modality:
            return r'\shortstack{' + r' \\ '.join(modality.split('_')) + '}'
        return modality

    def _fmt_cell(x):
        return '--' if pd.isna(x) else f'{float(x):.4f}'

    modality_headers = [_modality_header(m) for m in modalities]
    row1 = ''.join(['c'] * n_modalities)
    column_format = f'l|{row1}|{row1}'
    split_rand = '\\multicolumn{%d}{c|}{\\textbf{Random}}' % n_modalities
    split_geo = '\\multicolumn{%d}{c}{\\textbf{Geographic}}' % n_modalities
    header_top = '& ' + split_rand + ' & ' + split_geo + r' \\'
    sub_header = (
        '\\textbf{Model} & '
        + ' & '.join(modality_headers)
        + ' & '
        + ' & '.join(modality_headers)
        + r' \\'
    )

    latex_blocks = []
    dfs_by_task = {}

    for task in tasks:
        rows_out = []
        for architecture in architectures_plots:
            row_data = {'Model': architecture}
            for split in splits:
                for modality in modalities:
                    col = _column_label(split, modality)
                    seed_values = []
                    for tag in tags_jt:
                        prefix = '_'.join([task, architecture, 'JT', str(train_percent)]) + '_'
                        run = next((r for r in runs_by_tag[tag] if r.name.startswith(prefix)), None)
                        if run is not None:
                            key = _metric_key(split, modality)
                            v = run.summary_metrics.get(key, np.nan)
                            try:
                                v_float = float(v) if v is not None else np.nan
                                if not np.isnan(v_float):
                                    seed_values.append(v_float)
                            except (TypeError, ValueError):
                                pass

                    if seed_values:
                        row_data[col] = np.mean(seed_values)
                    else:
                        row_data[col] = np.nan
            rows_out.append(row_data)

        df = pd.DataFrame(rows_out)
        df = df[['Model'] + metric_columns]
        dfs_by_task[task] = df

        body_rows = []
        for _, row in df.iterrows():
            arch = row['Model']
            cells = [_fmt_cell(row[c]) for c in metric_columns]
            body_rows.append(display_arch_name(arch) + ' & ' + ' & '.join(cells) + r' \\')

        tabular = (
            f'\\begin{{tabular}}{{{column_format}}}\n'
            '\\toprule\n'
            + header_top + '\n'
            + sub_header + '\n'
            '\\midrule\n'
            + '\n'.join(body_rows) + '\n'
            '\\bottomrule\n'
            '\\end{tabular}'
        )

        task_tex = _task_caption_label(task).replace('&', r'\&')
        label_safe = task.replace('_', '')
        latex_blocks.append(
            '\\begin{table*}[ht]\n'
            '\\centering\n'
            f'\\caption{{\\textbf{{{task_tex} per-modality reconstruction quality after JT (averaged across seeds 41, 42, 43).}} '
            r'R$^2$ for continuous modalities, accuracy for '
            r'\texttt{DynamicWorld}, \texttt{ESA\_WorldCover}, \texttt{biome}, \texttt{ecoregion}. '
            r'}\n'
            '\\resizebox{\\linewidth}{!}{%\n'
            + tabular + '\n'
            '}\n'
            f'\\label{{tab:reconstruction_jt_raw_{label_safe}}}\n'
            '\\end{table*}\n'
            '\\vspace{2\\baselineskip}\n'
        )

        print(f'\n=== {task} (JT Raw Avg) ===')
        print(df.to_string(index=False))

    latex = ''.join(latex_blocks)

    tex_path = 'results_tex/reconstruction_quality_jt_raw.tex'
    with open(tex_path, 'w') as f:
        f.write(latex)
    print(f"\nWrote {tex_path}")
    compile_latex(tex_path)
    return dfs_by_task


def check_TTT_need():
    """Mean JT-TTT-Geo minus JT test performance for hand-picked model per (task, split), over 3 seeds.

    Uses W&B runs tagged ``chi_41``, ``chi_42``, ``chi_43``. Train fraction 100. Main task metric:
    Random/Geographic test R$^2$ (regression) or mAP (species). JT-TTT-Geo runs with ``excluded_*`` tags
    are ignored.

    Model per task and test split (R = Random, G = Geographic):

    - Biomass R/G: MPMAE
    - Soil N R: TerraMind; Soil N G: CopernicusFM
    - Soil OC R/G: CopernicusFM
    - Soil pH R/G: TerraMind
    - Species R/G: CopernicusFM
    """
    train_percent = 100
    tags = ['chi_41', 'chi_42', 'chi_43']

    architecture_by_task_split = {
        ('biomass', 'Random'): 'MPMAE',
        ('biomass', 'Geographic'): 'MPMAE',
        ('soil_nitrogen', 'Random'): 'TerraMind',
        ('soil_nitrogen', 'Geographic'): 'CopernicusFM',
        ('soil_organic_carbon', 'Random'): 'CopernicusFM',
        ('soil_organic_carbon', 'Geographic'): 'CopernicusFM',
        ('soil_pH', 'Random'): 'TerraMind',
        ('soil_pH', 'Geographic'): 'TerraMind',
        ('species', 'Random'): 'CopernicusFM',
        ('species', 'Geographic'): 'CopernicusFM',
    }

    def _standard_jt_ttt_run(run):
        return not any(t.startswith('excluded_') for t in getattr(run, 'tags', ()) or [])

    def _run_for(pool, task, architecture, adaptation_mode):
        prefix = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
        return next((r for r in pool if r.name.startswith(prefix)), None)

    all_runs_list = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': tags}})
    runs_by_tag = {tag: [r for r in all_runs_list if tag in r.tags] for tag in tags}

    rows = []
    for task in tasks:
        metric = 'R2' if task != 'species' else 'mAP'
        for split in ['Random', 'Geographic']:
            architecture = architecture_by_task_split[(task, split)]
            metric_name = f'{split} test {metric}'
            deltas = []
            for tag in tags:
                pool = runs_by_tag[tag]
                jt_run = _run_for(pool, task, architecture, 'JT')
                ttt_run = _run_for(pool, task, architecture, 'JT-TTT-Geo')
                if jt_run is None or ttt_run is None or not _standard_jt_ttt_run(ttt_run):
                    continue
                jv = jt_run.summary_metrics.get(metric_name, np.nan)
                tv = ttt_run.summary_metrics.get(metric_name, np.nan)
                if jv is None or tv is None:
                    continue
                try:
                    jv, tv = float(jv), float(tv)
                except (TypeError, ValueError):
                    continue
                if np.isnan(jv) or np.isnan(tv):
                    continue
                deltas.append(tv - jv)

            mean_d = float(np.mean(deltas)) if deltas else np.nan
            std_e = (
                float(np.std(deltas, ddof=1) / np.sqrt(len(deltas)))
                if len(deltas) > 1
                else (0.0 if deltas else np.nan)
            )
            rows.append({
                'task': task,
                'split': split,
                'architecture': architecture,
                'metric': metric,
                'mean_jt_ttt_minus_jt': mean_d,
                'std_error_across_seeds': std_e,
                'n_seeds': len(deltas),
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    def _task_label(task_name):
        return task_name.replace('_', ' ').capitalize().replace('ph', 'pH')

    def _fmt_val(x):
        if pd.isna(x):
            return '--'
        if abs(float(x)) < 0.001:
            return f'{float(x):.5f}'
        return f'{float(x):.4f}'

    def _fmt_delta(mean_d, std_e):
        if pd.isna(mean_d):
            return '--'
        mean_str = _fmt_val(mean_d)
        if pd.isna(std_e):
            return f'${mean_str}$'
        return f'${mean_str} \\pm {_fmt_val(std_e)}$'

    body_rows = []
    for task in tasks:
        task_rows = df[df['task'] == task]
        rand_row = task_rows[task_rows['split'] == 'Random']
        geo_row = task_rows[task_rows['split'] == 'Geographic']

        if not rand_row.empty:
            rand = rand_row.iloc[0]
            rand_model = display_arch_name(rand['architecture'])
            rand_delta = _fmt_delta(rand['mean_jt_ttt_minus_jt'], rand['std_error_across_seeds'])
        else:
            rand_model, rand_delta = '--', '--'

        if not geo_row.empty:
            geo = geo_row.iloc[0]
            geo_model = display_arch_name(geo['architecture'])
            geo_delta = _fmt_delta(geo['mean_jt_ttt_minus_jt'], geo['std_error_across_seeds'])
        else:
            geo_model, geo_delta = '--', '--'

        body_rows.append(
            f"\\textbf{{{_task_label(task)}}} & {rand_model} & {rand_delta} & {geo_model} & {geo_delta} \\\\"
        )

    column_format = 'l|cc|cc'
    header_top = (
        ' & \\multicolumn{2}{c|}{\\textbf{Random}} '
        '& \\multicolumn{2}{c}{\\textbf{Geographic}} \\\\'
    )
    header_sub = (
        r'\textbf{Task} & \textbf{Model} & $\Delta$ Performance '
        r'& \textbf{Model} & $\Delta$ Performance \\'
    )
    tabular = (
        f"\\begin{{tabular}}{{{column_format}}}\n"
        "\\toprule\n"
        + header_top + "\n"
        + header_sub + "\n"
        + "\\midrule\n"
        + "\n".join(body_rows) + "\n"
        + "\\bottomrule\n"
        + "\\end{tabular}\n"
    )
    latex = (
        "\\begin{table}[H]\n\\centering\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        + tabular +
        "}\n"
        "\\caption{Improvement of TTT-MMR-Geo over joint training (mean $\\pm$ standard error across "
        "3 seeds) for the best model for each task and test split at 100\\% training data.}\n"
        "\\label{tab:check_ttt_need}\n"
        "\\end{table}\n"
    )

    tex_path = 'results_tex/check_TTT_need.tex'
    with open(tex_path, 'w') as f:
        f.write(latex)
    compile_latex(tex_path)

    return df

def compute_two_y_ticks(y_min, y_max, step=0.1, min_spacing=0.2, lower_tick_above_min=False):
    """Compute two y-axis tick locations using the same logic as plot_rq3_performance."""
    tick_decimals = 0 if step >= 1 else 1
    tick_bottom = np.ceil(y_min / step) * step
    tick_top = np.floor(y_max / step) * step
    if step >= 1:
        tick_bottom = int(tick_bottom)
        tick_top = int(tick_top)
        if tick_top < tick_bottom:
            tick_bottom = int(np.ceil(y_min))
            tick_top = int(np.ceil(y_max))
    else:
        tick_bottom = round(tick_bottom, tick_decimals)
        tick_top = round(tick_top, tick_decimals)

    tick_top_expanded = tick_top - tick_bottom < min_spacing - 1e-9
    if tick_top_expanded:
        tick_top = tick_bottom + min_spacing
        if step >= 1:
            tick_top = int(tick_top)
        else:
            tick_top = round(tick_top, tick_decimals)

    if lower_tick_above_min and np.isclose(tick_bottom, y_min, rtol=0, atol=step / 10):
        tick_bottom += step
        if step >= 1:
            tick_bottom = int(tick_bottom)
        else:
            tick_bottom = round(tick_bottom, tick_decimals)
        if tick_top - tick_bottom < min_spacing - 1e-9:
            tick_top = tick_bottom + min_spacing
            tick_top_expanded = True
            if step >= 1:
                tick_top = int(tick_top)
            else:
                tick_top = round(tick_top, tick_decimals)

    return tick_bottom, tick_top, tick_top_expanded

def set_y_ticks_and_limits(ax, y_min, y_max, step=0.1, min_spacing=0.2, lower_tick_above_min=False, pin_lower_limit=False):
    """Set y-axis ticks and limits based on data extremes.

    Ticks are rounded INWARDS from data extremes (ceil for lower, floor for upper),
    with at least min_spacing between them. Limits are set slightly outside the ticks.
    """
    tick_bottom, tick_top, tick_top_expanded = compute_two_y_ticks(
        y_min, y_max, step, min_spacing, lower_tick_above_min=lower_tick_above_min,
    )

    ax.yaxis.set_major_locator(FixedLocator([tick_bottom, tick_top]))
    ax.tick_params(axis='y', labelsize=AXIS_LABEL_FONTSIZE)
    # Set limits slightly outside the ticks, ensuring room above the data max
    if pin_lower_limit:
        y_min_limit = tick_bottom
    else:
        y_min_limit = y_min - 0.05 * (y_max - y_min)
    y_max_limit = tick_top if tick_top_expanded else max(tick_top + 0.05 * (y_max - y_min_limit), y_max + 0.05 * (y_max - y_min_limit))
    ax.set_ylim([y_min_limit, y_max_limit])

def plot_rq1_performance(split, adaptation_mode):
    train_percents = [5, 50, 100]
    tags = ['chi_41', 'chi_42', 'chi_43']
    seeds = [41, 42, 43]

    # Load runs for each seed
    # Load all runs with all tags in a single API call
    all_runs_list = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': tags}})
    # Filter by tag in memory
    all_runs = {tag: [r for r in all_runs_list if tag in r.tags] for tag in tags}

    # Collect data for all tasks, aggregating over seeds
    all_data = []

    for task in tasks:
        metric = 'R2' if task != 'species' else 'mAP'
        metric_name = f'{split} test {metric}'

        for architecture in architectures_plots:
            for train_percent in train_percents:
                # Collect metrics from all seeds
                seed_metrics = []

                for tag, seed in zip(tags, seeds):
                    runs = all_runs[tag]
                    name = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
                    run = next((run for run in runs if run.name.startswith(name)), None)
                    test_metric = run.summary_metrics.get(metric_name) if run else None

                    if test_metric is not None and not np.isnan(test_metric):
                        seed_metrics.append(test_metric)

                # Average over seeds if we have any valid metrics
                if seed_metrics:
                    avg_metric = np.mean(seed_metrics)
                    # Calculate standard error across seeds
                    std_error = np.std(seed_metrics, ddof=1) / np.sqrt(len(seed_metrics)) if len(seed_metrics) > 1 else 0
                    all_data.append({'task': task, 'architecture': architecture, 'train_percent': train_percent, 'metric': avg_metric, 'std_error': std_error})

    df = pd.DataFrame(all_data) # converts the data to a DataFrame
    fig, axes = plt.subplots(1, 5, figsize=(COL_WIDTH, 1.5), gridspec_kw=dict(left=0.07, right=0.98, top=0.71, bottom=0.25, wspace=0.4))

    for i, task in enumerate(tasks):
        ax = axes[i] # gets the axis for the current task
        task_data = df[df['task'] == task]
        task_metrics = task_data['metric'].dropna()
        task_std_error = task_data['std_error'].dropna()
        # Account for mean ± SE in y-range
        y_min = (task_metrics - task_std_error).min() if not task_std_error.empty else task_metrics.min()
        y_max = (task_metrics + task_std_error).max() if not task_std_error.empty else task_metrics.max()

        # Plot each architecture
        for j, architecture in enumerate(architectures_plots):
            architecture_data = task_data[task_data['architecture'] == architecture]

            if not architecture_data.empty:
                architecture_data_sorted = architecture_data.sort_values('train_percent') # sort by train_percent to ensure proper line connection

                if architecture == 'ConvNeXtV2A':
                    color = 'black'
                    linestyle = '-'
                    marker = 'o'
                elif architecture in ['SatlasNet', 'MPMAE']:
                    color = ARCHITECTURE_COLORS[architecture]
                    linestyle = '--'
                    marker = 's'  # square
                elif architecture == 'ConvNeXtV2AMM':
                    color = ARCHITECTURE_COLORS[architecture]
                    linestyle = '-'
                    marker = '^'  # triangle up
                elif architecture in ['TerraMind', 'CopernicusFM', 'Galileo']:
                    color = ARCHITECTURE_COLORS[architecture]
                    linestyle = '--'
                    marker = '^'  # triangle up
                else:
                    color = ARCHITECTURE_COLORS[architecture]
                    linestyle = '--'
                    marker = 'o'  # circle

                # Plot shaded region for mean ± SE across seeds
                x_vals = architecture_data_sorted['train_percent'].values
                y_mean = architecture_data_sorted['metric'].values
                y_std_error = architecture_data_sorted['std_error'].values
                y_lower = y_mean - y_std_error
                y_upper = y_mean + y_std_error
                ax.fill_between(x_vals, y_lower, y_upper,
                               color=color, alpha=0.2, linewidth=0)

                # Plot the line and markers
                ax.plot(architecture_data_sorted['train_percent'], architecture_data_sorted['metric'],
                        marker, color=color, linestyle=linestyle, label=display_arch_name(architecture), markersize=MARKER_SIZE, linewidth=LINE_WIDTH, alpha=0.8)

        if i == 0:
            ax.set_ylabel('Performance', fontsize=AXIS_LABEL_FONTSIZE)
        elif i == 2:
            ax.set_xlabel('Training Data %', fontsize=AXIS_LABEL_FONTSIZE)

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)  # Adjust this value to change thickness (default is usually 1.0)

        for label in ax.get_yticklabels():
            label.set_ha('center')
            label.set_va('center')

        ax.tick_params(axis='y', rotation=90)
        ax.set_title(f"{task.replace('_', ' ').capitalize().replace('nitrogen', 'N').replace('organic carbon', 'OC').replace('ph', 'pH')}", fontsize=AXIS_LABEL_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(train_percents)
        ax.tick_params(axis='x', labelsize=AXIS_LABEL_FONTSIZE)
        set_y_ticks_and_limits(ax, y_min, y_max)

    handles, labels = axes[0].get_legend_handles_labels()

    # Create a mapping from display name to handle
    handle_map = {label: handle for handle, label in zip(handles, labels)}

    # Map architecture names to their display names
    arch_to_display = {arch: display_arch_name(arch) for arch in architectures_plots}

    # Define columns
    col1_archs = ['ConvNeXtV2A', 'ScaleMAE']
    col2_archs = ['DINOv3Web', 'DINOv3Sat']
    col3_archs = ['SatlasNet', 'MPMAE']
    col4_archs = ['TerraMind', 'CopernicusFM']
    col5_archs = ['Galileo', 'ConvNeXtV2AMM']

    # Build handles and labels for each column
    col1_handles = []
    col1_labels = []
    for arch in col1_archs:
        display_name = arch_to_display.get(arch, arch)
        if display_name in handle_map:
            col1_handles.append(handle_map[display_name])
            col1_labels.append(display_name)

    col2_handles = []
    col2_labels = []
    for arch in col2_archs:
        display_name = arch_to_display.get(arch, arch)
        if display_name in handle_map:
            col2_handles.append(handle_map[display_name])
            col2_labels.append(display_name)

    col3_handles = []
    col3_labels = []
    for arch in col3_archs:
        display_name = arch_to_display.get(arch, arch)
        if display_name in handle_map:
            col3_handles.append(handle_map[display_name])
            col3_labels.append(display_name)

    col4_handles = []
    col4_labels = []
    for arch in col4_archs:
        display_name = arch_to_display.get(arch, arch)
        if display_name in handle_map:
            col4_handles.append(handle_map[display_name])
            col4_labels.append(display_name)

    col5_handles = []
    col5_labels = []
    for arch in col5_archs:
        display_name = arch_to_display.get(arch, arch)
        if display_name in handle_map:
            col5_handles.append(handle_map[display_name])
            col5_labels.append(display_name)

    # Create five separate legends positioned side by side above the plot
    # Evenly distribute the five columns across the figure width
    legend_y_position = 1.05

    legend1 = fig.legend(col1_handles, col1_labels,
                        loc='upper center',
                        bbox_to_anchor=(0.1, legend_y_position),
                        ncol=1,
                        handletextpad=0.1,
                        handlelength=1,
                        labelspacing=0.1,
                        fontsize=LEGEND_FONTSIZE,
                        frameon=False)
    fig.add_artist(legend1)  # Add immediately to prevent removal

    legend2 = fig.legend(col2_handles, col2_labels,
                        loc='upper center',
                        bbox_to_anchor=(0.29, legend_y_position),
                        ncol=1,
                        handletextpad=0.1,
                        handlelength=1,
                        labelspacing=0.1,
                        fontsize=LEGEND_FONTSIZE,
                        frameon=False)
    fig.add_artist(legend2)  # Add immediately to prevent removal

    legend3 = fig.legend(col3_handles, col3_labels,
                        loc='upper center',
                        bbox_to_anchor=(0.46, legend_y_position),
                        ncol=1,
                        handletextpad=0.1,
                        handlelength=1,
                        labelspacing=0.1,
                        fontsize=LEGEND_FONTSIZE,
                        frameon=False)
    fig.add_artist(legend3)  # Add immediately to prevent removal

    legend4 = fig.legend(col4_handles, col4_labels,
                        loc='upper center',
                        bbox_to_anchor=(0.64, legend_y_position),
                        ncol=1,
                        handletextpad=0.1,
                        handlelength=1,
                        labelspacing=0.1,
                        fontsize=LEGEND_FONTSIZE,
                        frameon=False)
    fig.add_artist(legend4)  # Add immediately to prevent removal

    legend5 = fig.legend(col5_handles, col5_labels,
                        loc='upper center',
                        bbox_to_anchor=(0.875, legend_y_position),
                        ncol=1,
                        handletextpad=0.1,
                        handlelength=1,
                        labelspacing=0.1,
                        fontsize=LEGEND_FONTSIZE,
                        frameon=False)
    fig.add_artist(legend5)  # Add immediately to prevent removal

    plt.savefig(f'results_figures/RQ1_{adaptation_mode}_{split}_plot.pdf', dpi=300)

def plot_rq2_performance(adaptation_mode):
    train_percent = 100  # Use full training data for RQ2
    tags = ['chi_41', 'chi_42', 'chi_43']
    seeds = [41, 42, 43]

    # Load runs for each seed
    # Load all runs with all tags in a single API call
    all_runs_list = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': tags}})
    # Filter by tag in memory
    all_runs = {tag: [r for r in all_runs_list if tag in r.tags] for tag in tags}

    # Collect data for all tasks, aggregating over seeds
    all_data = []

    for task in tasks:
        metric = 'R2' if task != 'species' else 'mAP'
        random_metric_name = f'Random test {metric}'
        geographic_metric_name = f'Geographic test {metric}'

        for architecture in architectures_plots:
            # Collect metrics from all seeds for Random split
            random_seed_metrics = []
            geographic_seed_metrics = []
            for tag, seed in zip(tags, seeds):
                runs = all_runs[tag]
                name = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
                run = next((run for run in runs if run.name.startswith(name)), None)
                random_test_metric = run.summary_metrics.get(random_metric_name) if run else None
                geographic_test_metric = run.summary_metrics.get(geographic_metric_name) if run else None
                if random_test_metric is not None and not np.isnan(random_test_metric):
                    random_seed_metrics.append(random_test_metric)
                if geographic_test_metric is not None and not np.isnan(geographic_test_metric):
                    geographic_seed_metrics.append(geographic_test_metric)

            # Calculate mean and standard error for Random split
            if random_seed_metrics:
                random_avg = np.mean(random_seed_metrics)
                random_std_error = np.std(random_seed_metrics, ddof=1) / np.sqrt(len(random_seed_metrics)) if len(random_seed_metrics) > 1 else 0
                all_data.append({'task': task, 'architecture': architecture, 'split': 'Random', 'metric': random_avg, 'std_error': random_std_error})

            # Calculate mean and standard error for Geographic split
            if geographic_seed_metrics:
                geographic_avg = np.mean(geographic_seed_metrics)
                geographic_std_error = np.std(geographic_seed_metrics, ddof=1) / np.sqrt(len(geographic_seed_metrics)) if len(geographic_seed_metrics) > 1 else 0
                all_data.append({'task': task, 'architecture': architecture, 'split': 'Geographic', 'metric': geographic_avg, 'std_error': geographic_std_error})

    df = pd.DataFrame(all_data)

    # 1 row x 5 columns
    fig, axes = plt.subplots(1, 5, figsize=(COL_WIDTH, 1.1), gridspec_kw=dict(left=0.07, right=0.99, top=0.83, bottom=0.34, wspace=0.38))

    for i, task in enumerate(tasks):
        ax = axes[i]
        task_data = df[df['task'] == task]

        # y-range for this task, accounting for mean ± SE
        task_metrics = task_data['metric'].dropna()
        task_std_error = task_data['std_error'].dropna()
        if len(task_metrics) > 0:
            # Account for mean ± SE in y-range
            y_min = (task_metrics - task_std_error).min() if not task_std_error.empty else task_metrics.min()
            y_max = (task_metrics + task_std_error).max() if not task_std_error.empty else task_metrics.max()
        else:
            y_min = 0
            y_max = 1

        # Plot each architecture
        for j, architecture in enumerate(architectures_plots):
            architecture_data = task_data[task_data['architecture'] == architecture]
            if architecture_data.empty:
                continue
            random_data = architecture_data[architecture_data['split'] == 'Random']
            geographic_data = architecture_data[architecture_data['split'] == 'Geographic']

            if architecture == 'ConvNeXtV2A':
                color = 'black'; linestyle = '-'; marker = 'o'
            elif architecture in ['SatlasNet', 'MPMAE']:
                color = ARCHITECTURE_COLORS[architecture]; linestyle = '--'; marker = 's'
            elif architecture == 'ConvNeXtV2AMM':
                color = ARCHITECTURE_COLORS[architecture]; linestyle = '-'; marker = '^'
            elif architecture in ['TerraMind', 'CopernicusFM', 'Galileo']:
                color = ARCHITECTURE_COLORS[architecture]; linestyle = '--'; marker = '^'
            else:
                color = ARCHITECTURE_COLORS[architecture]; linestyle = '--'; marker = 'o'

            rv = random_data['metric'].iloc[0] if not random_data.empty else None
            gv = geographic_data['metric'].iloc[0] if not geographic_data.empty else None
            rv_std_error = random_data['std_error'].iloc[0] if not random_data.empty and 'std_error' in random_data.columns else 0
            gv_std_error = geographic_data['std_error'].iloc[0] if not geographic_data.empty and 'std_error' in geographic_data.columns else 0

            # Plot shaded region for mean ± SE across seeds
            if rv is not None and gv is not None:
                x_vals = np.array([0, 1])
                y_lower = np.array([rv - rv_std_error, gv - gv_std_error])
                y_upper = np.array([rv + rv_std_error, gv + gv_std_error])
                ax.fill_between(x_vals, y_lower, y_upper,
                               color=color, alpha=0.2, linewidth=0)

            # Plot line connecting the two points
            ax.plot([0, 1], [rv, gv], color=color, linestyle=linestyle, linewidth=LINE_WIDTH, alpha=0.8)

            # Plot points
            if rv is not None:
                ax.plot(0, rv, marker, color=color, linestyle=linestyle, markersize=MARKER_SIZE,
                       linewidth=LINE_WIDTH, alpha=0.8,
                       label=display_arch_name(architecture) if i == 0 else "")
            if gv is not None:
                ax.plot(1, gv, marker, color=color, linestyle=linestyle, markersize=MARKER_SIZE,
                       linewidth=LINE_WIDTH, alpha=0.8)

        if i == 0:
            ax.set_ylabel('Performance', fontsize=AXIS_LABEL_FONTSIZE)

        if i == 2:
            ax.set_xlabel('Test Split', fontsize=AXIS_LABEL_FONTSIZE)

        ax.tick_params(axis='y', rotation=90)

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)  # Adjust this value to change thickness (default is usually 1.0)

        for label in ax.get_yticklabels():
            label.set_ha('center')
            label.set_va('center')

        ax.set_title(f"{task.replace('_', ' ').capitalize().replace('nitrogen', 'N').replace('organic carbon', 'OC').replace('ph', 'pH')}", fontsize=AXIS_LABEL_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['R', 'G'])
        ax.tick_params(axis='x', labelsize=AXIS_LABEL_FONTSIZE)
        set_y_ticks_and_limits(ax, y_min, y_max)

    plt.savefig(f'results_figures/RQ2_{adaptation_mode}_plot.pdf', dpi=300)

def plot_rq3_performance(adaptation_mode):
    """Create an RQ3 plot with Random and Geographic splits in 2 rows x 5 columns (tasks).
    S2 is solid with circle markers; Multimodal is dashed with triangle markers.
    Legend under the plot; task names above each column.
    """
    train_percents = [5, 50, 100]
    splits = ['Random', 'Geographic']
    base_archs = ['TerraMind', 'CopernicusFM', 'Galileo']
    variants = {
        'S2': {'suffix': 'S2', 'linestyle': ':', 'marker': 's', 'label_suffix': 'S2'},
        'Multimodal': {'suffix': '', 'linestyle': '--', 'marker': '^', 'label_suffix': ''}
    }
    tags = ['chi_41', 'chi_42', 'chi_43']
    seeds = [41, 42, 43]

    # Load runs for each seed
    # Load all runs with all tags in a single API call
    all_runs_list = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': tags}})
    # Filter by tag in memory
    all_runs = {tag: [r for r in all_runs_list if tag in r.tags] for tag in tags}

    # Collect data for both splits, aggregating over seeds
    rows = []
    for task in tasks:
        metric = 'R2' if task != 'species' else 'mAP'
        for split in splits:
            metric_name = f"{split} test {metric}"
            for base in base_archs:
                for variant_name, cfg in variants.items():
                    arch_lookup = base + cfg['suffix']
                    for tp in train_percents:
                        # Collect metrics from all seeds
                        seed_metrics = []
                        for tag, seed in zip(tags, seeds):
                            runs = all_runs[tag]
                            name = '_'.join([task, arch_lookup, adaptation_mode, str(tp)]) + '_'
                            run = next((run for run in runs if run.name.startswith(name)), None)
                            val = run.summary_metrics.get(metric_name) if run else None
                            if val is not None and not np.isnan(val):
                                seed_metrics.append(val)

                        # Calculate mean and standard error if we have any valid metrics
                        if seed_metrics:
                            avg_metric = np.mean(seed_metrics)
                            std_error = np.std(seed_metrics, ddof=1) / np.sqrt(len(seed_metrics)) if len(seed_metrics) > 1 else 0
                            rows.append({
                                'task': task,
                                'split': split,
                                'base': base,
                                'variant': variant_name,
                                'train_percent': tp,
                                'metric': avg_metric,
                                'std_error': std_error
                            })

    df = pd.DataFrame(rows)

    # Figure: 2 rows x 5 columns (splits x tasks); leave right margin for legend
    fig, axes = plt.subplots(2, 5, figsize=(COL_WIDTH, 2), gridspec_kw=dict(left=0.07, right=0.7, top=0.89, bottom=0.20, wspace=0.4, hspace=0.3))

    for i, task in enumerate(tasks):
        for j, split in enumerate(splits):
            ax = axes[j, i]
            task_split_df = df[(df['task'] == task) & (df['split'] == split)]

            # y-range per panel, accounting for mean ± SE
            task_metrics = task_split_df['metric'].dropna()
            task_std_error = task_split_df['std_error'].dropna()
            if not task_metrics.empty:
                # Account for mean ± SE in y-range
                y_min = (task_metrics - task_std_error).min() if not task_std_error.empty else task_metrics.min()
                y_max = (task_metrics + task_std_error).max() if not task_std_error.empty else task_metrics.max()
            else:
                y_min = 0
                y_max = 1

            for base in base_archs:
                color = ARCHITECTURE_COLORS[base]
                for variant_name, cfg in variants.items():
                    sub = task_split_df[(task_split_df['base'] == base) & (task_split_df['variant'] == variant_name)]
                    if sub.empty:
                        continue
                    sub = sub.sort_values('train_percent')

                    # Plot shaded region for mean ± SE across seeds
                    x_vals = sub['train_percent'].values
                    y_mean = sub['metric'].values
                    y_std_error = sub['std_error'].values
                    y_lower = y_mean - y_std_error
                    y_upper = y_mean + y_std_error
                    ax.fill_between(x_vals, y_lower, y_upper,
                                   color=color, alpha=0.2, linewidth=0)

                    # Plot the line and markers
                    ax.plot(sub['train_percent'], sub['metric'],
                            linestyle=cfg['linestyle'], color=color, marker=cfg['marker'],
                            label=(f"{display_arch_name(base)} {cfg['label_suffix']}").strip() if i == 0 and j == 0 else None,
                            linewidth=LINE_WIDTH, markersize=MARKER_SIZE, alpha=0.9)

            # Axis cosmetics
            if i == 2 and j == 1:
                ax.set_xlabel('Training Data %', fontsize=AXIS_LABEL_FONTSIZE)

            # Set title only on top row
            if j == 0:
                ax.set_title(f"{task.replace('_', ' ').capitalize().replace('nitrogen', 'N').replace('organic carbon', 'OC').replace('ph', 'pH')}", fontsize=AXIS_LABEL_FONTSIZE)

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

            for label in ax.get_yticklabels():
                label.set_ha('center')
                label.set_va('center')

            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='y', rotation=90)
            ax.set_xticks(train_percents)
            ax.tick_params(axis='x', labelsize=AXIS_LABEL_FONTSIZE)
            set_y_ticks_and_limits(ax, y_min, y_max, min_spacing=0.1)

            # Remove x-axis ticks and labels on top row
            if j == 0:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', bottom=False, top=False)

    # Add "Performance" label in the middle between the two rows
    top_row_bottom = axes[0, 0].get_position().y0
    bottom_row_top = axes[1, 0].get_position().y1
    center_y = (top_row_bottom + bottom_row_top) / 2
    fig.text(0.02, center_y, 'Performance', fontsize=AXIS_LABEL_FONTSIZE, rotation=90, ha='center', va='center')

    # Add "Random" and "Geographic" labels to the right of the subplots
    top_row_center = (axes[0, 0].get_position().y0 + axes[0, 0].get_position().y1) / 2
    bottom_row_center = (axes[1, 0].get_position().y0 + axes[1, 0].get_position().y1) / 2
    fig.text(0.72, top_row_center, 'Random', fontsize=AXIS_LABEL_FONTSIZE, rotation=270, ha='center', va='center')
    fig.text(0.72, bottom_row_center, 'Geographic', fontsize=AXIS_LABEL_FONTSIZE, rotation=270, ha='center', va='center')

    # Legend under the plot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='center left',
               bbox_to_anchor=(0.72, 0.5),
               ncol=1,
               handletextpad=0.3,
               handlelength=1.3,
               labelspacing=0.1,
               fontsize=LEGEND_FONTSIZE,
               frameon=False)

    plt.savefig(f'results_figures/RQ3_{adaptation_mode}_plot.pdf', dpi=300)

def plot_ttt_improvement():
    """Create a combined TTT improvement plot with Random and Geographic splits side-by-side for each task.
    1 row x 5 columns (tasks).
    Each subplot shows 4 boxplots: Random (JT-TTT, JT-TTT-Geo) and Geographic (JT-TTT, JT-TTT-Geo).
    """

    splits = ['Random', 'Geographic']
    tags = ['chi_41', 'chi_42', 'chi_43']
    seeds = [41, 42, 43]

    # Load runs for each tag
    all_runs_list = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': tags}})
    all_runs = {tag: [r for r in all_runs_list if tag in r.tags] for tag in tags}

    # Collect improvements per seed
    improvements_per_seed = {seed: {task: {split: {mode: [] for mode in ['JT-TTT', 'JT-TTT-Geo']}
                                           for split in splits} for task in tasks} for seed in seeds}

    for tag, seed in zip(tags, seeds):
        runs = all_runs[tag]

        for task in tasks:
            metric = 'R2' if task != 'species' else 'mAP'

            for split in splits:
                metric_name = f'{split} test {metric}'

                # Baseline JT per architecture for this seed
                jt_baseline = {}
                for architecture in architectures_plots:
                    run_name = '_'.join([task, architecture, 'JT', str(100)]) + '_'
                    run = next((run for run in runs if run.name.startswith(run_name)), None)
                    if run:
                        jt_baseline[architecture] = run.summary_metrics.get(metric_name)

                # Improvements for JT-TTT and JT-TTT-Geo for this seed
                for mode in ['JT-TTT', 'JT-TTT-Geo']:
                    for architecture in architectures_plots:
                        run_name = '_'.join([task, architecture, mode, str(100)]) + '_'
                        run = next((run for run in runs if run.name.startswith(run_name)), None)

                        if not run:
                            continue

                        performance = run.summary_metrics.get(metric_name)
                        base = jt_baseline.get(architecture)

                        if base is not None and performance is not None and not np.isnan(base) and not np.isnan(performance):
                            improvement = performance - base
                            if not np.isnan(improvement):
                                improvements_per_seed[seed][task][split][mode].append(improvement)

    # Compute boxplot statistics per seed, then average them
    averaged_stats = {}

    for task in tasks:
        averaged_stats[task] = {}
        for split in splits:
            averaged_stats[task][split] = {}
            for mode in ['JT-TTT', 'JT-TTT-Geo']:
                seed_stats = []
                for seed in seeds:
                    improvements = improvements_per_seed[seed][task][split][mode]
                    if len(improvements) > 0:
                        improvements_array = np.array(improvements)
                        q1 = np.percentile(improvements_array, 25)
                        median = np.percentile(improvements_array, 50)
                        q3 = np.percentile(improvements_array, 75)
                        mean = np.mean(improvements_array)
                        iqr = q3 - q1
                        whislo = q1 - 1.5 * iqr
                        whishi = q3 + 1.5 * iqr
                        whislo = max(whislo, np.min(improvements_array))
                        whishi = min(whishi, np.max(improvements_array))
                        seed_stats.append({
                            'q1': q1, 'med': median, 'q3': q3, 'mean': mean,
                            'whislo': whislo, 'whishi': whishi
                        })

                if len(seed_stats) > 0:
                    averaged_stats[task][split][mode] = {
                        'q1': np.mean([s['q1'] for s in seed_stats]),
                        'med': np.mean([s['med'] for s in seed_stats]),
                        'q3': np.mean([s['q3'] for s in seed_stats]),
                        'mean': np.mean([s['mean'] for s in seed_stats]),
                        'whislo': np.mean([s['whislo'] for s in seed_stats]),
                        'whishi': np.mean([s['whishi'] for s in seed_stats])
                    }
                else:
                    averaged_stats[task][split][mode] = None

    # 1 row x 5 columns (tasks)
    fig, axes = plt.subplots(
            1, 5,
            figsize=(COL_WIDTH, 1.4),
            gridspec_kw=dict(wspace=0.35, left=0.07, right=0.99, top=0.87, bottom=0.22)
        )

    colors = ['#1f77b4', '#ff7f0e']  # Blue for TTT-MMR, Orange for TTT-MMR-Geo

    for i, task in enumerate(tasks):
        ax = axes[i]

        # We'll plot 4 boxes: Random(JT-TTT), Random(JT-TTT-Geo), Geo(JT-TTT), Geo(JT-TTT-Geo)
        # Positions: 1, 2 (small gap) 3, 4

        stats_list = []
        positions = []
        mode_colors = []

        # Random split
        split = 'Random'
        base_pos = 1
        for j, (mode, color) in enumerate(zip(['JT-TTT', 'JT-TTT-Geo'], colors)):
            stats = averaged_stats[task][split][mode]
            if stats is not None:
                stats_list.append(stats)
                positions.append(base_pos + j)
                mode_colors.append(color)

        # Geographic split
        split = 'Geographic'
        base_pos = 3
        for j, (mode, color) in enumerate(zip(['JT-TTT', 'JT-TTT-Geo'], colors)):
            stats = averaged_stats[task][split][mode]
            if stats is not None:
                stats_list.append(stats)
                positions.append(base_pos + j)
                mode_colors.append(color)

        if len(stats_list) > 0:
            bp = ax.bxp(stats_list,
                        positions=positions,
                        widths=0.6,
                        patch_artist=True,
                        showmeans=True,
                        showfliers=False,
                        meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=MARKER_SIZE),
                        medianprops=dict(color='black', linewidth=LINE_WIDTH),
                        boxprops=dict(linewidth=LINE_WIDTH),
                        whiskerprops=dict(linewidth=LINE_WIDTH),
                        capprops=dict(linewidth=LINE_WIDTH))

            for patch, color in zip(bp['boxes'], mode_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor(color)
                patch.set_linewidth(0.5)

            # Color whiskers and caps (2 each per box) to match the fill color
            for k, color in enumerate(mode_colors):
                for elem in bp['whiskers'][2 * k:2 * k + 2]:
                    elem.set_color(color)
                for elem in bp['caps'][2 * k:2 * k + 2]:
                    elem.set_color(color)

        ax.axhline(y=0, color='black', linewidth=LINE_WIDTH)

        # X-axis labels
        ax.set_xticks([1.5, 3.5])
        ax.set_xticklabels(['R', 'G'], fontsize=AXIS_LABEL_FONTSIZE)

        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='y', labelsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis='y', rotation=90)

        # Smart y-ticks
        ymin, ymax = ax.get_ylim()
        rounded_ymin = np.round(ymin / 0.01) * 0.01
        rounded_ymax = np.round(ymax / 0.01) * 0.01
        ax.set_yticks([rounded_ymin, rounded_ymax])

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        for label in ax.get_yticklabels():
            label.set_ha('center')
            label.set_va('center')

        ax.set_title(task.replace('_', ' ').capitalize().replace('nitrogen', 'N').replace('organic carbon', 'OC').replace('ph', 'pH'), fontsize=AXIS_LABEL_FONTSIZE)

        if i == 0:
             ax.set_ylabel('Δ Performance', fontsize=AXIS_LABEL_FONTSIZE)

    # Legend
    legend_labels = ['TTT-MMR', 'TTT-MMR-Geo']
    legend_handles = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7) for color in colors]
    fig.legend(legend_handles, legend_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.07),
            ncol=len(legend_labels),
            fontsize=LEGEND_FONTSIZE,
            frameon=False)

    plt.savefig('results_figures/TTT_plot.pdf', dpi=300)

def plot_ttt_improvement_normalized():
    """Create a combined TTT improvement plot with Random and Geographic splits side-by-side for each task.
    1 row x 5 columns (tasks).
    Each subplot shows 4 boxplots: Random (JT-TTT, JT-TTT-Geo) and Geographic (JT-TTT, JT-TTT-Geo).
    Improvement is calculated as normalized: (r2_new - r2_old) / (1 - r2_old)
    Statistics are computed separately for each seed, then averaged across seeds.
    """

    splits = ['Random', 'Geographic']
    tags = ['chi_41', 'chi_42', 'chi_43']
    seeds = [41, 42, 43]

    # Load runs for each tag
    all_runs_list = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': tags}})
    all_runs = {tag: [r for r in all_runs_list if tag in r.tags] for tag in tags}

    # Collect improvements per seed
    improvements_per_seed = {seed: {task: {split: {mode: [] for mode in ['JT-TTT', 'JT-TTT-Geo']}
                                           for split in splits} for task in tasks} for seed in seeds}

    for tag, seed in zip(tags, seeds):
        runs = all_runs[tag]

        for task in tasks:
            metric = 'R2' if task != 'species' else 'mAP'

            for split in splits:
                metric_name = f'{split} test {metric}'

                # Baseline JT per architecture for this seed
                jt_baseline = {}
                for architecture in architectures_plots:
                    run_name = '_'.join([task, architecture, 'JT', str(100)]) + '_'
                    run = next((run for run in runs if run.name.startswith(run_name)), None)
                    if run:
                        jt_baseline[architecture] = run.summary_metrics.get(metric_name)

                # Improvements for JT-TTT and JT-TTT-Geo for this seed
                for mode in ['JT-TTT', 'JT-TTT-Geo']:
                    for architecture in architectures_plots:
                        run_name = '_'.join([task, architecture, mode, str(100)]) + '_'
                        run = next((run for run in runs if run.name.startswith(run_name)), None)

                        if not run:
                            continue

                        performance = run.summary_metrics.get(metric_name)
                        base = jt_baseline.get(architecture)

                        # Normalized improvement: (r2_new - r2_old) / (1 - r2_old)
                        if base is not None and performance is not None and not np.isnan(base) and not np.isnan(performance):
                            if base != 1:
                                improvement = (performance - base) / (1 - base)
                                if not np.isnan(improvement):
                                    improvements_per_seed[seed][task][split][mode].append(improvement)

    # Compute boxplot statistics per seed, then average them
    averaged_stats = {}

    for task in tasks:
        averaged_stats[task] = {}
        for split in splits:
            averaged_stats[task][split] = {}
            for mode in ['JT-TTT', 'JT-TTT-Geo']:
                seed_stats = []
                for seed in seeds:
                    improvements = improvements_per_seed[seed][task][split][mode]
                    if len(improvements) > 0:
                        improvements_array = np.array(improvements)
                        q1 = np.percentile(improvements_array, 25)
                        median = np.percentile(improvements_array, 50)
                        q3 = np.percentile(improvements_array, 75)
                        mean = np.mean(improvements_array)
                        iqr = q3 - q1
                        whislo = q1 - 1.5 * iqr
                        whishi = q3 + 1.5 * iqr
                        whislo = max(whislo, np.min(improvements_array))
                        whishi = min(whishi, np.max(improvements_array))
                        seed_stats.append({
                            'q1': q1,
                            'med': median,
                            'q3': q3,
                            'mean': mean,
                            'whislo': whislo,
                            'whishi': whishi
                        })

                if len(seed_stats) > 0:
                    averaged_stats[task][split][mode] = {
                        'q1': np.mean([s['q1'] for s in seed_stats]),
                        'med': np.mean([s['med'] for s in seed_stats]),
                        'q3': np.mean([s['q3'] for s in seed_stats]),
                        'mean': np.mean([s['mean'] for s in seed_stats]),
                        'whislo': np.mean([s['whislo'] for s in seed_stats]),
                        'whishi': np.mean([s['whishi'] for s in seed_stats])
                    }
                else:
                    averaged_stats[task][split][mode] = None

    # 1 row x 5 columns (tasks)
    fig, axes = plt.subplots(
            1, 5,
            figsize=(COL_WIDTH, 1.4),
            gridspec_kw=dict(wspace=0.35, left=0.07, right=0.99, top=0.87, bottom=0.22)
        )

    colors = ['#1f77b4', '#ff7f0e']  # Blue for TTT-MMR, Orange for TTT-MMR-Geo

    for i, task in enumerate(tasks):
        ax = axes[i]

        # We'll plot 4 boxes: Random(JT-TTT), Random(JT-TTT-Geo), Geo(JT-TTT), Geo(JT-TTT-Geo)
        # Positions: 1, 2 (small gap) 3, 4

        stats_list = []
        positions = []
        mode_colors = []

        # Random split
        split = 'Random'
        base_pos = 1
        for j, (mode, color) in enumerate(zip(['JT-TTT', 'JT-TTT-Geo'], colors)):
            stats = averaged_stats[task][split][mode]
            if stats is not None:
                stats_list.append({k: v * 100 for k, v in stats.items()})
                positions.append(base_pos + j)
                mode_colors.append(color)

        # Geographic split
        split = 'Geographic'
        base_pos = 3
        for j, (mode, color) in enumerate(zip(['JT-TTT', 'JT-TTT-Geo'], colors)):
            stats = averaged_stats[task][split][mode]
            if stats is not None:
                stats_list.append({k: v * 100 for k, v in stats.items()})
                positions.append(base_pos + j)
                mode_colors.append(color)

        if len(stats_list) > 0:
            bp = ax.bxp(stats_list,
                        positions=positions,
                        widths=0.6,
                        patch_artist=True,
                        showmeans=True,
                        showfliers=False,
                        meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=MARKER_SIZE),
                        medianprops=dict(color='black', linewidth=LINE_WIDTH),
                        boxprops=dict(linewidth=LINE_WIDTH),
                        whiskerprops=dict(linewidth=LINE_WIDTH),
                        capprops=dict(linewidth=LINE_WIDTH))

            for patch, color in zip(bp['boxes'], mode_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor(color)
                patch.set_linewidth(0.5)

            # Color whiskers and caps (2 each per box) to match the fill color
            for k, color in enumerate(mode_colors):
                for elem in bp['whiskers'][2 * k:2 * k + 2]:
                    elem.set_color(color)
                for elem in bp['caps'][2 * k:2 * k + 2]:
                    elem.set_color(color)

        ax.axhline(y=0, color='black', linewidth=LINE_WIDTH)

        # X-axis labels
        ax.set_xticks([1.5, 3.5])
        ax.set_xticklabels(['R', 'G'], fontsize=AXIS_LABEL_FONTSIZE)

        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='y', labelsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis='y', rotation=90)

        # Smart y-ticks
        ymin, ymax = ax.get_ylim()
        rounded_ymin = np.round(ymin / 1) * 1
        rounded_ymax = np.round(ymax / 1) * 1
        ax.set_yticks([rounded_ymin, rounded_ymax])

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        for label in ax.get_yticklabels():
            label.set_ha('center')
            label.set_va('center')

        ax.set_title(task.replace('_', ' ').capitalize().replace('nitrogen', 'N').replace('organic carbon', 'OC').replace('ph', 'pH'), fontsize=AXIS_LABEL_FONTSIZE)

        if i == 0:
             ax.set_ylabel('RI (%)', fontsize=AXIS_LABEL_FONTSIZE)

    # Legend
    legend_labels = ['TTT-MMR', 'TTT-MMR-Geo']
    legend_handles = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7) for color in colors]
    fig.legend(legend_handles, legend_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.07),
            ncol=len(legend_labels),
            fontsize=LEGEND_FONTSIZE,
            frameon=False)

    plt.savefig('results_figures/TTT_plot_normalized.pdf', dpi=300)

def tabulate_ttt_by_model():
    """Create a table showing average improvement (normalized, averaged over tasks) of TTT-MMR and TTT-MMR-Geo over JT.
    Rows are methods (TTT-MMR and TTT-MMR-Geo) for Random, then for Geographic (4 rows total).
    Columns are architectures. Includes standard error.
    Separate results for Random and Geographic test splits.
    Improvement is calculated as normalized: (r2_new - r2_old) / (1 - r2_old)
    First average improvements over tasks for each seed, then average those averages over seeds.
    """
    adaptation_modes = ['JT-TTT', 'JT-TTT-Geo']
    splits = ['Random', 'Geographic']
    tags = ['chi_41', 'chi_42', 'chi_43']
    seeds = [41, 42, 43]

    # Load runs for each tag
    # Load all runs with all tags in a single API call
    all_runs_list = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': tags}})
    # Filter by tag in memory
    all_runs = {tag: [r for r in all_runs_list if tag in r.tags] for tag in tags}

    # Collect improvements per seed and task: {architecture: {split: {seed: {task: {mode: improvement}}}}}
    improvement_data = {architecture: {split: {seed: {task: {mode: None for mode in adaptation_modes} for task in tasks} for seed in seeds} for split in splits}
                       for architecture in architectures_plots}

    for architecture in architectures_plots:
        for task in tasks:
            metric = 'R2' if task != 'species' else 'mAP'

            # Get JT baseline performance for each split and seed
            for split in splits:
                metric_name = f'{split} test {metric}'

                for tag, seed in zip(tags, seeds):
                    runs = all_runs[tag]

                    # Get JT baseline for this seed
                    jt_run_name = '_'.join([task, architecture, 'JT', str(100)]) + '_'
                    jt_run = next((run for run in runs if run.name.startswith(jt_run_name)), None)
                    baseline = None
                    if jt_run:
                        baseline_value = jt_run.summary_metrics.get(metric_name)
                        if baseline_value is not None and not np.isnan(baseline_value):
                            baseline = baseline_value

                    if baseline is not None:
                        # Calculate improvements for JT-TTT and JT-TTT-Geo for this seed
                        for mode in adaptation_modes:
                            run_name = '_'.join([task, architecture, mode, str(100)]) + '_'
                            run = next((run for run in runs if run.name.startswith(run_name)), None)

                            if run:
                                performance = run.summary_metrics.get(metric_name)
                                if performance is not None and not np.isnan(performance) and not np.isnan(baseline):
                                    # Calculate normalized improvement: (r2_new - r2_old) / (1 - r2_old)
                                    # Handle edge cases: if baseline is 1, avoid division by zero
                                    if baseline != 1:
                                        improvement = (performance - baseline) / (1 - baseline)
                                    else:
                                        improvement = np.nan  # Skip if baseline is 1 (perfect R2)
                                    if not np.isnan(improvement):
                                        improvement_data[architecture][split][seed][task][mode] = improvement

    # First, average over tasks for each seed: {architecture: {split: {seed: {mode: avg_improvement}}}}
    seed_avg_improvements = {architecture: {split: {seed: {mode: None for mode in adaptation_modes} for seed in seeds} for split in splits}
                            for architecture in architectures_plots}

    for architecture in architectures_plots:
        for split in splits:
            for seed in seeds:
                for mode in adaptation_modes:
                    # Collect improvements for this seed across all tasks
                    task_improvements = []
                    for task in tasks:
                        improvement = improvement_data[architecture][split][seed][task][mode]
                        if improvement is not None:
                            task_improvements.append(improvement)
                    # Average over tasks for this seed
                    if len(task_improvements) > 0:
                        seed_avg_improvements[architecture][split][seed][mode] = np.mean(task_improvements)
                    else:
                        seed_avg_improvements[architecture][split][seed][mode] = None

    # Then, average over seeds and calculate mean and standard error
    data_dict = {}

    for split in splits:
        data_dict[split] = {}
        for mode in adaptation_modes:
            data_dict[split][mode] = {}
            for architecture in architectures_plots:
                # Collect seed-averaged improvements for the mean
                seed_avgs = []
                for seed in seeds:
                    avg_improvement = seed_avg_improvements[architecture][split][seed][mode]
                    if avg_improvement is not None:
                        seed_avgs.append(avg_improvement)

                # Collect all individual improvements (across tasks and seeds) for the SE
                all_improvements = []
                for seed in seeds:
                    for task in tasks:
                        improvement = improvement_data[architecture][split][seed][task][mode]
                        if improvement is not None:
                            all_improvements.append(improvement)

                if len(seed_avgs) > 0:
                    mean = np.mean(seed_avgs)  # Mean: average over tasks per seed, then over seeds
                    # SE: computed from all individual improvements to reflect task-level variation
                    se = np.std(all_improvements, ddof=1) / np.sqrt(len(all_improvements)) if len(all_improvements) > 1 else 0.0
                    data_dict[split][mode][architecture] = {'mean': mean, 'se': se, 'n': len(seed_avgs)}
                else:
                    data_dict[split][mode][architecture] = {'mean': np.nan, 'se': np.nan, 'n': 0}

    # Find the method with the highest mean for each split-architecture combination
    # This will be used to bold the best (highest improvement) value per column within each split
    best_methods = {}  # {(split, architecture): mode}
    for split in splits:
        for architecture in architectures_plots:
            best_mean = float('-inf')
            best_mode = None
            for mode in adaptation_modes:
                stats = data_dict[split][mode][architecture]
                if stats['n'] > 0 and not np.isnan(stats['mean']):
                    if stats['mean'] > best_mean:
                        best_mean = stats['mean']
                        best_mode = mode
            if best_mode is not None:
                best_methods[(split, architecture)] = best_mode

    # Create DataFrame with mean ± SE format
    # Rows: TTT-MMR, TTT-MMR-Geo (for Random), then TTT-MMR, TTT-MMR-Geo (for Geographic)
    # Columns: Split, Method, then architectures
    display_decimals = 1
    formatted_data = {}
    split_column = []
    method_column = []
    row_keys = []

    # Create row labels and data with unique keys
    display_name_mapping = {'JT-TTT': 'TTT-MMR', 'JT-TTT-Geo': 'TTT-MMR-Geo'}

    for split in splits:
        for mode in adaptation_modes:
            display_mode = display_name_mapping[mode]
            # Use unique key that includes split to avoid duplicates
            row_key = f"{split}_{display_mode}"
            row_keys.append(row_key)
            split_column.append(split)
            method_column.append(display_mode)
            formatted_data[row_key] = {}

            for architecture in architectures_plots:
                stats = data_dict[split][mode][architecture]
                if stats['n'] > 0 and not np.isnan(stats['mean']):
                    # Multiply by 100 to convert to percentage
                    mean_val = stats['mean'] * 100
                    se_val = stats['se'] * 100
                    mean_str = f"{mean_val:.{display_decimals}f}"
                    se_str = f"{se_val:.{display_decimals}f}"
                    # Bold if this is the best (highest mean) method for this split-architecture
                    if best_methods.get((split, architecture)) == mode:
                        formatted_data[row_key][architecture] = f"$\\mathbf{{{mean_str} \\pm {se_str}}}$"
                    else:
                        formatted_data[row_key][architecture] = f"${mean_str} \\pm {se_str}$"
                else:
                    formatted_data[row_key][architecture] = "--"

    # Create DataFrame
    df = pd.DataFrame(formatted_data).T
    df = df.reindex(row_keys)  # Ensure correct row order
    df = df.reindex(architectures_plots, axis=1)  # Ensure correct column order
    df.columns = [display_arch_name(arch) for arch in architectures_plots]

    # Insert Split and Method columns at the beginning
    df.insert(0, 'Method', method_column)
    df.insert(0, 'Split', split_column)

    # Create LaTeX table
    header_line = ' & '.join(['\\textbf{Split}', '\\textbf{Method}'] + [f'\\textbf{{{c}}}' for c in df.columns[2:]]) + r' \\'
    latex = df.to_latex(index=False,
                        header=False,
                        escape=False,
                        column_format='l' + 'l' + 'c' * (len(df.columns) - 2),
                        na_rep='--')

    # Insert custom header and use multicolumn for Split column
    lines = latex.split('\n')
    toprule_idx = next(i for i, line in enumerate(lines) if '\\toprule' in line)
    lines.insert(toprule_idx + 1, header_line)
    # Remove any existing midrule that might be right after toprule
    if toprule_idx + 2 < len(lines) and '\\midrule' in lines[toprule_idx + 2]:
        lines.pop(toprule_idx + 2)
    lines.insert(toprule_idx + 2, '\\midrule')

    # Add multicolumn for Split column to group rows
    # Find the data rows (after midrule)
    midrule_idx = next(i for i, line in enumerate(lines) if '\\midrule' in line and i > toprule_idx)
    # Replace Split values with multicolumn for first row of each split group
    row_count = 0
    geographic_start_idx = None
    for i in range(midrule_idx + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.strip().startswith('\\'):
            parts = line.split(' & ')
            if len(parts) >= 2:
                # First row of Random (row_count 0) and first row of Geographic (row_count 2)
                if row_count == 0:  # First Random row
                    parts[0] = f"\\multirow{{2}}{{*}}{{\\textbf{{Random}}}}"
                elif row_count == 2:  # First Geographic row
                    geographic_start_idx = i
                    parts[0] = f"\\multirow{{2}}{{*}}{{\\textbf{{Geographic}}}}"
                else:
                    parts[0] = ""  # Empty for subsequent rows in group
                lines[i] = ' & '.join(parts)
                row_count += 1

    # Insert horizontal line before Geographic section
    if geographic_start_idx is not None:
        # Check if there's already a midrule before Geographic (skip empty lines)
        prev_idx = geographic_start_idx - 1
        while prev_idx >= 0 and lines[prev_idx].strip() == '':
            prev_idx -= 1
        # Only insert if there isn't already a midrule right before
        if prev_idx < 0 or '\\midrule' not in lines[prev_idx]:
            lines.insert(geographic_start_idx, '\\midrule')

    latex = '\n'.join(lines)

    # Add table environment with resizebox and table*
    latex = ("\\begin{table*}[ht]\n\\centering\n" +
            "\\resizebox{\\linewidth}{!}{%\n" +
            latex +
            "\n}\n" +
            "\\caption{Average improvement over JT (averaged over tasks) for TTT-MMR and TTT-MMR-Geo by architecture and test split. Values shown as mean $\\pm$ standard error (multiplied by 100). Improvement is calculated as normalized: $(R^2_{\\text{new}} - R^2_{\\text{old}}) / (1 - R^2_{\\text{old}})$.}\n" +
            "\\label{tab:ttt_by_model}\n" +
            "\\end{table*}\n")

    tex_file = 'results_tex/ttt_by_model.tex'
    with open(tex_file, 'w') as file:
        file.write(latex)
    compile_latex(tex_file)

def tabulate_ttt_ranks_by_model():
    """Create a table showing average ranks of JT, TTT-MMR, and TTT-MMR-Geo by model.
    For each model, task, split, and seed, rank the three methods (1=best, 3=worst).
    First average ranks over tasks for each seed, then average those averages over seeds.
    Rows are methods (JT, TTT-MMR, TTT-MMR-Geo) for Random, then for Geographic (6 rows total).
    Columns are models. Shows mean ± SE of ranks.
    """
    adaptation_modes = ['JT', 'JT-TTT', 'JT-TTT-Geo']
    splits = ['Random', 'Geographic']
    tags = ['chi_41', 'chi_42', 'chi_43']
    seeds = [41, 42, 43]

    # Load runs for each tag
    # Load all runs with all tags in a single API call
    all_runs_list = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': tags}})
    # Filter by tag in memory
    all_runs = {tag: [r for r in all_runs_list if tag in r.tags] for tag in tags}

    # Collect ranks per seed and task: {architecture: {split: {seed: {task: {mode: rank}}}}}
    rank_data = {architecture: {split: {seed: {task: {mode: None for mode in adaptation_modes} for task in tasks} for seed in seeds} for split in splits}
                 for architecture in architectures_plots}

    for architecture in architectures_plots:
        for task in tasks:
            metric = 'R2' if task != 'species' else 'mAP'

            # Rank methods for each split and seed (1 = best, 3 = worst)
            for split in splits:
                metric_name = f'{split} test {metric}'

                for tag, seed in zip(tags, seeds):
                    runs = all_runs[tag]

                    # Collect performance for each method for this seed
                    method_perfs = {}
                    for mode in adaptation_modes:
                        run_name = '_'.join([task, architecture, mode, str(100)]) + '_'
                        run = next((run for run in runs if run.name.startswith(run_name)), None)
                        if run:
                            perf = run.summary_metrics.get(metric_name)
                            if perf is not None and not np.isnan(perf):
                                method_perfs[mode] = perf

                    # Rank methods for this seed if all three methods have data
                    if len(method_perfs) == 3:  # All three methods have data
                        # Sort by performance (descending, so best gets rank 1)
                        sorted_methods = sorted(method_perfs.items(), key=lambda x: x[1], reverse=True)

                        # Assign ranks (1 = best, 2 = middle, 3 = worst)
                        # Handle ties: if two methods have the same performance, they get the same rank
                        ranks = {}
                        current_rank = 1
                        prev_perf = None
                        for i, (mode, perf) in enumerate(sorted_methods):
                            if prev_perf is not None and abs(perf - prev_perf) < 1e-10:  # Tie
                                ranks[mode] = current_rank
                            else:
                                current_rank = i + 1
                                ranks[mode] = current_rank
                            prev_perf = perf

                        # Store ranks for this seed and task
                        for mode in adaptation_modes:
                            if mode in ranks:
                                rank_data[architecture][split][seed][task][mode] = ranks[mode]

    # First, average over tasks for each seed: {architecture: {split: {seed: {mode: avg_rank}}}}
    seed_avg_ranks = {architecture: {split: {seed: {mode: None for mode in adaptation_modes} for seed in seeds} for split in splits}
                      for architecture in architectures_plots}

    for architecture in architectures_plots:
        for split in splits:
            for seed in seeds:
                for mode in adaptation_modes:
                    # Collect ranks for this seed across all tasks
                    task_ranks = []
                    for task in tasks:
                        rank = rank_data[architecture][split][seed][task][mode]
                        if rank is not None:
                            task_ranks.append(rank)
                    # Average over tasks for this seed
                    if len(task_ranks) > 0:
                        seed_avg_ranks[architecture][split][seed][mode] = np.mean(task_ranks)
                    else:
                        seed_avg_ranks[architecture][split][seed][mode] = None

    # Then, average over seeds and calculate mean and standard error
    data_dict = {}
    display_name_mapping = {'JT': 'JT', 'JT-TTT': 'TTT-MMR', 'JT-TTT-Geo': 'TTT-MMR-Geo'}

    for split in splits:
        data_dict[split] = {}
        for mode in adaptation_modes:
            data_dict[split][mode] = {}
            for architecture in architectures_plots:
                # Collect seed-averaged ranks for the mean
                seed_avgs = []
                for seed in seeds:
                    avg_rank = seed_avg_ranks[architecture][split][seed][mode]
                    if avg_rank is not None:
                        seed_avgs.append(avg_rank)

                # Collect all individual ranks (across tasks and seeds) for the SE
                all_ranks = []
                for seed in seeds:
                    for task in tasks:
                        rank = rank_data[architecture][split][seed][task][mode]
                        if rank is not None:
                            all_ranks.append(rank)

                if len(seed_avgs) > 0:
                    mean = np.mean(seed_avgs)  # Mean: average over tasks per seed, then over seeds
                    # SE: computed from all individual ranks to reflect task-level variation
                    se = np.std(all_ranks, ddof=1) / np.sqrt(len(all_ranks)) if len(all_ranks) > 1 else 0.0
                    data_dict[split][mode][architecture] = {'mean': mean, 'se': se, 'n': len(seed_avgs)}
                else:
                    data_dict[split][mode][architecture] = {'mean': np.nan, 'se': np.nan, 'n': 0}

    # Create DataFrame with mean ± SE format
    # Rows: JT, TTT-MMR, TTT-MMR-Geo (for Random), then JT, TTT-MMR, TTT-MMR-Geo (for Geographic)
    # Columns: Split, Method, then architectures
    display_decimals = 1

    # Find the best (lowest) rounded mean for each split-architecture combination
    # This will be used to bold all values that round to the same value as the best
    best_rounded_means = {}  # {(split, architecture): best_rounded_mean}
    for split in splits:
        for architecture in architectures_plots:
            best_mean = float('inf')
            for mode in adaptation_modes:
                stats = data_dict[split][mode][architecture]
                if stats['n'] > 0 and not np.isnan(stats['mean']):
                    if stats['mean'] < best_mean:
                        best_mean = stats['mean']
            if best_mean != float('inf'):
                # Round to display_decimals to get the visual appearance
                best_rounded_means[(split, architecture)] = round(best_mean, display_decimals)
    formatted_data = {}
    split_column = []
    method_column = []
    row_keys = []

    # Create row labels and data with unique keys
    for split in splits:
        for mode in adaptation_modes:
            display_mode = display_name_mapping[mode]
            # Use unique key that includes split to avoid duplicates
            row_key = f"{split}_{display_mode}"
            row_keys.append(row_key)
            split_column.append(split)
            method_column.append(display_mode)
            formatted_data[row_key] = {}

            for architecture in architectures_plots:
                stats = data_dict[split][mode][architecture]
                if stats['n'] > 0 and not np.isnan(stats['mean']):
                    mean_str = f"{stats['mean']:.{display_decimals}f}"
                    se_str = f"{stats['se']:.{display_decimals}f}"
                    # Bold if this value rounds to the same as the best (lowest) rounded mean
                    rounded_mean = round(stats['mean'], display_decimals)
                    best_rounded = best_rounded_means.get((split, architecture))
                    if best_rounded is not None and rounded_mean == best_rounded:
                        formatted_data[row_key][architecture] = f"$\\mathbf{{{mean_str} \\pm {se_str}}}$"
                    else:
                        formatted_data[row_key][architecture] = f"${mean_str} \\pm {se_str}$"
                else:
                    formatted_data[row_key][architecture] = "--"

    # Create DataFrame
    df = pd.DataFrame(formatted_data).T
    df = df.reindex(row_keys)  # Ensure correct row order
    df = df.reindex(architectures_plots, axis=1)  # Ensure correct column order
    df.columns = [display_arch_name(arch) for arch in architectures_plots]

    # Insert Split and Method columns at the beginning
    df.insert(0, 'Method', method_column)
    df.insert(0, 'Split', split_column)

    # Create LaTeX table
    header_line = ' & '.join(['\\textbf{Split}', '\\textbf{Method}'] + [f'\\textbf{{{c}}}' for c in df.columns[2:]]) + r' \\'
    latex = df.to_latex(index=False,
                        header=False,
                        escape=False,
                        column_format='l' + 'l' + 'c' * (len(df.columns) - 2),
                        na_rep='--')

    # Insert custom header and use multicolumn for Split column
    lines = latex.split('\n')
    toprule_idx = next(i for i, line in enumerate(lines) if '\\toprule' in line)
    lines.insert(toprule_idx + 1, header_line)
    # Remove any existing midrule that might be right after toprule
    if toprule_idx + 2 < len(lines) and '\\midrule' in lines[toprule_idx + 2]:
        lines.pop(toprule_idx + 2)
    lines.insert(toprule_idx + 2, '\\midrule')

    # Add multicolumn for Split column to group rows
    # Find the data rows (after midrule)
    midrule_idx = next(i for i, line in enumerate(lines) if '\\midrule' in line and i > toprule_idx)
    # Replace Split values with multicolumn for first row of each split group
    row_count = 0
    geographic_start_idx = None
    for i in range(midrule_idx + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.strip().startswith('\\'):
            parts = line.split(' & ')
            if len(parts) >= 2:
                # First row of Random (row_count 0) and first row of Geographic (row_count 3)
                if row_count == 0:  # First Random row
                    parts[0] = f"\\multirow{{3}}{{*}}{{\\textbf{{Random}}}}"
                elif row_count == 3:  # First Geographic row
                    geographic_start_idx = i
                    parts[0] = f"\\multirow{{3}}{{*}}{{\\textbf{{Geographic}}}}"
                else:
                    parts[0] = ""  # Empty for subsequent rows in group
                lines[i] = ' & '.join(parts)
                row_count += 1

    # Insert horizontal line before Geographic section
    if geographic_start_idx is not None:
        # Check if there's already a midrule before Geographic (skip empty lines)
        prev_idx = geographic_start_idx - 1
        while prev_idx >= 0 and lines[prev_idx].strip() == '':
            prev_idx -= 1
        # Only insert if there isn't already a midrule right before
        if prev_idx < 0 or '\\midrule' not in lines[prev_idx]:
            lines.insert(geographic_start_idx, '\\midrule')

    latex = '\n'.join(lines)

    # Add table environment (table* spans two columns)
    latex = ("\\begin{table*}[ht]\n\\centering\n" +
            "\\resizebox{\\linewidth}{!}{%\n" +
            latex +
            "}\n" +
            "\\caption{Average ranks of JT, TTT-MMR, and TTT-MMR-Geo by architecture (first averaged over tasks for each seed, then averaged over seeds for each test split). Ranks: 1 = best, 3 = worst. Values shown as mean $\\pm$ standard error.}\n" +
            "\\label{tab:ttt_ranks_by_model}\n" +
            "\\end{table*}\n")

    tex_file = 'results_tex/ttt_ranks_by_model.tex'

    with open(tex_file, 'w') as file:
        file.write(latex)

    compile_latex(tex_file)

def tabulate_ft_ranks_by_task():
    """Create a table showing average ranks of architectures by task for FT at 100% training data.

    For each task, split, and seed, rank architectures by performance (1 = best).
    Then average ranks over seeds (41, 42, 43) and report mean ± standard error.

    Rows are tasks for Random, then tasks for Geographic. Columns are architectures.
    """
    adaptation_mode = 'FT'
    train_percent = 100
    splits = ['Random', 'Geographic']
    tags = ['chi_41', 'chi_42', 'chi_43']
    seeds = [41, 42, 43]

    # Load runs with all tags in one API call, then filter in memory
    all_runs_list = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': tags}})
    all_runs = {tag: [r for r in all_runs_list if tag in r.tags] for tag in tags}

    # Collect ranks: {split: {task: {seed: {architecture: rank}}}}
    rank_data = {
        split: {
            task: {seed: {arch: None for arch in architectures_plots} for seed in seeds}
            for task in tasks
        }
        for split in splits
    }

    for split in splits:
        for task in tasks:
            metric = 'R2' if task != 'species' else 'mAP'
            metric_name = f'{split} test {metric}'

            for tag, seed in zip(tags, seeds):
                runs = all_runs[tag]

                # Collect performance per architecture for this seed
                arch_perfs = {}
                for architecture in architectures_plots:
                    run_name = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
                    run = next((run for run in runs if run.name.startswith(run_name)), None)
                    if not run:
                        continue
                    perf = run.summary_metrics.get(metric_name)
                    if perf is not None and not np.isnan(perf):
                        arch_perfs[architecture] = perf

                if len(arch_perfs) < 2:
                    continue

                # Sort by performance (descending) so best gets rank 1
                sorted_archs = sorted(arch_perfs.items(), key=lambda x: x[1], reverse=True)
                ranks = {}
                prev_perf = None
                current_rank = 1
                for i, (arch, perf) in enumerate(sorted_archs):
                    if prev_perf is not None and abs(perf - prev_perf) < 1e-10:
                        ranks[arch] = current_rank
                    else:
                        current_rank = i + 1
                        ranks[arch] = current_rank
                    prev_perf = perf

                for architecture, rank in ranks.items():
                    rank_data[split][task][seed][architecture] = rank

    # Aggregate over seeds: {split: {task: {arch: {mean,se,n}}}}
    stats = {split: {task: {arch: {'mean': np.nan, 'se': np.nan, 'n': 0} for arch in architectures_plots}
                     for task in tasks}
             for split in splits}

    for split in splits:
        for task in tasks:
            for architecture in architectures_plots:
                ranks = [rank_data[split][task][seed][architecture] for seed in seeds
                         if rank_data[split][task][seed][architecture] is not None]
                if len(ranks) == 0:
                    continue
                mean = float(np.mean(ranks))
                se = float(np.std(ranks, ddof=1) / np.sqrt(len(ranks))) if len(ranks) > 1 else 0.0
                stats[split][task][architecture] = {'mean': mean, 'se': se, 'n': len(ranks)}

    # Compute "All" column: first average over tasks for each seed, then average over seeds
    # {split: {arch: {mean, se, n}}}
    all_stats = {split: {arch: {'mean': np.nan, 'se': np.nan, 'n': 0} for arch in architectures_plots}
                 for split in splits}

    for split in splits:
        for architecture in architectures_plots:
            # Collect seed-averaged ranks (average over tasks for each seed)
            seed_avgs = []
            for seed in seeds:
                task_ranks = []
                for task in tasks:
                    rank = rank_data[split][task][seed][architecture]
                    if rank is not None:
                        task_ranks.append(rank)
                if len(task_ranks) > 0:
                    seed_avgs.append(np.mean(task_ranks))

            # Collect all individual ranks (across tasks and seeds) for SE
            all_ranks = []
            for seed in seeds:
                for task in tasks:
                    rank = rank_data[split][task][seed][architecture]
                    if rank is not None:
                        all_ranks.append(rank)

            if len(seed_avgs) > 0:
                mean = float(np.mean(seed_avgs))  # Mean: average over tasks per seed, then over seeds
                # SE: computed from all individual ranks to reflect task-level variation
                se = float(np.std(all_ranks, ddof=1) / np.sqrt(len(all_ranks))) if len(all_ranks) > 1 else 0.0
                all_stats[split][architecture] = {'mean': mean, 'se': se, 'n': len(seed_avgs)}

    # Formatting helpers
    def display_task_name(task_name: str) -> str:
        if task_name == 'soil_nitrogen':
            return 'Soil N'
        if task_name == 'soil_organic_carbon':
            return 'Soil OC'
        if task_name == 'soil_pH':
            return 'Soil pH'
        return task_name.replace('_', ' ').title()

    display_decimals = 1

    # Best rounded mean per (split, task) to bold ties after rounding (lowest rank per task)
    best_rounded_means = {}
    for split in splits:
        for task in tasks:
            best = float('inf')
            for architecture in architectures_plots:
                s = stats[split][task][architecture]
                if s['n'] > 0 and not np.isnan(s['mean']):
                    best = min(best, s['mean'])
            if best != float('inf'):
                best_rounded_means[(split, task)] = round(best, display_decimals)

    # Best rounded mean for "All" column per split
    best_rounded_all = {}
    for split in splits:
        best = float('inf')
        for architecture in architectures_plots:
            s = all_stats[split][architecture]
            if s['n'] > 0 and not np.isnan(s['mean']):
                best = min(best, s['mean'])
        if best != float('inf'):
            best_rounded_all[split] = round(best, display_decimals)

    # Build DataFrame: rows are architectures (grouped by split), columns are tasks + "All"
    formatted_data = {}
    split_column = []
    model_column = []
    row_keys = []

    for split in splits:
        for architecture in architectures_plots:
            row_key = f"{split}_{architecture}"
            row_keys.append(row_key)
            split_column.append(split)
            model_column.append(display_arch_name(architecture))
            formatted_data[row_key] = {}

            # Add task columns
            for task in tasks:
                s = stats[split][task][architecture]
                best_rounded = best_rounded_means.get((split, task))
                if s['n'] > 0 and not np.isnan(s['mean']):
                    mean_str = f"{s['mean']:.{display_decimals}f}"
                    se_str = f"{s['se']:.{display_decimals}f}"
                    rounded_mean = round(s['mean'], display_decimals)
                    if best_rounded is not None and rounded_mean == best_rounded:
                        formatted_data[row_key][task] = f"$\\mathbf{{{mean_str} \\pm {se_str}}}$"
                    else:
                        formatted_data[row_key][task] = f"${mean_str} \\pm {se_str}$"
                else:
                    formatted_data[row_key][task] = "--"

            # Add "All tasks" column
            s = all_stats[split][architecture]
            best_rounded = best_rounded_all.get(split)
            if s['n'] > 0 and not np.isnan(s['mean']):
                mean_str = f"{s['mean']:.{display_decimals}f}"
                se_str = f"{s['se']:.{display_decimals}f}"
                rounded_mean = round(s['mean'], display_decimals)
                if best_rounded is not None and rounded_mean == best_rounded:
                    formatted_data[row_key]['All tasks'] = f"$\\mathbf{{{mean_str} \\pm {se_str}}}$"
                else:
                    formatted_data[row_key]['All tasks'] = f"${mean_str} \\pm {se_str}$"
            else:
                formatted_data[row_key]['All tasks'] = "--"

    df = pd.DataFrame(formatted_data).T
    df = df.reindex(row_keys)
    # Reindex columns to include "All tasks" first, then tasks
    column_order = ['All tasks'] + list(tasks)
    df = df.reindex(column_order, axis=1)
    df.columns = [display_task_name(task) if task in tasks else task for task in column_order]
    df.insert(0, 'Model', model_column)
    df.insert(0, 'Split', split_column)

    header_line = ' & '.join(['\\textbf{Split}', '\\textbf{Model}'] + [f'\\textbf{{{c}}}' for c in df.columns[2:]]) + r' \\'
    latex = df.to_latex(index=False,
                        header=False,
                        escape=False,
                        column_format='l' + 'l' + 'c' * (len(df.columns) - 2),
                        na_rep='--')

    lines = latex.split('\n')
    toprule_idx = next(i for i, line in enumerate(lines) if '\\toprule' in line)
    lines.insert(toprule_idx + 1, header_line)
    if toprule_idx + 2 < len(lines) and '\\midrule' in lines[toprule_idx + 2]:
        lines.pop(toprule_idx + 2)
    lines.insert(toprule_idx + 2, '\\midrule')

    # Multirow grouping for Split (Random then Geographic)
    midrule_idx = next(i for i, line in enumerate(lines) if '\\midrule' in line and i > toprule_idx)
    row_count = 0
    geographic_start_idx = None
    group_size = len(architectures_plots)

    for i in range(midrule_idx + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.strip().startswith('\\'):
            parts = line.split(' & ')
            if len(parts) >= 2:
                if row_count == 0:
                    parts[0] = f"\\multirow{{{group_size}}}{{*}}{{\\textbf{{Random}}}}"
                elif row_count == group_size:
                    geographic_start_idx = i
                    parts[0] = f"\\multirow{{{group_size}}}{{*}}{{\\textbf{{Geographic}}}}"
                else:
                    parts[0] = ""
                lines[i] = ' & '.join(parts)
                row_count += 1

    if geographic_start_idx is not None:
        prev_idx = geographic_start_idx - 1
        while prev_idx >= 0 and lines[prev_idx].strip() == '':
            prev_idx -= 1
        if prev_idx < 0 or '\\midrule' not in lines[prev_idx]:
            lines.insert(geographic_start_idx, '\\midrule')

    latex = '\n'.join(lines)
    latex = (
        "\\begin{table*}[ht]\n\\centering\n" +
        "\\caption{\\textbf{Average model ranks for finetuning on all training data.} Ranks are mean $\\pm$ standard error averaged over seeds, or over tasks and seeds for the ``All tasks'' column. Lower is better.}\n" +
        "\\label{tab:ft_ranks_by_task}\n" +
        "\\resizebox{\\linewidth}{!}{%\n" +
        latex +
        "}\n" +
        "\\end{table*}\n"
    )

    tex_file = 'results_tex/ft_ranks_by_task.tex'
    with open(tex_file, 'w') as file:
        file.write(latex)
    compile_latex(tex_file)

def tabulate_ft_metrics_by_task():
    """Create a table showing average test performance by task for FT at 100% training data.

    For each task, split, and seed, record the test metric (R2 for regression tasks; mAP for species).
    Then average metrics over seeds (41, 42, 43) and report mean ± standard error.

    Rows are architectures for Random, then architectures for Geographic. Columns are tasks.
    """
    adaptation_mode = 'FT'
    train_percent = 100
    splits = ['Random', 'Geographic']
    tags = ['chi_41', 'chi_42', 'chi_43']
    seeds = [41, 42, 43]

    # Load runs with all tags in one API call, then filter in memory
    all_runs_list = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': tags}})
    all_runs = {tag: [r for r in all_runs_list if tag in r.tags] for tag in tags}

    # Collect performance: {split: {task: {seed: {architecture: perf}}}}
    perf_data = {
        split: {
            task: {seed: {arch: None for arch in architectures_plots} for seed in seeds}
            for task in tasks
        }
        for split in splits
    }

    for split in splits:
        for task in tasks:
            metric = 'R2' if task != 'species' else 'mAP'
            metric_name = f'{split} test {metric}'

            for tag, seed in zip(tags, seeds):
                runs = all_runs[tag]
                for architecture in architectures_plots:
                    run_name = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
                    run = next((run for run in runs if run.name.startswith(run_name)), None)
                    if not run:
                        continue
                    perf = run.summary_metrics.get(metric_name)
                    if perf is not None and not np.isnan(perf):
                        perf_data[split][task][seed][architecture] = float(perf)

    # Aggregate over seeds: {split: {task: {arch: {mean,se,n}}}}
    stats = {split: {task: {arch: {'mean': np.nan, 'se': np.nan, 'n': 0} for arch in architectures_plots}
                     for task in tasks}
             for split in splits}

    for split in splits:
        for task in tasks:
            for architecture in architectures_plots:
                perfs = [perf_data[split][task][seed][architecture] for seed in seeds
                         if perf_data[split][task][seed][architecture] is not None]
                if len(perfs) == 0:
                    continue
                mean = float(np.mean(perfs))
                se = float(np.std(perfs, ddof=1) / np.sqrt(len(perfs))) if len(perfs) > 1 else 0.0
                stats[split][task][architecture] = {'mean': mean, 'se': se, 'n': len(perfs)}

    # Formatting helpers
    def display_task_name(task_name: str) -> str:
        if task_name == 'soil_nitrogen':
            return 'Soil N'
        if task_name == 'soil_organic_carbon':
            return 'Soil OC'
        if task_name == 'soil_pH':
            return 'Soil pH'
        return task_name.replace('_', ' ').title()

    display_decimals = 2

    # Best rounded mean per (split, task) to bold ties after rounding (highest metric per task)
    best_rounded_means = {}
    for split in splits:
        for task in tasks:
            best = -float('inf')
            for architecture in architectures_plots:
                s = stats[split][task][architecture]
                if s['n'] > 0 and not np.isnan(s['mean']):
                    best = max(best, s['mean'])
            if best != -float('inf'):
                best_rounded_means[(split, task)] = round(best, display_decimals)

    # Build DataFrame: rows are architectures (grouped by split), columns are tasks
    formatted_data = {}
    split_column = []
    model_column = []
    row_keys = []

    for split in splits:
        for architecture in architectures_plots:
            row_key = f"{split}_{architecture}"
            row_keys.append(row_key)
            split_column.append(split)
            model_column.append(display_arch_name(architecture))
            formatted_data[row_key] = {}

            for task in tasks:
                s = stats[split][task][architecture]
                best_rounded = best_rounded_means.get((split, task))
                if s['n'] > 0 and not np.isnan(s['mean']):
                    mean_str = f"{s['mean']:.{display_decimals}f}"
                    se_str = f"{s['se']:.{display_decimals}f}"
                    rounded_mean = round(s['mean'], display_decimals)
                    if best_rounded is not None and rounded_mean == best_rounded:
                        formatted_data[row_key][task] = f"$\\mathbf{{{mean_str} \\pm {se_str}}}$"
                    else:
                        formatted_data[row_key][task] = f"${mean_str} \\pm {se_str}$"
                else:
                    formatted_data[row_key][task] = "--"

    df = pd.DataFrame(formatted_data).T
    df = df.reindex(row_keys)
    df = df.reindex(list(tasks), axis=1)
    df.columns = [display_task_name(task) for task in tasks]
    df.insert(0, 'Model', model_column)
    df.insert(0, 'Split', split_column)

    header_line = ' & '.join(['\\textbf{Split}', '\\textbf{Model}'] + [f'\\textbf{{{c}}}' for c in df.columns[2:]]) + r' \\'
    latex = df.to_latex(index=False,
                        header=False,
                        escape=False,
                        column_format='l' + 'l' + 'c' * (len(df.columns) - 2),
                        na_rep='--')

    lines = latex.split('\n')
    toprule_idx = next(i for i, line in enumerate(lines) if '\\toprule' in line)
    lines.insert(toprule_idx + 1, header_line)
    if toprule_idx + 2 < len(lines) and '\\midrule' in lines[toprule_idx + 2]:
        lines.pop(toprule_idx + 2)
    lines.insert(toprule_idx + 2, '\\midrule')

    # Multirow grouping for Split (Random then Geographic)
    midrule_idx = next(i for i, line in enumerate(lines) if '\\midrule' in line and i > toprule_idx)
    row_count = 0
    geographic_start_idx = None
    group_size = len(architectures_plots)

    for i in range(midrule_idx + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.strip().startswith('\\'):
            parts = line.split(' & ')
            if len(parts) >= 2:
                if row_count == 0:
                    parts[0] = f"\\multirow{{{group_size}}}{{*}}{{\\textbf{{Random}}}}"
                elif row_count == group_size:
                    geographic_start_idx = i
                    parts[0] = f"\\multirow{{{group_size}}}{{*}}{{\\textbf{{Geographic}}}}"
                else:
                    parts[0] = ""
                lines[i] = ' & '.join(parts)
                row_count += 1

    if geographic_start_idx is not None:
        prev_idx = geographic_start_idx - 1
        while prev_idx >= 0 and lines[prev_idx].strip() == '':
            prev_idx -= 1
        if prev_idx < 0 or '\\midrule' not in lines[prev_idx]:
            lines.insert(geographic_start_idx, '\\midrule')

    latex = '\n'.join(lines)
    latex = (
        "\\begin{table*}[ht]\n\\centering\n" +
        "\\caption{\\textbf{Average model test performance after finetuning on all training data.} Values are mean $\\pm$ standard error over seeds (R$^2$ for regression tasks; mAP for species). Higher is better.}\n" +
        "\\label{tab:ft_metrics_by_task}\n" +
        "\\resizebox{\\linewidth}{!}{%\n" +
        latex +
        "}\n" +
        "\\end{table*}\n"
    )

    tex_file = 'results_tex/ft_metrics_by_task.tex'
    with open(tex_file, 'w') as file:
        file.write(latex)
    compile_latex(tex_file)

def tabulate_ft_ranked_models_by_task():
    """Create tables showing the ranked architectures per task (and overall) for FT at 5%, 50%, and 100% training data.

    Uses the same ranking logic as tabulate_ft_ranks_by_task():
    - For each task, split, and seed, rank architectures by performance (1 = best)
    - Average ranks over seeds
    - For the "All tasks" column, average ranks over tasks for each seed, then over seeds
    """

    adaptation_mode = 'FT'
    train_percents = [5, 50, 100]
    splits = ['Random', 'Geographic']
    tags = ['chi_41', 'chi_42', 'chi_43']
    seeds = [41, 42, 43]

    # Exclude ConvNeXtV2A from this table
    architectures_for_table = [arch for arch in architectures_plots]

    # Load runs with all tags in one API call, then filter in memory (shared across all train_percents)
    all_runs_list = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': tags}})
    all_runs = {tag: [r for r in all_runs_list if tag in r.tags] for tag in tags}

    # Formatting helpers (used for all tables)
    def display_task_name(task_name: str) -> str:
        if task_name == 'soil_nitrogen':
            return 'Soil N'
        if task_name == 'soil_organic_carbon':
            return 'Soil OC'
        if task_name == 'soil_pH':
            return 'Soil pH'
        return task_name.replace('_', ' ').title()

    # Rank labels: 1..N (one row per model rank)
    rank_labels = [str(i) for i in range(1, len(architectures_for_table) + 1)]

    # Use same RGB colors as in plots (ARCHITECTURE_COLORS from matplotlib tab10)
    def _arch_to_latex_rgb(arch: str) -> str:
        if arch not in ARCHITECTURE_COLORS:
            return None
        r, g, b = ARCHITECTURE_COLORS[arch][:3]
        return f"{int(r*255)},{int(g*255)},{int(b*255)}"

    def colorize_arch_name(arch: str) -> str:
        rgb = _arch_to_latex_rgb(arch)
        name = display_arch_name(arch)
        if rgb:
            return f"\\textcolor[RGB]{{{rgb}}}{{{name}}}"
        return f"\\textcolor{{black}}{{{name}}}"

    for train_percent in train_percents:
        # Collect ranks for this train_percent: {split: {task: {seed: {architecture: rank}}}}
        rank_data = {
            split: {
                task: {seed: {arch: None for arch in architectures_for_table} for seed in seeds}
                for task in tasks
            }
            for split in splits
        }

        for split in splits:
            for task in tasks:
                metric = 'R2' if task != 'species' else 'mAP'
                metric_name = f'{split} test {metric}'

                for tag, seed in zip(tags, seeds):
                    runs = all_runs[tag]

                    arch_perfs = {}
                    for architecture in architectures_for_table:
                        run_name = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
                        run = next((run for run in runs if run.name.startswith(run_name)), None)
                        if not run:
                            continue
                        perf = run.summary_metrics.get(metric_name)
                        if perf is not None and not np.isnan(perf):
                            arch_perfs[architecture] = float(perf)

                    if len(arch_perfs) < 2:
                        continue

                    sorted_archs = sorted(arch_perfs.items(), key=lambda x: x[1], reverse=True)
                    ranks = {}
                    prev_perf = None
                    current_rank = 1
                    for i, (arch, perf) in enumerate(sorted_archs):
                        if prev_perf is not None and abs(perf - prev_perf) < 1e-10:
                            ranks[arch] = current_rank
                        else:
                            current_rank = i + 1
                            ranks[arch] = current_rank
                        prev_perf = perf

                    for architecture, rank in ranks.items():
                        rank_data[split][task][seed][architecture] = rank

        # Aggregate mean rank over seeds
        stats = {split: {task: {arch: {'mean': np.nan, 'n': 0} for arch in architectures_for_table}
                         for task in tasks}
                 for split in splits}

        for split in splits:
            for task in tasks:
                for architecture in architectures_for_table:
                    ranks = [rank_data[split][task][seed][architecture] for seed in seeds
                             if rank_data[split][task][seed][architecture] is not None]
                    if len(ranks) == 0:
                        continue
                    stats[split][task][architecture] = {'mean': float(np.mean(ranks)), 'n': len(ranks)}

        # Aggregate mean rank for "All tasks" column
        all_stats = {split: {arch: {'mean': np.nan, 'n': 0} for arch in architectures_for_table}
                     for split in splits}

        for split in splits:
            for architecture in architectures_for_table:
                seed_avgs = []
                for seed in seeds:
                    task_ranks = []
                    for task in tasks:
                        rank = rank_data[split][task][seed][architecture]
                        if rank is not None:
                            task_ranks.append(rank)
                    if len(task_ranks) > 0:
                        seed_avgs.append(float(np.mean(task_ranks)))

                if len(seed_avgs) > 0:
                    all_stats[split][architecture] = {'mean': float(np.mean(seed_avgs)), 'n': len(seed_avgs)}

        # Compute full ordering of architectures per (split, task) and per split (All tasks)
        ordered_archs = {split: {task: [] for task in tasks} for split in splits}
        ordered_archs_all = {split: [] for split in splits}

        for split in splits:
            for task in tasks:
                arch_means = []
                for architecture in architectures_for_table:
                    s = stats[split][task][architecture]
                    if s['n'] > 0 and not np.isnan(s['mean']):
                        arch_means.append((architecture, s['mean']))
                arch_means.sort(key=lambda x: (x[1], x[0]))
                ordered_archs[split][task] = [arch for arch, _ in arch_means]

            arch_means_all = []
            for architecture in architectures_for_table:
                s = all_stats[split][architecture]
                if s['n'] > 0 and not np.isnan(s['mean']):
                    arch_means_all.append((architecture, s['mean']))
            arch_means_all.sort(key=lambda x: (x[1], x[0]))
            ordered_archs_all[split] = [arch for arch, _ in arch_means_all]

        # Build DataFrame: rows are ranks (grouped by split), columns are tasks + "All tasks"
        formatted_data = {}
        split_column = []
        rank_column = []
        row_keys = []

        for split in splits:
            for i, label in enumerate(rank_labels):
                row_key = f"{split}_{i}"
                row_keys.append(row_key)
                split_column.append(split)
                rank_column.append(label)
                formatted_data[row_key] = {}

                for task in tasks:
                    archs = ordered_archs[split][task]
                    if i < len(archs):
                        formatted_data[row_key][task] = colorize_arch_name(archs[i])
                    else:
                        formatted_data[row_key][task] = "--"

                archs_all = ordered_archs_all[split]
                if i < len(archs_all):
                    formatted_data[row_key]['All tasks'] = colorize_arch_name(archs_all[i])
                else:
                    formatted_data[row_key]['All tasks'] = "--"

        df = pd.DataFrame(formatted_data).T
        df = df.reindex(row_keys)
        column_order = ['All tasks'] + list(tasks)
        df = df.reindex(column_order, axis=1)
        df.columns = [display_task_name(task) if task in tasks else task for task in column_order]
        df.insert(0, 'Rank', rank_column)
        df.insert(0, 'Split', split_column)

        header_line = ' & '.join(['\\textbf{Split}', '\\textbf{Rank}'] + [f'\\textbf{{{c}}}' for c in df.columns[2:]]) + r' \\'
        latex = df.to_latex(index=False,
                            header=False,
                            escape=False,
                            column_format='l' + 'c' + 'c' * (len(df.columns) - 2),
                            na_rep='--')

        lines = latex.split('\n')
        toprule_idx = next(i for i, line in enumerate(lines) if '\\toprule' in line)
        lines.insert(toprule_idx + 1, header_line)
        if toprule_idx + 2 < len(lines) and '\\midrule' in lines[toprule_idx + 2]:
            lines.pop(toprule_idx + 2)
        lines.insert(toprule_idx + 2, '\\midrule')

        # Multirow grouping for Split (Random then Geographic)
        midrule_idx = next(i for i, line in enumerate(lines) if '\\midrule' in line and i > toprule_idx)
        row_count = 0
        geographic_start_idx = None
        group_size = len(rank_labels)

        for i in range(midrule_idx + 1, len(lines)):
            line = lines[i]
            if line.strip() and not line.strip().startswith('\\'):
                parts = line.split(' & ')
                if len(parts) >= 2:
                    if row_count == 0:
                        parts[0] = f"\\multirow{{{group_size}}}{{*}}{{\\textbf{{Random}}}}"
                    elif row_count == group_size:
                        geographic_start_idx = i
                        parts[0] = f"\\multirow{{{group_size}}}{{*}}{{\\textbf{{Geographic}}}}"
                    else:
                        parts[0] = ""
                    lines[i] = ' & '.join(parts)
                    row_count += 1

        if geographic_start_idx is not None:
            prev_idx = geographic_start_idx - 1
            while prev_idx >= 0 and lines[prev_idx].strip() == '':
                prev_idx -= 1
            if prev_idx < 0 or '\\midrule' not in lines[prev_idx]:
                lines.insert(geographic_start_idx, '\\midrule')

        # Bold rank-1 rows for readability
        def _split_row_terminator(cell: str) -> tuple[str, str]:
            stripped = cell.rstrip()
            if stripped.endswith('\\\\'):
                idx = stripped.rfind('\\\\')
                content = stripped[:idx].rstrip()
                return content, ' \\\\'
            return cell, ''

        for i in range(midrule_idx + 1, len(lines)):
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                continue
            if any(token in stripped for token in ['\\toprule', '\\midrule', '\\bottomrule', '\\end{tabular}', '\\begin{tabular}']):
                continue
            if ' & ' not in line:
                continue

            parts = line.split(' & ')
            if len(parts) < 2:
                continue

            last_content, last_term = _split_row_terminator(parts[-1])
            parts[-1] = last_content

            rank_cell = parts[1].strip()
            if rank_cell == '1':
                for j in range(len(parts)):
                    cell = parts[j]
                    if j == 0 and '\\multirow' in cell:
                        continue
                    if cell.strip() != '':
                        parts[j] = f"\\textbf{{{cell}}}"

            if last_term:
                parts[-1] = parts[-1] + last_term

            lines[i] = ' & '.join(parts)

        tabular_latex = '\n'.join(lines)
        train_pct_str = f"{train_percent}\\% of training data" if train_percent < 100 else "all training data"
        caption = (f"\\caption{{\\textbf{{Model rankings by task and overall after finetuning on {train_pct_str}.}} "
                   "Models are ordered by lowest average rank over seeds for each task and split, or over tasks and seeds for the ``All tasks'' column. Lower is better.}\n")
        latex = ("\\begin{table*}[ht]\n\\centering\n" +
                 caption +
                 f"\\label{{tab:ft_ranked_models_by_task_{train_percent}}}\n" +
                 "\\resizebox{\\linewidth}{!}{%\n" +
                 tabular_latex +
                 "}\n" +
                 "\\end{table*}\n")

        tex_file = f'results_tex/ft_ranked_models_by_task_{train_percent}.tex'
        with open(tex_file, 'w') as file:
            file.write(latex)
        compile_latex(tex_file)

def plot_residuals(task, JT_only=False, JT_TTT_MMR_only=False):
    if task == 'biomass':
        unit = '(Mg/ha)'
    elif task == 'soil_pH':
        unit = ''
    else:
        unit = '(g/kg)'

    # runs = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': ['residuals_42']}}) # filters to only include runs with a certain tag
    runs = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': ['residuals2']}}) # filters to only include runs with a certain tag
    # models = architectures_plots
    models = ['ConvNeXtV2A']
    data = {}

    for split in ['random_test', 'geographic_test']:
        data[split] = {}

        for model in models:
            # names = ['_'.join([task, model, 'JT-TTT', str(100)]) + '_', '_'.join([task, model, 'JT-TTT-Geo', str(100)]) + '_']
            names = ['_'.join([task, model, 'JT-TTT', str(100)]) + '_']
            data[split][model] = {}

            for name in names:
                data[split][model][name] = {}
                run = [run for run in runs if run.name.startswith(name)][0] # finds the run with the matching name

                for artifact in run.logged_artifacts():
                    if artifact.name.startswith(f'predictions_targets_{split}.pt'):
                        predictions_targets = torch.load(f'{artifact.download()}/predictions_targets_{split}.pt')
                        predictions_JT = predictions_targets['predictions_JT'].float().numpy()
                        predictions_TTT = predictions_targets['predictions_TTT'].float().numpy()
                        targets = predictions_targets['targets'].float().numpy()

                        if task == 'species':
                            predictions_JT = predictions_JT.T
                            predictions_TTT = predictions_TTT.T
                            targets = targets.T
                        else:
                            predictions_JT = predictions_JT.flatten()
                            predictions_TTT = predictions_TTT.flatten()
                            targets = targets.flatten()

                        print(predictions_JT.shape, predictions_TTT.shape, targets.shape)

                        if task == 'species':
                            species_occurrences = targets.sum(axis=1)
                            species_order = np.argsort(species_occurrences)[::-1]
                            predictions_JT = predictions_JT[species_order]
                            predictions_TTT = predictions_TTT[species_order]
                            targets = targets[species_order]

                        data[split][model][name]['predictions_JT'] = predictions_JT
                        data[split][model][name]['predictions_TTT'] = predictions_TTT
                        data[split][model][name]['targets'] = targets

    # print(np.array_equal(sorted(data['random_test'][names[0]]['predictions_JT']), sorted(data['random_test'][names[1]]['predictions_JT'])))
    # print(np.array_equal(sorted(data['random_test'][names[0]]['targets']), sorted(data['random_test'][names[1]]['targets'])))
    # print(np.array_equal(sorted(data['geographic_test'][names[0]]['predictions_JT']), sorted(data['geographic_test'][names[1]]['predictions_JT'])))
    # print(np.array_equal(sorted(data['geographic_test'][names[0]]['targets']), sorted(data['geographic_test'][names[1]]['targets'])))

    # exit()

    for split, split_data in data.items():
        targets_TTT = []
        targets_TTT_Geo = []
        predictions_JT = []
        predictions_TTT = []
        predictions_TTT_Geo = []

        for model in models:
            targets_TTT.append(split_data[model]['_'.join([task, model, 'JT-TTT', str(100)]) + '_']['targets'])
            targets_TTT_Geo.append(split_data[model]['_'.join([task, model, 'JT-TTT-Geo', str(100)]) + '_']['targets'])
            predictions_JT.append(split_data[model]['_'.join([task, model, 'JT-TTT', str(100)]) + '_']['predictions_JT'])
            predictions_TTT.append(split_data[model]['_'.join([task, model, 'JT-TTT', str(100)]) + '_']['predictions_TTT'])
            predictions_TTT_Geo.append(split_data[model]['_'.join([task, model, 'JT-TTT-Geo', str(100)]) + '_']['predictions_TTT'])

        targets_TTT = np.concatenate(targets_TTT)
        residuals_JT = np.concatenate(predictions_JT) - targets_TTT
        residuals_TTT = np.concatenate(predictions_TTT) - targets_TTT
        residuals_TTT_Geo = np.concatenate(predictions_TTT_Geo) - np.concatenate(targets_TTT_Geo)

        df = pd.DataFrame({'Target': targets_TTT, 'JT Residuals': residuals_JT, 'TTT-MMR Residuals': residuals_TTT, 'TTT-MMR-Geo Residuals': residuals_TTT_Geo})
        min_value = np.floor(df['Target'].min())
        max_value = np.ceil(df['Target'].max())
        print(min_value)
        print(max_value)
        bin_size = (max_value - min_value) // 5

        if task == 'biomass':
            bins = np.array([0, 10, 50, 100, 150, 200, 250, 300, 350, 400, 500, max_value])
        elif task == 'soil_nitrogen':
            bins = np.array([0, 2, 5, 10, 15, 20, 25, max_value])
        elif task == 'soil_organic_carbon':
            bins = np.array([0, 20, 50, 100, 200, 300, 400, max_value])
        else:
            bins = np.arange(min_value, max_value + bin_size, bin_size)

        df['Bin'] = pd.cut(df['Target'], bins=bins, include_lowest=True)
        bin_intervals = []

        for i in range(len(bins) - 1):
            if i == 0:
                interval = pd.Interval(left=sorted(df['Bin'].unique())[0].left, right=bins[i+1], closed='right')
            else:
                interval = pd.Interval(left=bins[i], right=bins[i+1], closed='right')

            bin_intervals.append(interval)

        plot_data_JT = []
        plot_data_TTT = []
        plot_data_TTT_Geo = []

        for bin_interval in bin_intervals:
            subset = df[df['Bin'] == bin_interval]
            plot_data_JT.append(subset['JT Residuals'].values)
            plot_data_TTT.append(subset['TTT-MMR Residuals'].values)
            plot_data_TTT_Geo.append(subset['TTT-MMR-Geo Residuals'].values)

        fig, ax = plt.subplots(figsize=(COL_WIDTH, 4))
        indices = np.arange(len(bin_intervals))
        tick_labels = [int(x) for x in bins]

        ax2 = ax.twinx()
        counts = np.array([len(df[df['Bin'] == bin_interval]) for bin_interval in bin_intervals])
        percentages = (counts / len(df)) * 100
        ax2.bar(indices, percentages, width=1.0, color='gray', alpha=0.25, zorder=0)
        ax2.set_ylabel('Percentage (%)', color='gray', rotation=270, labelpad=15, fontsize=LEGEND_FONTSIZE)
        ax2.tick_params(axis='y', labelsize=LEGEND_FONTSIZE, labelcolor='gray', color='gray')

        width = 0.25
        offset = 0.3
        median_props = dict(color='black', linewidth=1.5)
        boxplot_JT = ax.boxplot(plot_data_JT, positions=indices-offset, widths=width, patch_artist=True, showfliers=False, boxprops=dict(facecolor='red', color='black'), medianprops=median_props)

        if not JT_only:
            boxplot_TTT = ax.boxplot(plot_data_TTT, positions=indices, widths=width, patch_artist=True, showfliers=False, boxprops=dict(facecolor='#1f77b4', color='black'), medianprops=median_props)

        if not JT_only and not JT_TTT_MMR_only:
            boxplot_TTT_Geo = ax.boxplot(plot_data_TTT_Geo, positions=indices+offset, widths=width, patch_artist=True, showfliers=False, boxprops=dict(facecolor='#ff7f0e', color='black'), medianprops=median_props)

        ax.set_title(f'{task.replace("_", " ").title().replace("Ph", "pH")} {split.replace("_", " ").title()} Residual Distribution', fontsize=LEGEND_FONTSIZE)
        ax.set_xlabel(f'Target Value {unit}', fontsize=LEGEND_FONTSIZE)
        ax.set_ylabel(f'Residual {unit}', fontsize=LEGEND_FONTSIZE)
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        tick_positions = np.arange(len(bins)) - 0.5
        ax.set_xticks(np.arange(len(bins)) - 0.5)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=LEGEND_FONTSIZE)
        ax.tick_params(axis='y', labelsize=LEGEND_FONTSIZE)

        if JT_only:
            ax.legend([boxplot_JT['boxes'][0]], ['JT'], loc='lower left', fontsize=LEGEND_FONTSIZE)
        elif JT_TTT_MMR_only:
            ax.legend([boxplot_JT['boxes'][0], boxplot_TTT['boxes'][0]], ['JT', 'TTT-MMR'], loc='lower left', fontsize=LEGEND_FONTSIZE)
        else:
            ax.legend([boxplot_JT['boxes'][0], boxplot_TTT['boxes'][0], boxplot_TTT_Geo['boxes'][0]], ['JT', 'TTT-MMR', 'TTT-MMR-Geo'], loc='lower left', fontsize=LEGEND_FONTSIZE)

        plt.tight_layout()

        if JT_only:
            plt.savefig(f'results_figures/{task}_{split}_residual_distribution_JT_only.pdf', dpi=300)
        elif JT_TTT_MMR_only:
            plt.savefig(f'results_figures/{task}_{split}_residual_distribution_JT_TTT_MMR_only.pdf', dpi=300)
        else:
            plt.savefig(f'results_figures/{task}_{split}_residual_distribution.pdf', dpi=300)

        plt.close()

def plot_residuals_species(JT_only=False, JT_TTT_MMR_only=False):
    task = 'species'
    models = architectures_plots
    data = _load_residuals_predictions_data(task, models, ['random_test', 'geographic_test'], flatten=False)

    group_size = 20
    num_groups = 5

    for split in ['random_test', 'geographic_test']:
        predictions_JT_list = []
        predictions_TTT_list = []
        predictions_TTT_Geo_list = []
        targets_list = []
        targets_Geo_list = []

        for model in models:
            name_ttt = '_'.join([task, model, 'JT-TTT', str(100)]) + '_'
            name_geo = '_'.join([task, model, 'JT-TTT-Geo', str(100)]) + '_'
            predictions_JT_list.append(data[split][model][name_ttt]['predictions_JT'])
            predictions_TTT_list.append(data[split][model][name_ttt]['predictions_TTT'])
            predictions_TTT_Geo_list.append(data[split][model][name_geo]['predictions_TTT'])
            targets_list.append(data[split][model][name_ttt]['targets'])
            targets_Geo_list.append(data[split][model][name_geo]['targets'])

        predictions_JT = np.concatenate(predictions_JT_list, axis=0)
        predictions_TTT = np.concatenate(predictions_TTT_list, axis=0)
        predictions_TTT_Geo = np.concatenate(predictions_TTT_Geo_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        targets_Geo = np.concatenate(targets_Geo_list, axis=0)

        species_occurrences = targets.sum(axis=0)
        species_order = np.argsort(species_occurrences)[::-1]
        predictions_JT = predictions_JT[:, species_order]
        predictions_TTT = predictions_TTT[:, species_order]
        predictions_TTT_Geo = predictions_TTT_Geo[:, species_order]
        targets = targets[:, species_order]
        targets_Geo = targets_Geo[:, species_order]

        residuals_JT = predictions_JT - targets
        residuals_TTT = predictions_TTT - targets
        residuals_TTT_Geo = predictions_TTT_Geo - targets_Geo

        residuals_groups_JT = []
        residuals_groups_TTT = []
        residuals_groups_TTT_Geo = []
        species_occurrences_sorted = species_occurrences[species_order]
        total_occurrences = species_occurrences_sorted.sum()
        group_percentages = []

        for g in range(num_groups):
            start = g * group_size
            end = (g + 1) * group_size
            residuals_groups_JT.append(residuals_JT[:, start:end].ravel())
            residuals_groups_TTT.append(residuals_TTT[:, start:end].ravel())
            residuals_groups_TTT_Geo.append(residuals_TTT_Geo[:, start:end].ravel())
            group_percentages.append(species_occurrences_sorted[start:end].sum() / total_occurrences * 100)

        fig, ax = plt.subplots(figsize=(COL_WIDTH, 4))
        positions = np.arange(num_groups)

        ax2 = ax.twinx()
        ax2.bar(positions, group_percentages, width=1.0, color='gray', alpha=0.25, zorder=0)
        ax2.set_ylabel('% of occurrences', color='gray', rotation=270, labelpad=15, fontsize=LEGEND_FONTSIZE)
        ax2.tick_params(axis='y', labelsize=LEGEND_FONTSIZE, labelcolor='gray', color='gray')

        width = 0.25
        offset = 0.3
        median_props = dict(color='black', linewidth=1.5)

        boxplot_JT = ax.boxplot(
            residuals_groups_JT, positions=positions - offset, widths=width,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor='red', color='black'), medianprops=median_props)

        if not JT_only:
            boxplot_TTT = ax.boxplot(
                residuals_groups_TTT, positions=positions, widths=width,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='#1f77b4', color='black'), medianprops=median_props)

        if not JT_only and not JT_TTT_MMR_only:
            boxplot_TTT_Geo = ax.boxplot(
                residuals_groups_TTT_Geo, positions=positions + offset, widths=width,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='#ff7f0e', color='black'), medianprops=median_props)

        ax.set_title(f'Species {split.replace("_", " ").title()} Residual Distribution', fontsize=LEGEND_FONTSIZE)
        ax.set_xlabel('Species group', fontsize=LEGEND_FONTSIZE)
        ax.set_ylabel('Residual', fontsize=LEGEND_FONTSIZE)
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xticks(np.arange(num_groups + 1) - 0.5, minor=True)
        ax.tick_params(axis='x', which='minor', length=4, width=1)
        ax.set_xticks(positions)
        ax.set_xticklabels(['Dominant', 'Abundant', 'Common', 'Occasional', 'Rare'], fontsize=LEGEND_FONTSIZE)
        ax.tick_params(axis='x', which='major', length=0)
        ax.tick_params(axis='y', labelsize=LEGEND_FONTSIZE)

        if JT_only:
            ax.legend([boxplot_JT['boxes'][0]], ['JT'], loc='upper right', fontsize=LEGEND_FONTSIZE)
        elif JT_TTT_MMR_only:
            ax.legend([boxplot_JT['boxes'][0], boxplot_TTT['boxes'][0]], ['JT', 'TTT-MMR'], loc='upper right', fontsize=LEGEND_FONTSIZE)
        else:
            ax.legend([boxplot_JT['boxes'][0], boxplot_TTT['boxes'][0], boxplot_TTT_Geo['boxes'][0]], ['JT', 'TTT-MMR', 'TTT-MMR-Geo'], loc='upper right', fontsize=LEGEND_FONTSIZE)

        plt.tight_layout()

        if JT_only:
            plt.savefig(f'results_figures/species_{split}_residual_distribution_JT_only.pdf', dpi=300)
        elif JT_TTT_MMR_only:
            plt.savefig(f'results_figures/species_{split}_residual_distribution_JT_TTT_MMR_only.pdf', dpi=300)
        else:
            plt.savefig(f'results_figures/species_{split}_residual_distribution.pdf', dpi=300)

        plt.close()

def plot_bce_species(JT_only=False, JT_TTT_MMR_only=False):
    task = 'species'
    models = architectures_plots
    data = _load_residuals_predictions_data(task, models, ['random_test', 'geographic_test'], flatten=False)

    group_size = 20
    num_groups = 5
    eps = 1e-7

    for split in ['random_test', 'geographic_test']:
        predictions_JT_list = []
        predictions_TTT_list = []
        predictions_TTT_Geo_list = []
        targets_list = []
        targets_Geo_list = []

        for model in models:
            name_ttt = '_'.join([task, model, 'JT-TTT', str(100)]) + '_'
            name_geo = '_'.join([task, model, 'JT-TTT-Geo', str(100)]) + '_'
            predictions_JT_list.append(data[split][model][name_ttt]['predictions_JT'])
            predictions_TTT_list.append(data[split][model][name_ttt]['predictions_TTT'])
            predictions_TTT_Geo_list.append(data[split][model][name_geo]['predictions_TTT'])
            targets_list.append(data[split][model][name_ttt]['targets'])
            targets_Geo_list.append(data[split][model][name_geo]['targets'])

        predictions_JT = np.concatenate(predictions_JT_list, axis=0)
        predictions_TTT = np.concatenate(predictions_TTT_list, axis=0)
        predictions_TTT_Geo = np.concatenate(predictions_TTT_Geo_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        targets_Geo = np.concatenate(targets_Geo_list, axis=0)

        species_occurrences = targets.sum(axis=0)
        species_order = np.argsort(species_occurrences)[::-1]
        predictions_JT = predictions_JT[:, species_order]
        predictions_TTT = predictions_TTT[:, species_order]
        predictions_TTT_Geo = predictions_TTT_Geo[:, species_order]
        targets = targets[:, species_order]
        targets_Geo = targets_Geo[:, species_order]

        predictions_JT = np.clip(predictions_JT, eps, 1 - eps)
        predictions_TTT = np.clip(predictions_TTT, eps, 1 - eps)
        predictions_TTT_Geo = np.clip(predictions_TTT_Geo, eps, 1 - eps)

        bce_JT = -(targets * np.log(predictions_JT) + (1 - targets) * np.log(1 - predictions_JT)).mean(axis=0)
        bce_TTT = -(targets * np.log(predictions_TTT) + (1 - targets) * np.log(1 - predictions_TTT)).mean(axis=0)
        bce_TTT_Geo = -(targets_Geo * np.log(predictions_TTT_Geo) + (1 - targets_Geo) * np.log(1 - predictions_TTT_Geo)).mean(axis=0)

        bce_groups_JT = []
        bce_groups_TTT = []
        bce_groups_TTT_Geo = []
        species_occurrences_sorted = species_occurrences[species_order]
        total_occurrences = species_occurrences_sorted.sum()
        group_percentages = []

        for g in range(num_groups):
            start = g * group_size
            end = (g + 1) * group_size
            bce_groups_JT.append(bce_JT[start:end])
            bce_groups_TTT.append(bce_TTT[start:end])
            bce_groups_TTT_Geo.append(bce_TTT_Geo[start:end])
            group_percentages.append(species_occurrences_sorted[start:end].sum() / total_occurrences * 100)

        fig, ax = plt.subplots(figsize=(COL_WIDTH, 4))
        positions = np.arange(num_groups)

        ax2 = ax.twinx()
        ax2.bar(positions, group_percentages, width=1.0, color='gray', alpha=0.25, zorder=0)
        ax2.set_ylabel('% of occurrences', color='gray', rotation=270, labelpad=15, fontsize=LEGEND_FONTSIZE)
        ax2.tick_params(axis='y', labelsize=LEGEND_FONTSIZE, labelcolor='gray', color='gray')

        width = 0.25
        offset = 0.3
        median_props = dict(color='black', linewidth=1.5)

        boxplot_JT = ax.boxplot(
            bce_groups_JT, positions=positions - offset, widths=width,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor='red', color='black'), medianprops=median_props)

        if not JT_only:
            boxplot_TTT = ax.boxplot(
                bce_groups_TTT, positions=positions, widths=width,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='#1f77b4', color='black'), medianprops=median_props)

        if not JT_only and not JT_TTT_MMR_only:
            boxplot_TTT_Geo = ax.boxplot(
                bce_groups_TTT_Geo, positions=positions + offset, widths=width,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='#ff7f0e', color='black'), medianprops=median_props)

        ax.set_title(f'Species {split.replace("_", " ").title()} BCE Loss', fontsize=LEGEND_FONTSIZE)
        ax.set_xlabel('Species group', fontsize=LEGEND_FONTSIZE)
        ax.set_ylabel('BCE loss (per species)', fontsize=LEGEND_FONTSIZE)
        ax.set_xticks(np.arange(num_groups + 1) - 0.5, minor=True)
        ax.tick_params(axis='x', which='minor', length=4, width=1)
        ax.set_xticks(positions)
        ax.set_xticklabels(['Dominant', 'Abundant', 'Common', 'Occasional', 'Rare'], fontsize=LEGEND_FONTSIZE)
        ax.tick_params(axis='x', which='major', length=0)
        ax.tick_params(axis='y', labelsize=LEGEND_FONTSIZE)

        if JT_only:
            ax.legend([boxplot_JT['boxes'][0]], ['JT'], loc='upper right', fontsize=LEGEND_FONTSIZE)
        elif JT_TTT_MMR_only:
            ax.legend([boxplot_JT['boxes'][0], boxplot_TTT['boxes'][0]], ['JT', 'TTT-MMR'], loc='upper right', fontsize=LEGEND_FONTSIZE)
        else:
            ax.legend([boxplot_JT['boxes'][0], boxplot_TTT['boxes'][0], boxplot_TTT_Geo['boxes'][0]], ['JT', 'TTT-MMR', 'TTT-MMR-Geo'], loc='upper right', fontsize=LEGEND_FONTSIZE)

        plt.tight_layout()

        if JT_only:
            plt.savefig(f'results_figures/species_{split}_bce_JT_only.pdf', dpi=300)
        elif JT_TTT_MMR_only:
            plt.savefig(f'results_figures/species_{split}_bce_JT_TTT_MMR_only.pdf', dpi=300)
        else:
            plt.savefig(f'results_figures/species_{split}_bce.pdf', dpi=300)

        plt.close()

def plot_ap_species(JT_only=False, JT_TTT_MMR_only=False):
    task = 'species'
    models = architectures_plots
    data = _load_residuals_predictions_data(task, models, ['random_test', 'geographic_test'], flatten=False)

    group_size = 20
    num_groups = 5

    fig, axes = plt.subplots(2, 1, figsize=(COL_WIDTH, 8), sharex=True, gridspec_kw=dict(hspace=0.15))
    for split_idx, split in enumerate(['random_test', 'geographic_test']):
        predictions_JT_list = []
        predictions_TTT_list = []
        predictions_TTT_Geo_list = []
        targets_list = []
        targets_Geo_list = []

        for model in models:
            name_ttt = '_'.join([task, model, 'JT-TTT', str(100)]) + '_'
            name_geo = '_'.join([task, model, 'JT-TTT-Geo', str(100)]) + '_'
            predictions_JT_list.append(data[split][model][name_ttt]['predictions_JT'])
            predictions_TTT_list.append(data[split][model][name_ttt]['predictions_TTT'])
            predictions_TTT_Geo_list.append(data[split][model][name_geo]['predictions_TTT'])
            targets_list.append(data[split][model][name_ttt]['targets'])
            targets_Geo_list.append(data[split][model][name_geo]['targets'])

        predictions_JT = np.concatenate(predictions_JT_list, axis=0)
        predictions_TTT = np.concatenate(predictions_TTT_list, axis=0)
        predictions_TTT_Geo = np.concatenate(predictions_TTT_Geo_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        targets_Geo = np.concatenate(targets_Geo_list, axis=0)

        species_occurrences = targets.sum(axis=0)
        species_order = np.argsort(species_occurrences)[::-1]
        predictions_JT = predictions_JT[:, species_order]
        predictions_TTT = predictions_TTT[:, species_order]
        predictions_TTT_Geo = predictions_TTT_Geo[:, species_order]
        targets = targets[:, species_order]
        targets_Geo = targets_Geo[:, species_order]

        num_species = targets.shape[1]
        ap_JT = np.array([average_precision_score(targets[:, i], predictions_JT[:, i]) for i in range(num_species)])
        ap_TTT = np.array([average_precision_score(targets[:, i], predictions_TTT[:, i]) for i in range(num_species)])
        ap_TTT_Geo = np.array([average_precision_score(targets_Geo[:, i], predictions_TTT_Geo[:, i]) for i in range(num_species)])

        ap_groups_JT = []
        ap_groups_TTT = []
        ap_groups_TTT_Geo = []
        species_occurrences_sorted = species_occurrences[species_order]
        total_occurrences = species_occurrences_sorted.sum()
        group_percentages = []

        for g in range(num_groups):
            start = g * group_size
            end = (g + 1) * group_size
            ap_groups_JT.append(ap_JT[start:end])
            ap_groups_TTT.append(ap_TTT[start:end])
            ap_groups_TTT_Geo.append(ap_TTT_Geo[start:end])
            group_percentages.append(species_occurrences_sorted[start:end].sum() / total_occurrences * 100)

        ax = axes[split_idx]
        positions = np.arange(num_groups)

        ax2 = ax.twinx()
        ax2.bar(positions, group_percentages, width=1.0, color='gray', alpha=0.5, zorder=0)
        ax2.set_ylabel('% of occurrences', color='gray', rotation=270, labelpad=15, fontsize=LEGEND_FONTSIZE+4)
        ax2.tick_params(axis='y', labelsize=LEGEND_FONTSIZE, labelcolor='gray', color='gray')

        width = 0.25
        offset = 0.3
        median_props = dict(color='black', linewidth=1.5)

        boxplot_JT = ax.boxplot(
            ap_groups_JT, positions=positions - offset, widths=width,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor='red', color='black'), medianprops=median_props)

        if not JT_only:
            boxplot_TTT = ax.boxplot(
                ap_groups_TTT, positions=positions, widths=width,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='#1f77b4', color='black'), medianprops=median_props)

        if not JT_only and not JT_TTT_MMR_only:
            boxplot_TTT_Geo = ax.boxplot(
                ap_groups_TTT_Geo, positions=positions + offset, widths=width,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='#ff7f0e', color='black'), medianprops=median_props)

        ax.set_title(f'Species {split.replace("_", " ").title()} Average Precision', fontsize=LEGEND_FONTSIZE+4)
        ax.set_ylabel('Average precision (per species)', fontsize=LEGEND_FONTSIZE+4)
        ax.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xticks(np.arange(num_groups + 1) - 0.5, minor=True)
        ax.tick_params(axis='x', which='minor', length=4, width=1)
        ax.set_xticks(positions)
        if split_idx == 1:
            ax.set_xticklabels(['Dominant', 'Abundant', 'Common', 'Occasional', 'Rare'], fontsize=LEGEND_FONTSIZE+4)
            ax.set_xlabel('Species group', fontsize=LEGEND_FONTSIZE+4)
        else:
            ax.set_xticklabels([])
            ax.set_xlabel('')
        ax.tick_params(axis='x', which='major', length=0)
        ax.tick_params(axis='y', labelsize=LEGEND_FONTSIZE)

        if split_idx == 0:
            if JT_only:
                ax.legend([boxplot_JT['boxes'][0]], ['JT'], loc='upper right', fontsize=LEGEND_FONTSIZE+4)
            elif JT_TTT_MMR_only:
                ax.legend([boxplot_JT['boxes'][0], boxplot_TTT['boxes'][0]], ['JT', 'TTT-MMR'], loc='upper right', fontsize=LEGEND_FONTSIZE+4)
            else:
                ax.legend([boxplot_JT['boxes'][0], boxplot_TTT['boxes'][0], boxplot_TTT_Geo['boxes'][0]], ['JT', 'TTT-MMR', 'TTT-MMR-Geo'], loc='upper right', fontsize=LEGEND_FONTSIZE+4)

    plt.tight_layout()

    if JT_only:
        plt.savefig(f'results_figures/species_ap_combined_JT_only.pdf', dpi=300)
        plt.savefig(f'results_figures/species_ap_combined_JT_only.png', dpi=300)
    elif JT_TTT_MMR_only:
        plt.savefig(f'results_figures/species_ap_combined_JT_TTT_MMR_only.pdf', dpi=300)
        plt.savefig(f'results_figures/species_ap_combined_JT_TTT_MMR_only.png', dpi=300)
    else:
        plt.savefig(f'results_figures/species_ap_combined.pdf', dpi=300)
        plt.savefig(f'results_figures/species_ap_combined.png', dpi=300)

    plt.close()

def plot_ap_species_ungrouped(JT_only=False, JT_TTT_MMR_only=False):
    """Per-species AP plot with no prevalence grouping: one box per species.

    Each box summarizes the distribution of that species' average precision across
    the architectures in `architectures_plots`. Species are ordered from most to
    least common (descending occurrence count). Two rows: random / geographic test.
    """
    task = 'species'
    models = architectures_plots
    data = _load_residuals_predictions_data(task, models, ['random_test', 'geographic_test'], flatten=False)

    def _ap_per_species(t, p):
        S = t.shape[1]
        out = np.full(S, np.nan)
        for i in range(S):
            if t[:, i].sum() > 0:
                out[i] = average_precision_score(t[:, i], p[:, i])
        return out

    def _box_data(mat, order):
        # mat: (num_models, num_species); return list (per ordered species) of finite AP values
        cols = mat[:, order]
        return [cols[:, i][np.isfinite(cols[:, i])] for i in range(cols.shape[1])]

    num_species = None
    fig = axes = None

    for split_idx, split in enumerate(['random_test', 'geographic_test']):
        ap_JT_rows = []
        ap_TTT_rows = []
        ap_TTT_Geo_rows = []
        occ_accum = None

        for model in models:
            name_ttt = '_'.join([task, model, 'JT-TTT', str(100)]) + '_'
            name_geo = '_'.join([task, model, 'JT-TTT-Geo', str(100)]) + '_'
            d = data[split][model]
            if name_ttt not in d or name_geo not in d:
                continue
            tj = d[name_ttt]['targets']
            tg = d[name_geo]['targets']
            ap_JT_rows.append(_ap_per_species(tj, d[name_ttt]['predictions_JT']))
            ap_TTT_rows.append(_ap_per_species(tj, d[name_ttt]['predictions_TTT']))
            ap_TTT_Geo_rows.append(_ap_per_species(tg, d[name_geo]['predictions_TTT']))
            occ_accum = tj.sum(axis=0) if occ_accum is None else occ_accum + tj.sum(axis=0)

        if not ap_JT_rows:
            continue

        ap_JT_mat = np.vstack(ap_JT_rows)
        ap_TTT_mat = np.vstack(ap_TTT_rows)
        ap_TTT_Geo_mat = np.vstack(ap_TTT_Geo_rows)

        species_order = np.argsort(occ_accum)[::-1]
        occ_sorted = occ_accum[species_order]
        total_occurrences = occ_sorted.sum()

        if num_species is None:
            num_species = ap_JT_mat.shape[1]
            fig, axes = plt.subplots(
                2, 1, figsize=(max(COL_WIDTH, num_species * 0.38), 9),
                sharex=True, gridspec_kw=dict(hspace=0.15),
            )

        ax = axes[split_idx]
        positions = np.arange(num_species)

        ax2 = ax.twinx()
        ax2.bar(positions, occ_sorted / total_occurrences * 100, width=1.0, color='gray', alpha=0.5, zorder=0)
        ax2.set_ylabel('% of occurrences', color='gray', rotation=270, labelpad=15, fontsize=LEGEND_FONTSIZE)
        ax2.tick_params(axis='y', labelsize=LEGEND_FONTSIZE - 2, labelcolor='gray', color='gray')

        width = 0.22
        offset = 0.26
        median_props = dict(color='black', linewidth=1.0)

        boxplot_JT = ax.boxplot(
            _box_data(ap_JT_mat, species_order), positions=positions - offset, widths=width,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor='red', color='black', linewidth=0.4), medianprops=median_props,
            whiskerprops=dict(linewidth=0.4), capprops=dict(linewidth=0.4))

        boxplot_TTT = boxplot_TTT_Geo = None
        if not JT_only:
            boxplot_TTT = ax.boxplot(
                _box_data(ap_TTT_mat, species_order), positions=positions, widths=width,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='#1f77b4', color='black', linewidth=0.4), medianprops=median_props,
                whiskerprops=dict(linewidth=0.4), capprops=dict(linewidth=0.4))

        if not JT_only and not JT_TTT_MMR_only:
            boxplot_TTT_Geo = ax.boxplot(
                _box_data(ap_TTT_Geo_mat, species_order), positions=positions + offset, widths=width,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='#ff7f0e', color='black', linewidth=0.4), medianprops=median_props,
                whiskerprops=dict(linewidth=0.4), capprops=dict(linewidth=0.4))

        ax.set_title(f'Species {split.replace("_", " ").title()} Average Precision (per species)', fontsize=LEGEND_FONTSIZE + 2)
        ax.set_ylabel('Average precision', fontsize=LEGEND_FONTSIZE + 2)
        ax.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlim(-0.5, num_species - 0.5)
        ax2.set_xlim(-0.5, num_species - 0.5)

        tick_step = 10
        xticks = np.arange(0, num_species, tick_step)
        ax.set_xticks(xticks)
        if split_idx == 1:
            ax.set_xticklabels([str(t + 1) for t in xticks], fontsize=LEGEND_FONTSIZE - 2)
            ax.set_xlabel('Species rank (most \u2192 least common)', fontsize=LEGEND_FONTSIZE + 2)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis='y', labelsize=LEGEND_FONTSIZE - 2)

        if split_idx == 0:
            handles = [boxplot_JT['boxes'][0]]
            labels = ['JT']
            if boxplot_TTT is not None:
                handles.append(boxplot_TTT['boxes'][0]); labels.append('TTT-MMR')
            if boxplot_TTT_Geo is not None:
                handles.append(boxplot_TTT_Geo['boxes'][0]); labels.append('TTT-MMR-Geo')
            ax.legend(handles, labels, loc='upper right', fontsize=LEGEND_FONTSIZE + 2)

    plt.tight_layout()

    suffix = '_JT_only' if JT_only else ('_JT_TTT_MMR_only' if JT_TTT_MMR_only else '')
    plt.savefig(f'results_figures/species_ap_ungrouped{suffix}.pdf', dpi=300)
    plt.savefig(f'results_figures/species_ap_ungrouped{suffix}.png', dpi=300)
    plt.close()

def plot_ap_species_ungrouped_line(JT_only=False, JT_TTT_MMR_only=False):
    """Per-species AP, no grouping: plot the mean AP across architectures as a point
    per species and connect the points with a line (one line per method).

    Species are ordered from most to least common. Two rows: random / geographic test.
    """
    task = 'species'
    models = architectures_plots
    data = _load_residuals_predictions_data(task, models, ['random_test', 'geographic_test'], flatten=False)

    def _ap_per_species(t, p):
        S = t.shape[1]
        out = np.full(S, np.nan)
        for i in range(S):
            if t[:, i].sum() > 0:
                out[i] = average_precision_score(t[:, i], p[:, i])
        return out

    num_species = None
    fig = axes = None

    for split_idx, split in enumerate(['random_test', 'geographic_test']):
        ap_JT_rows = []
        ap_TTT_rows = []
        ap_TTT_Geo_rows = []
        occ_accum = None

        for model in models:
            name_ttt = '_'.join([task, model, 'JT-TTT', str(100)]) + '_'
            name_geo = '_'.join([task, model, 'JT-TTT-Geo', str(100)]) + '_'
            d = data[split][model]
            if name_ttt not in d or name_geo not in d:
                continue
            tj = d[name_ttt]['targets']
            tg = d[name_geo]['targets']
            ap_JT_rows.append(_ap_per_species(tj, d[name_ttt]['predictions_JT']))
            ap_TTT_rows.append(_ap_per_species(tj, d[name_ttt]['predictions_TTT']))
            ap_TTT_Geo_rows.append(_ap_per_species(tg, d[name_geo]['predictions_TTT']))
            occ_accum = tj.sum(axis=0) if occ_accum is None else occ_accum + tj.sum(axis=0)

        if not ap_JT_rows:
            continue

        species_order = np.argsort(occ_accum)[::-1]
        occ_sorted = occ_accum[species_order]
        total_occurrences = occ_sorted.sum()

        mean_JT = np.nanmean(np.vstack(ap_JT_rows), axis=0)[species_order]
        mean_TTT = np.nanmean(np.vstack(ap_TTT_rows), axis=0)[species_order]
        mean_TTT_Geo = np.nanmean(np.vstack(ap_TTT_Geo_rows), axis=0)[species_order]

        if num_species is None:
            num_species = mean_JT.shape[0]
            fig, axes = plt.subplots(
                2, 1, figsize=(max(COL_WIDTH, num_species * 0.22), 9),
                sharex=True, gridspec_kw=dict(hspace=0.15),
            )

        ax = axes[split_idx]
        positions = np.arange(num_species)

        ax2 = ax.twinx()
        ax2.bar(positions, occ_sorted / total_occurrences * 100, width=1.0, color='gray', alpha=0.3, zorder=0)
        ax2.set_ylabel('% of occurrences', color='gray', rotation=270, labelpad=15, fontsize=LEGEND_FONTSIZE)
        ax2.tick_params(axis='y', labelsize=LEGEND_FONTSIZE - 2, labelcolor='gray', color='gray')

        ax.plot(positions, mean_JT, marker='o', markersize=3, linewidth=1.0, color='red', label='JT', zorder=5)
        if not JT_only:
            ax.plot(positions, mean_TTT, marker='o', markersize=3, linewidth=1.0, color='#1f77b4', label='TTT-MMR', zorder=5)
        if not JT_only and not JT_TTT_MMR_only:
            ax.plot(positions, mean_TTT_Geo, marker='o', markersize=3, linewidth=1.0, color='#ff7f0e', label='TTT-MMR-Geo', zorder=5)

        ax.set_title(f'Species {split.replace("_", " ").title()} Average Precision (mean over architectures)', fontsize=LEGEND_FONTSIZE + 2)
        ax.set_ylabel('Average precision', fontsize=LEGEND_FONTSIZE + 2)
        ax.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlim(-0.5, num_species - 0.5)
        ax2.set_xlim(-0.5, num_species - 0.5)

        tick_step = 10
        xticks = np.arange(0, num_species, tick_step)
        ax.set_xticks(xticks)
        if split_idx == 1:
            ax.set_xticklabels([str(t + 1) for t in xticks], fontsize=LEGEND_FONTSIZE - 2)
            ax.set_xlabel('Species rank (most \u2192 least common)', fontsize=LEGEND_FONTSIZE + 2)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis='y', labelsize=LEGEND_FONTSIZE - 2)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

        if split_idx == 0:
            ax.legend(loc='upper right', fontsize=LEGEND_FONTSIZE + 2)

    plt.tight_layout()

    suffix = '_JT_only' if JT_only else ('_JT_TTT_MMR_only' if JT_TTT_MMR_only else '')
    plt.savefig(f'results_figures/species_ap_ungrouped_line{suffix}.pdf', dpi=300)
    plt.savefig(f'results_figures/species_ap_ungrouped_line{suffix}.png', dpi=300)
    plt.close()

def plot_residuals_combined(JT_only=False, JT_TTT_MMR_only=False):
    """Plot residual distributions for all four regression tasks in a 2x4 grid.
    Top row: Random split. Bottom row: Geographic split.
    JT_only: only plot the JT boxplot.
    JT_TTT_MMR_only: plot JT and TTT-MMR boxplots, omitting TTT-MMR-Geo.
    """
    all_tasks = ['biomass', 'soil_nitrogen', 'soil_organic_carbon', 'soil_pH']
    splits = ['random_test', 'geographic_test']

    task_units = {
        'biomass': '(Mg/ha)',
        'soil_nitrogen': '(g/kg)',
        'soil_organic_carbon': '(g/kg)',
        'soil_pH': '',
    }

    task_display_names = {
        'biomass': 'Biomass',
        'soil_nitrogen': 'Soil N',
        'soil_organic_carbon': 'Soil OC',
        'soil_pH': 'Soil pH',
    }

    task_bins = {
        'biomass': {
            'random_test': np.array([0, 5, 30, 100, 200]),
            'geographic_test': np.array([0, 2, 20, 100, 200]),
        },
        'soil_nitrogen': {
            'random_test': np.array([0, 1, 3, 10, 20]),
            'geographic_test': np.array([0, 1, 3, 10, 15]),
        },
        'soil_organic_carbon': {
            'random_test': np.array([0, 20, 50, 200, 350]),
            'geographic_test': np.array([0, 10, 20, 50, 100]),
        },
        'soil_pH': {
            'random_test': np.array([3, 4, 5, 7, 8]),
            'geographic_test': np.array([3, 5, 7, 8, 9]),
        }
    }

    models = architectures_plots

    # Load all data: {task: {split: {model: {name: {predictions_JT, predictions_TTT, targets}}}}}
    all_data = {}
    for task in all_tasks:
        print(f"Loading data for {task}...")
        all_data[task] = _load_residuals_predictions_data(task, models, splits, flatten=True)

    fig, axes = plt.subplots(2, 4, figsize=(COL_WIDTH * 2, 4.5),
                             gridspec_kw=dict(left=0.08, right=0.95, top=0.88, bottom=0.1, wspace=0.40, hspace=0.25))

    for col_idx, task in enumerate(all_tasks):
        for row_idx, split in enumerate(splits):
            ax = axes[row_idx, col_idx]

            # Aggregate residuals across models
            targets_TTT_list = []
            targets_TTT_Geo_list = []
            predictions_JT_list = []
            predictions_TTT_list = []
            predictions_TTT_Geo_list = []

            for model in models:
                name_ttt = '_'.join([task, model, 'JT-TTT', str(100)]) + '_'
                name_geo = '_'.join([task, model, 'JT-TTT-Geo', str(100)]) + '_'
                d = all_data[task][split][model]
                if name_ttt in d and 'targets' in d[name_ttt] and name_geo in d and 'targets' in d[name_geo]:
                    targets_TTT_list.append(d[name_ttt]['targets'])
                    targets_TTT_Geo_list.append(d[name_geo]['targets'])
                    predictions_JT_list.append(d[name_ttt]['predictions_JT'])
                    predictions_TTT_list.append(d[name_ttt]['predictions_TTT'])
                    predictions_TTT_Geo_list.append(d[name_geo]['predictions_TTT'])

            if not targets_TTT_list:
                continue

            targets_TTT = np.concatenate(targets_TTT_list)
            residuals_JT = np.concatenate(predictions_JT_list) - targets_TTT
            residuals_TTT = np.concatenate(predictions_TTT_list) - targets_TTT
            residuals_TTT_Geo = np.concatenate(predictions_TTT_Geo_list) - np.concatenate(targets_TTT_Geo_list)

            df = pd.DataFrame({'Target': targets_TTT, 'JT': residuals_JT, 'TTT': residuals_TTT, 'TTT-Geo': residuals_TTT_Geo})

            min_value = np.floor(df['Target'].min())
            max_value = np.ceil(df['Target'].max())

            if task_bins[task][split] is not None:
                bins = np.append(task_bins[task][split], max_value) if max_value > task_bins[task][split][-1] else task_bins[task][split]
            else:
                bin_size = (max_value - min_value) // 5
                bins = np.arange(min_value, max_value + bin_size, bin_size)

            df['Bin'] = pd.cut(df['Target'], bins=bins, include_lowest=True)

            bin_intervals = []
            unique_bins = sorted(df['Bin'].dropna().unique())
            for i in range(len(bins) - 1):
                if i == 0 and unique_bins:
                    interval = pd.Interval(left=unique_bins[0].left, right=bins[i+1], closed='right')
                else:
                    interval = pd.Interval(left=bins[i], right=bins[i+1], closed='right')
                bin_intervals.append(interval)

            plot_data_JT = []
            plot_data_TTT = []
            plot_data_TTT_Geo = []

            for bi in bin_intervals:
                subset = df[df['Bin'] == bi]
                plot_data_JT.append(subset['JT'].values if len(subset) > 0 else [])
                plot_data_TTT.append(subset['TTT'].values if len(subset) > 0 else [])
                plot_data_TTT_Geo.append(subset['TTT-Geo'].values if len(subset) > 0 else [])

            indices = np.arange(len(bin_intervals))

            # Histogram background
            ax2 = ax.twinx()
            counts = np.array([len(df[df['Bin'] == bi]) for bi in bin_intervals])
            percentages = (counts / len(df)) * 100
            ax2.bar(indices, percentages, width=1.0, color='gray', alpha=0.5, zorder=0)
            ax2.tick_params(axis='y', labelsize=LEGEND_FONTSIZE , labelcolor='gray', color='gray')

            # Boxplots
            median_props = dict(color='black', linewidth=1)
            c_jt = 'red'
            c_ttt = '#1f77b4'
            c_geo = '#ff7f0e'
            if JT_only:
                width = 0.5
                bp_jt = ax.boxplot(
                    plot_data_JT, positions=indices, widths=width, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=c_jt, edgecolor=c_jt), medianprops=median_props,
                    whiskerprops=dict(color=c_jt, linewidth=1), capprops=dict(color=c_jt, linewidth=1),
                )
                bp_ttt = bp_geo = None
                active_bps = [bp_jt]
            elif JT_TTT_MMR_only:
                width = 0.3
                offset = 0.2
                bp_jt = ax.boxplot(
                    plot_data_JT, positions=indices - offset, widths=width, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=c_jt, edgecolor=c_jt), medianprops=median_props,
                    whiskerprops=dict(color=c_jt, linewidth=1), capprops=dict(color=c_jt, linewidth=1),
                )
                bp_ttt = ax.boxplot(
                    plot_data_TTT, positions=indices + offset, widths=width, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=c_ttt, edgecolor=c_ttt), medianprops=median_props,
                    whiskerprops=dict(color=c_ttt, linewidth=1), capprops=dict(color=c_ttt, linewidth=1),
                )
                bp_geo = None
                active_bps = [bp_jt, bp_ttt]
            else:
                width = 0.25
                offset = 0.3
                bp_jt = ax.boxplot(
                    plot_data_JT, positions=indices - offset, widths=width, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=c_jt, edgecolor=c_jt), medianprops=median_props,
                    whiskerprops=dict(color=c_jt, linewidth=1), capprops=dict(color=c_jt, linewidth=1),
                )
                bp_ttt = ax.boxplot(
                    plot_data_TTT, positions=indices, widths=width, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=c_ttt, edgecolor=c_ttt), medianprops=median_props,
                    whiskerprops=dict(color=c_ttt, linewidth=1), capprops=dict(color=c_ttt, linewidth=1),
                )
                bp_geo = ax.boxplot(
                    plot_data_TTT_Geo, positions=indices + offset, widths=width, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=c_geo, edgecolor=c_geo), medianprops=median_props,
                    whiskerprops=dict(color=c_geo, linewidth=1), capprops=dict(color=c_geo, linewidth=1),
                )
                active_bps = [bp_jt, bp_ttt, bp_geo]

            for bp in active_bps:
                for box in bp['boxes']:
                    box.set_zorder(5)
                for whisker in bp['whiskers']:
                    whisker.set_zorder(3)
                for cap in bp['caps']:
                    cap.set_zorder(3)
                for median in bp['medians']:
                    median.set_zorder(6)

            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)

            # Ticks and consistent x-axis limits
            tick_positions = np.arange(len(bins)) - 0.5
            tick_labels_list = [int(x) for x in bins]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels_list, fontsize=LEGEND_FONTSIZE)
            ax.tick_params(axis='y', labelsize=LEGEND_FONTSIZE)
            ax.set_xlim(-0.5, len(bin_intervals) - 0.5)
            ax2.set_xlim(-0.5, len(bin_intervals) - 0.5)

            # Title on top row only
            if row_idx == 0:
                unit = task_units[task]
                ax.set_title(f"{task_display_names[task]} {unit}", fontsize=LEGEND_FONTSIZE+4)

            # X-axis label removed from individual subplots (added as fig.text below)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    # Add "Target Value" label centered between columns 2 and 3 at the bottom
    col2_right = axes[1, 1].get_position().x1
    col3_left = axes[1, 2].get_position().x0
    center_x = (col2_right + col3_left) / 2
    fig.text(center_x, 0.02, 'Target Value', fontsize=LEGEND_FONTSIZE+4, ha='center', va='center')

    # Add "Random" and "Geographic" labels to the left, then "Residual" further right
    top_center = (axes[0, 0].get_position().y0 + axes[0, 0].get_position().y1) / 2
    bottom_center = (axes[1, 0].get_position().y0 + axes[1, 0].get_position().y1) / 2
    fig.text(0.01, top_center, 'Random', fontsize=LEGEND_FONTSIZE+4, rotation=90, ha='center', va='center')
    fig.text(0.01, bottom_center, 'Geographic', fontsize=LEGEND_FONTSIZE+4, rotation=90, ha='center', va='center')

    # Add "Residual" label centered between the two rows, to the right of "Random"/"Geographic"
    top_row_bottom = axes[0, 0].get_position().y0
    bottom_row_top = axes[1, 0].get_position().y1
    center_y = (top_row_bottom + bottom_row_top) / 2
    left_edge = axes[0, 0].get_position().x0
    fig.text(left_edge - 0.05, center_y, 'Residual', fontsize=LEGEND_FONTSIZE+4, rotation=90, ha='right', va='center')

    # Add "Percentage (%)" label on the right, centered between the two rows
    right_edge = axes[0, -1].get_position().x1
    fig.text(right_edge + 0.03, center_y, 'Percentage (%)', fontsize=LEGEND_FONTSIZE+4, rotation=270, ha='left', va='center', color='gray')

    # Shared legend at top
    if JT_only:
        legend_handles = [bp_jt['boxes'][0]]
        legend_labels = ['JT']
    elif JT_TTT_MMR_only:
        legend_handles = [bp_jt['boxes'][0], bp_ttt['boxes'][0]]
        legend_labels = ['JT', 'TTT-MMR']
    else:
        legend_handles = [bp_jt['boxes'][0], bp_ttt['boxes'][0], bp_geo['boxes'][0]]
        legend_labels = ['JT', 'TTT-MMR', 'TTT-MMR-Geo']
    fig.legend(legend_handles, legend_labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=len(legend_labels), fontsize=LEGEND_FONTSIZE+4, frameon=False,
               handletextpad=0.3, handlelength=1, columnspacing=1)

    if JT_only:
        suffix = '_JT_only'
    elif JT_TTT_MMR_only:
        suffix = '_JT_TTT_MMR_only'
    else:
        suffix = ''
    fname = f'results_figures/residual_distribution_combined{suffix}.pdf'
    fname_png = f'results_figures/residual_distribution_combined{suffix}.png'
    plt.savefig(fname, dpi=300)
    plt.savefig(fname_png, dpi=300)
    plt.close()
    print(f"Saved {fname} and {fname_png}")

def plot_stratified_results(JT_only=False, JT_TTT_MMR_only=False):
    """Combined stratified results: regression residual distributions (cols 0-3) and species AP (col 4).

    Top row: random_test. Bottom row: geographic_test.
    """
    regression_tasks = ['biomass', 'soil_nitrogen', 'soil_organic_carbon', 'soil_pH']
    splits = ['random_test', 'geographic_test']

    task_units = {
        'biomass': '(Mg/ha)',
        'soil_nitrogen': '(g/kg)',
        'soil_organic_carbon': '(g/kg)',
        'soil_pH': '',
    }

    task_display_names = {
        'biomass': 'Biomass',
        'soil_nitrogen': 'Soil N',
        'soil_organic_carbon': 'Soil OC',
        'soil_pH': 'Soil pH',
    }

    task_bins = {
        'biomass': {
            'random_test': np.array([0, 200, 400, 600]),
            'geographic_test': np.array([0, 200, 400, 600]),
        },
        'soil_nitrogen': {
            'random_test': np.array([0, 10, 20, 30]),
            'geographic_test': np.array([0, 5, 10, 20]),
        },
        'soil_organic_carbon': {
            'random_test': np.array([0, 200, 400, 600]),
            'geographic_test': np.array([0, 100, 200, 300]),
        },
        'soil_pH': {
            'random_test': np.array([3, 4, 5, 8]),
            'geographic_test': np.array([3, 5, 7, 9]),
        }
    }

    models = architectures_plots

    all_data = {}
    for task in regression_tasks:
        print(f"Loading data for {task}...")
        all_data[task] = _load_residuals_predictions_data(task, models, splits, flatten=True)

    print("Loading data for species...")
    species_data = _load_residuals_predictions_data('species', models, splits, flatten=False)

    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import FormatStrFormatter
    frequency_bar_color = 'gray'
    frequency_bar_alpha = 0.5
    frequency_label_color = '#666666'
    wspace_inner = 0.60
    gap_ratio = 0.8
    fig = plt.figure(figsize=(COL_WIDTH, 2))
    outer = GridSpec(
        1, 3, figure=fig, width_ratios=[4 + 3 * wspace_inner, gap_ratio, 1],
        left=0.1, right=0.93, top=0.86, bottom=0.2, wspace=0,
    )
    gs_reg = outer[0].subgridspec(2, 4, wspace=wspace_inner, hspace=0.43)
    gs_species = outer[2].subgridspec(2, 1, hspace=0.43)
    axes = np.empty((2, 5), dtype=object)
    for row in range(2):
        for col in range(4):
            axes[row, col] = fig.add_subplot(gs_reg[row, col])
        axes[row, 4] = fig.add_subplot(gs_species[row, 0])

    c_jt = 'red'
    c_ttt = '#1f77b4'
    c_geo = '#ff7f0e'
    median_props = dict(color='black', linewidth=0.5)
    species_group_labels = ['Dominant', 'Occasional', 'Infrequent', 'Rare']
    species_group_sizes = [40, 20, 20, 20]
    regression_x_tick_rotation = 35
    species_x_tick_rotation = 30

    def _center_y_ticklabels(axis):
        for label in axis.get_yticklabels():
            label.set_ha('center')
            label.set_va('center')

    def _boxplot_whisker_min_max(boxplots):
        whisker_ys = []
        for bp in boxplots:
            for whisker in bp['whiskers']:
                whisker_ys.extend(whisker.get_ydata())
        if not whisker_ys:
            return 0.0, 1.0
        return float(min(whisker_ys)), float(max(whisker_ys))

    def _draw_residual_boxplots(ax, plot_data_JT, plot_data_TTT, plot_data_TTT_Geo, indices):
        if JT_only:
            width = 0.5
            bp_jt = ax.boxplot(
                plot_data_JT, positions=indices, widths=width, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=c_jt, edgecolor=c_jt, linewidth=0.5), medianprops=median_props,
                whiskerprops=dict(color=c_jt, linewidth=0.5), capprops=dict(color=c_jt, linewidth=0.5),
            )
            return bp_jt, None, None, [bp_jt]

        if JT_TTT_MMR_only:
            width = 0.3
            offset = 0.2
            bp_jt = ax.boxplot(
                plot_data_JT, positions=indices - offset, widths=width, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=c_jt, edgecolor=c_jt, linewidth=0.5), medianprops=median_props,
                whiskerprops=dict(color=c_jt, linewidth=0.5), capprops=dict(color=c_jt, linewidth=0.5),
            )
            bp_ttt = ax.boxplot(
                plot_data_TTT, positions=indices + offset, widths=width, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=c_ttt, edgecolor=c_ttt, linewidth=0.5), medianprops=median_props,
                whiskerprops=dict(color=c_ttt, linewidth=0.5), capprops=dict(color=c_ttt, linewidth=0.5),
            )
            return bp_jt, bp_ttt, None, [bp_jt, bp_ttt]

        width = 0.25
        offset = 0.3
        bp_jt = ax.boxplot(
            plot_data_JT, positions=indices - offset, widths=width, patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=c_jt, edgecolor=c_jt, linewidth=0.5), medianprops=median_props,
            whiskerprops=dict(color=c_jt, linewidth=0.5), capprops=dict(color=c_jt, linewidth=0.5),
        )
        bp_ttt = ax.boxplot(
            plot_data_TTT, positions=indices, widths=width, patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=c_ttt, edgecolor=c_ttt, linewidth=0.5), medianprops=median_props,
            whiskerprops=dict(color=c_ttt, linewidth=0.5), capprops=dict(color=c_ttt, linewidth=0.5),
        )
        bp_geo = ax.boxplot(
            plot_data_TTT_Geo, positions=indices + offset, widths=width, patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=c_geo, edgecolor=c_geo, linewidth=0.5), medianprops=median_props,
            whiskerprops=dict(color=c_geo, linewidth=0.5), capprops=dict(color=c_geo, linewidth=0.5),
        )
        return bp_jt, bp_ttt, bp_geo, [bp_jt, bp_ttt, bp_geo]

    def _draw_species_boxplots(ax, ap_groups_JT, ap_groups_TTT, ap_groups_TTT_Geo, positions):
        if JT_only:
            width = 0.5
            species_median_props = dict(color='black', linewidth=0.5)
            boxplot_JT = ax.boxplot(
                ap_groups_JT, positions=positions, widths=width,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=c_jt, edgecolor=c_jt, linewidth=0.5), medianprops=species_median_props,
                whiskerprops=dict(color=c_jt, linewidth=0.5), capprops=dict(color=c_jt, linewidth=0.5),
            )
            return boxplot_JT, None, None, [boxplot_JT]

        if JT_TTT_MMR_only:
            width = 0.3
            offset = 0.2
            species_median_props = dict(color='black', linewidth=0.5)
            boxplot_JT = ax.boxplot(
                ap_groups_JT, positions=positions - offset, widths=width,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=c_jt, edgecolor=c_jt, linewidth=0.5), medianprops=species_median_props,
                whiskerprops=dict(color=c_jt, linewidth=0.5), capprops=dict(color=c_jt, linewidth=0.5),
            )
            boxplot_TTT = ax.boxplot(
                ap_groups_TTT, positions=positions + offset, widths=width,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=c_ttt, edgecolor=c_ttt, linewidth=0.5), medianprops=species_median_props,
                whiskerprops=dict(color=c_ttt, linewidth=0.5), capprops=dict(color=c_ttt, linewidth=0.5),
            )
            return boxplot_JT, boxplot_TTT, None, [boxplot_JT, boxplot_TTT]

        width = 0.25
        offset = 0.3
        species_median_props = dict(color='black', linewidth=0.5)
        boxplot_JT = ax.boxplot(
            ap_groups_JT, positions=positions - offset, widths=width,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=c_jt, edgecolor=c_jt, linewidth=0.5), medianprops=species_median_props,
            whiskerprops=dict(color=c_jt, linewidth=0.5), capprops=dict(color=c_jt, linewidth=0.5),
        )
        boxplot_TTT = ax.boxplot(
            ap_groups_TTT, positions=positions, widths=width,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=c_ttt, edgecolor=c_ttt, linewidth=0.5), medianprops=species_median_props,
            whiskerprops=dict(color=c_ttt, linewidth=0.5), capprops=dict(color=c_ttt, linewidth=0.5),
        )
        boxplot_TTT_Geo = ax.boxplot(
            ap_groups_TTT_Geo, positions=positions + offset, widths=width,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=c_geo, edgecolor=c_geo, linewidth=0.5), medianprops=species_median_props,
            whiskerprops=dict(color=c_geo, linewidth=0.5), capprops=dict(color=c_geo, linewidth=0.5),
        )
        return boxplot_JT, boxplot_TTT, boxplot_TTT_Geo, [boxplot_JT, boxplot_TTT, boxplot_TTT_Geo]

    bp_jt = bp_ttt = bp_geo = None

    for col_idx, task in enumerate(regression_tasks):
        for row_idx, split in enumerate(splits):
            ax = axes[row_idx, col_idx]

            targets_TTT_list = []
            targets_TTT_Geo_list = []
            predictions_JT_list = []
            predictions_TTT_list = []
            predictions_TTT_Geo_list = []

            for model in models:
                name_ttt = '_'.join([task, model, 'JT-TTT', str(100)]) + '_'
                name_geo = '_'.join([task, model, 'JT-TTT-Geo', str(100)]) + '_'
                d = all_data[task][split][model]
                if name_ttt in d and 'targets' in d[name_ttt] and name_geo in d and 'targets' in d[name_geo]:
                    targets_TTT_list.append(d[name_ttt]['targets'])
                    targets_TTT_Geo_list.append(d[name_geo]['targets'])
                    predictions_JT_list.append(d[name_ttt]['predictions_JT'])
                    predictions_TTT_list.append(d[name_ttt]['predictions_TTT'])
                    predictions_TTT_Geo_list.append(d[name_geo]['predictions_TTT'])

            if not targets_TTT_list:
                continue

            targets_TTT = np.concatenate(targets_TTT_list)
            residuals_JT = np.concatenate(predictions_JT_list) - targets_TTT
            residuals_TTT = np.concatenate(predictions_TTT_list) - targets_TTT
            residuals_TTT_Geo = np.concatenate(predictions_TTT_Geo_list) - np.concatenate(targets_TTT_Geo_list)

            df = pd.DataFrame({'Target': targets_TTT, 'JT': residuals_JT, 'TTT': residuals_TTT, 'TTT-Geo': residuals_TTT_Geo})

            min_value = np.floor(df['Target'].min())
            max_value = np.ceil(df['Target'].max())

            if task_bins[task][split] is not None:
                bins = np.append(task_bins[task][split], max_value) if max_value > task_bins[task][split][-1] else task_bins[task][split]
            else:
                bin_size = (max_value - min_value) // 5
                bins = np.arange(min_value, max_value + bin_size, bin_size)

            df['Bin'] = pd.cut(df['Target'], bins=bins, include_lowest=True)

            bin_intervals = []
            unique_bins = sorted(df['Bin'].dropna().unique())
            for i in range(len(bins) - 1):
                if i == 0 and unique_bins:
                    interval = pd.Interval(left=unique_bins[0].left, right=bins[i + 1], closed='right')
                else:
                    interval = pd.Interval(left=bins[i], right=bins[i + 1], closed='right')
                bin_intervals.append(interval)

            plot_data_JT = []
            plot_data_TTT = []
            plot_data_TTT_Geo = []

            for bi in bin_intervals:
                subset = df[df['Bin'] == bi]
                plot_data_JT.append(subset['JT'].values if len(subset) > 0 else [])
                plot_data_TTT.append(subset['TTT'].values if len(subset) > 0 else [])
                plot_data_TTT_Geo.append(subset['TTT-Geo'].values if len(subset) > 0 else [])

            indices = np.arange(len(bin_intervals))

            ax2 = ax.twinx()
            counts = np.array([len(df[df['Bin'] == bi]) for bi in bin_intervals])
            percentages = (counts / len(df)) * 100
            ax2.bar(indices, percentages, width=1.0, color=frequency_bar_color, alpha=frequency_bar_alpha, zorder=0)

            bp_jt, bp_ttt, bp_geo, active_bps = _draw_residual_boxplots(
                ax, plot_data_JT, plot_data_TTT, plot_data_TTT_Geo, indices,
            )

            for bp in active_bps:
                for box in bp['boxes']:
                    box.set_zorder(5)
                for whisker in bp['whiskers']:
                    whisker.set_zorder(3)
                for cap in bp['caps']:
                    cap.set_zorder(3)
                for median in bp['medians']:
                    median.set_zorder(6)

            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.7)

            tick_positions = np.arange(len(bins)) - 0.5
            tick_labels_list = [int(x) for x in bins]

            y_min, y_max = _boxplot_whisker_min_max(active_bps)
            set_y_ticks_and_limits(ax, y_min, y_max, step=1, min_spacing=1, lower_tick_above_min=True)
            set_y_ticks_and_limits(
                ax2, 0, float(np.max(percentages)) if len(percentages) else 1.0, step=1, min_spacing=1,
                pin_lower_limit=True,
            )

            ax.tick_params(axis='y', labelsize=LEGEND_FONTSIZE-1, labelrotation=90)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            _center_y_ticklabels(ax)
            ax2.tick_params(axis='y', labelsize=LEGEND_FONTSIZE-1, labelcolor=frequency_label_color, color=frequency_label_color, labelrotation=-90)
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            _center_y_ticklabels(ax2)
            for label in ax2.get_yticklabels():
                label.set_color(frequency_label_color)

            ax.set_xticks(tick_positions)
            x_tick_rotation = 0 if task in ('soil_nitrogen', 'soil_pH') else regression_x_tick_rotation
            ax.set_xticklabels(
                tick_labels_list, fontsize=LEGEND_FONTSIZE-2,
                rotation=x_tick_rotation, ha='center', rotation_mode='anchor',
            )
            ax.tick_params(axis='x', labelsize=LEGEND_FONTSIZE-2)
            ax.set_xlim(-0.5, len(bin_intervals) - 0.5)
            ax2.set_xlim(-0.5, len(bin_intervals) - 0.5)

            if row_idx == 0:
                unit = task_units[task]
                if unit:
                    ax.set_title(f"{task_display_names[task]}\n{unit}", fontsize=AXIS_LABEL_FONTSIZE-1)
                else:
                    ax.set_title(task_display_names[task], fontsize=AXIS_LABEL_FONTSIZE-1)

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    for row_idx, split in enumerate(splits):
        ax = axes[row_idx, 4]

        species_task = 'species'

        predictions_JT_list = []
        predictions_TTT_list = []
        predictions_TTT_Geo_list = []
        targets_list = []
        targets_Geo_list = []

        for model in models:
            name_ttt = '_'.join([species_task, model, 'JT-TTT', str(100)]) + '_'
            name_geo = '_'.join([species_task, model, 'JT-TTT-Geo', str(100)]) + '_'
            d = species_data[split][model]
            if name_ttt not in d or name_geo not in d:
                continue
            predictions_JT_list.append(d[name_ttt]['predictions_JT'])
            predictions_TTT_list.append(d[name_ttt]['predictions_TTT'])
            predictions_TTT_Geo_list.append(d[name_geo]['predictions_TTT'])
            targets_list.append(d[name_ttt]['targets'])
            targets_Geo_list.append(d[name_geo]['targets'])

        if not predictions_JT_list:
            continue

        predictions_JT = np.concatenate(predictions_JT_list, axis=0)
        predictions_TTT = np.concatenate(predictions_TTT_list, axis=0)
        predictions_TTT_Geo = np.concatenate(predictions_TTT_Geo_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        targets_Geo = np.concatenate(targets_Geo_list, axis=0)

        species_occurrences = targets.sum(axis=0)
        species_order = np.argsort(species_occurrences)[::-1]
        predictions_JT = predictions_JT[:, species_order]
        predictions_TTT = predictions_TTT[:, species_order]
        predictions_TTT_Geo = predictions_TTT_Geo[:, species_order]
        targets = targets[:, species_order]
        targets_Geo = targets_Geo[:, species_order]

        num_species = targets.shape[1]
        ap_JT = np.array([average_precision_score(targets[:, i], predictions_JT[:, i]) for i in range(num_species)])
        ap_TTT = np.array([average_precision_score(targets[:, i], predictions_TTT[:, i]) for i in range(num_species)])
        ap_TTT_Geo = np.array([average_precision_score(targets_Geo[:, i], predictions_TTT_Geo[:, i]) for i in range(num_species)])

        ap_groups_JT = []
        ap_groups_TTT = []
        ap_groups_TTT_Geo = []
        species_occurrences_sorted = species_occurrences[species_order]
        total_occurrences = species_occurrences_sorted.sum()
        group_percentages = []

        for g, size in enumerate(species_group_sizes):
            start = sum(species_group_sizes[:g])
            end = start + size
            ap_groups_JT.append(ap_JT[start:end])
            ap_groups_TTT.append(ap_TTT[start:end])
            ap_groups_TTT_Geo.append(ap_TTT_Geo[start:end])
            group_percentages.append(species_occurrences_sorted[start:end].sum() / total_occurrences * 100)

        positions = np.arange(len(species_group_sizes))

        ax2 = ax.twinx()
        ax2.bar(positions, group_percentages, width=1.0, color=frequency_bar_color, alpha=frequency_bar_alpha, zorder=0)

        bp_s_jt, bp_s_ttt, bp_s_geo, active_s_bps = _draw_species_boxplots(ax, ap_groups_JT, ap_groups_TTT, ap_groups_TTT_Geo, positions)

        for bp in active_s_bps:
            for box in bp['boxes']:
                box.set_zorder(5)
            for whisker in bp['whiskers']:
                whisker.set_zorder(3)
            for cap in bp['caps']:
                cap.set_zorder(3)
            for median in bp['medians']:
                median.set_zorder(6)

        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.axhline(1, color='black', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xlim(-0.5, len(positions) - 0.5)
        ax2.set_xlim(-0.5, len(positions) - 0.5)

        # Set minor ticks on the edges of the bars
        ax.set_xticks(np.arange(len(species_group_sizes) + 1) - 0.5, minor=True)
        ax.tick_params(axis='x', which='minor', length=4, width=0.5)

        y_min, y_max = _boxplot_whisker_min_max(active_s_bps)
        set_y_ticks_and_limits(ax, y_min, y_max, lower_tick_above_min=True)
        set_y_ticks_and_limits(
            ax2, 0, float(np.max(group_percentages)) if group_percentages else 1.0, step=1, min_spacing=1,
            pin_lower_limit=True,
        )

        ax.tick_params(axis='y', labelsize=LEGEND_FONTSIZE-1, labelrotation=90)
        _center_y_ticklabels(ax)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.tick_params(axis='y', labelsize=LEGEND_FONTSIZE-1, labelcolor=frequency_label_color, color=frequency_label_color, labelrotation=-90)
        _center_y_ticklabels(ax2)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        for label in ax2.get_yticklabels():
            label.set_color(frequency_label_color)

        if row_idx == 1:
            ax.set_xticklabels(
                species_group_labels, fontsize=LEGEND_FONTSIZE-2,
                rotation=40, ha='right', rotation_mode='anchor',
            )
        else:
            ax.set_xticklabels([])

        ax.tick_params(axis='x', which='major', length=0, labelsize=LEGEND_FONTSIZE-2)

        if row_idx == 0:
            ax.set_title('Species', fontsize=AXIS_LABEL_FONTSIZE-1)

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    for i in range(4):
        col_center = (axes[1, i].get_position().x0 + axes[1, i].get_position().x1) / 2
        fig.text(col_center, 0.08, 'Target value', fontsize=AXIS_LABEL_FONTSIZE-1, ha='center', va='center')

    top_center = (axes[0, 0].get_position().y0 + axes[0, 0].get_position().y1) / 2
    bottom_center = (axes[1, 0].get_position().y0 + axes[1, 0].get_position().y1) / 2
    fig.text(0.015, top_center, 'Random', fontsize=AXIS_LABEL_FONTSIZE, rotation=90, ha='center', va='center')
    fig.text(0.015, bottom_center, 'Geographic', fontsize=AXIS_LABEL_FONTSIZE, rotation=90, ha='center', va='center')

    top_row_bottom = axes[0, 0].get_position().y0
    bottom_row_top = axes[1, 0].get_position().y1
    center_y = (top_row_bottom + bottom_row_top) / 2
    left_edge = axes[0, 0].get_position().x0
    fig.text(left_edge - 0.04, center_y, 'Residual', fontsize=AXIS_LABEL_FONTSIZE, rotation=90, ha='right', va='center')

    species_center_y = (axes[0, 4].get_position().y0 + axes[1, 4].get_position().y1) / 2
    soil_pH_right = axes[0, 3].get_position().x1
    species_left = axes[0, 4].get_position().x0
    ap_label_x = (soil_pH_right + species_left) / 2
    fig.text(ap_label_x, species_center_y, 'Average precision', fontsize=AXIS_LABEL_FONTSIZE, rotation=90, ha='center', va='center')

    species_right = axes[0, 4].get_position().x1
    fig.text(species_right + 0.04, center_y, 'Frequency (%)', fontsize=AXIS_LABEL_FONTSIZE, rotation=270, ha='left', va='center', color=frequency_label_color)

    if JT_only:
        legend_handles = [bp_jt['boxes'][0]]
        legend_labels = ['JT']
    elif JT_TTT_MMR_only:
        legend_handles = [bp_jt['boxes'][0], bp_ttt['boxes'][0]]
        legend_labels = ['JT', 'TTT-MMR']
    else:
        legend_handles = [bp_jt['boxes'][0], bp_ttt['boxes'][0], bp_geo['boxes'][0]]
        legend_labels = ['JT', 'TTT-MMR', 'TTT-MMR-Geo']
    fig.legend(legend_handles, legend_labels,
               loc='upper center', bbox_to_anchor=(0.5, 0.095),
               ncol=len(legend_labels), fontsize=LEGEND_FONTSIZE-1, frameon=False)

    if JT_only:
        suffix = '_JT_only'
    elif JT_TTT_MMR_only:
        suffix = '_JT_TTT_MMR_only'
    else:
        suffix = ''
    fname = f'results_figures/stratified_results{suffix}.pdf'
    fname_png = f'results_figures/stratified_results{suffix}.png'
    plt.savefig(fname, dpi=300)
    plt.savefig(fname_png, dpi=300)
    plt.close()
    print(f"Saved {fname} and {fname_png}")

if __name__ == '__main__':
    # main paper
    plot_rq1_performance('Random', 'FT') # Figure 4
    plot_rq2_performance('FT') # Figure 5
    plot_rq3_performance('FT') # Figure 6
    plot_ttt_improvement() # Figure 7
    tabulate_ttt_ranks_by_model() # Table 5
    plot_stratified_results() # Figure 8

    # appendix
    # plot_rq1_performance('Geographic', 'FT') # Figure A.10
    # plot_rq1_performance('Random', 'LP') # Figure A.11
    # plot_rq1_performance('Geographic', 'LP') # Figure A.12
    # plot_rq2_performance('LP') # Figure A.13
    # plot_rq3_performance('LP') # Figure A.14
    # tabulate_ft_ranked_models_by_task() # Tables A.10-12
    # tabulate_ft_ranks_by_task() # Table A.13
    # tabulate_ft_metrics_by_task() # Table A.14
    # plot_ttt_improvement_normalized() # Figure A.15
    # tabulate_ttt_by_model() # Table A.15
    # tabulate_iteration_numbers() # Table A.16
    # plot_TTT_modality_excluded() # Figure A.16
    # check_TTT_need() # Table A.17
    # tabulate_JT_reconstruction_quality() # Tables A.18-A.22
    # compare_JT_TTT_reconstruction_quality() # Tables A.23-A.27
    # tabulate_results('FT') # Tables A.28-A.32
    # tabulate_TTT_results() # Tables A.33-A.47
    # tabulate_results('LP') # Tables A.48-A.52

    # plot_residuals('biomass', JT_only=True)
    # plot_residuals('biomass', JT_TTT_MMR_only=True)
    # plot_residuals('soil_nitrogen')
    # plot_residuals('soil_organic_carbon')
    # plot_residuals('soil_pH')

    # plot_residuals('biomass')
    # plot_residuals('soil_nitrogen')
    # plot_residuals('soil_organic_carbon')
    # plot_residuals('soil_pH')
    # plot_residuals('species')
    # plot_residuals_species()
    # plot_bce_species()

    # plot_residuals_combined(JT_only=True)
    # plot_residuals_combined(JT_TTT_MMR_only=True)
    # plot_residuals_combined()

    # plot_ap_species(JT_only=True)
    # plot_ap_species(JT_TTT_MMR_only=True)
    # plot_ap_species()
    # plot_ap_species_ungrouped()
    # plot_ap_species_ungrouped_line()

    # plot_stratified_results(JT_only=True)
    # plot_stratified_results(JT_TTT_MMR_only=True)
    # tabulate_TTT_modality_excluded()

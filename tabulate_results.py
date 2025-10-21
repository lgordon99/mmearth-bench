import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
import wandb

entity = utils.read_yaml('config-user.yml')['entity']
project = utils.read_yaml('config-user.yml')['project']
tag = 'pi'
runs = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': [tag]}})
tasks = ['biomass', 'soil_nitrogen', 'soil_organic_carbon', 'soil_pH', 'species']
architectures = ['ConvNeXtV2A', 'ScaleMAE','DINOv3Web', 'DINOv3Sat', 'MPMAE', 'TerraMind', 'CopernicusFM']

def tabulate_results_RQ1_task(task):
    adaptation_mode = 'FT'
    train_percents = [5, 50, 100]
    data = {architecture: {train_percent: np.nan for train_percent in train_percents} for architecture in architectures}

    for architecture in architectures:
        for train_percent in train_percents:
            name = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
            run = next((run for run in runs if run.name.startswith(name)), None)
            metric = 'R2' if task != 'species' else 'MAP'
            random_test_metric = run.summary_metrics.get(f'Random test {metric}') if run else np.nan
            data[architecture][train_percent] = random_test_metric

    # Create DataFrame with architectures as rows and train_percents as columns
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index.name = 'Architecture'
    df.columns.name = 'Train %'

    # Use the same rounding for selection and display
    display_decimals = 2
    df_disp = df.round(display_decimals)

    # --- highlight best per column (after rounding) ---
    mask = pd.DataFrame(False, index=df_disp.index, columns=df_disp.columns, dtype=bool)
    for col in df_disp.columns:
        best = df_disp[col].max(skipna=True)
        eq = df_disp[col].eq(best).fillna(False)
        mask[col] = eq

    # Format from the rounded values
    df_fmt = df_disp.apply(lambda col: col.map(lambda x: '--' if pd.isna(x) else f'{x:.{display_decimals}f}'))
    bold = '\\textbf{' + df_fmt + '}'
    df_fmt = df_fmt.where(~mask, bold)

    # Create LaTeX table
    cols = df_fmt.columns.tolist()
    header_line = ' & '.join(['\\textbf{Architecture}'] + [f'\\textbf{{{c}\\%}}' for c in cols]) + r' \\'
    latex = df_fmt.to_latex(index=True,
                            header=False,
                            index_names=False,
                            escape=False,
                            column_format='l' + 'r'*len(cols))
    latex = latex.replace('\\toprule', '\\toprule\n' + header_line + '\n\\midrule', 1)

    caption_metric = r"R$^2$" if task != 'species' else "MAP"
    latex = ("\\begin{table}[ht]\n\\centering\n" +
            latex +
            f"\\caption{{{task.replace('_', ' ').capitalize().replace('ph', 'pH')} random test {caption_metric} by architecture and training percentage}}\n" +
            f"\\label{{tab:{task}_rq1}}\n" +
            "\\end{table}\n")

    return latex

def tabulate_results_RQ2_task(task):
    adaptation_mode = 'FT'
    train_percent = 100  # Use full training data for RQ2

    # Create data structure: {architecture: metric_value}
    data = {architecture: np.nan for architecture in architectures}

    for architecture in architectures:
        name = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
        run = next((run for run in runs if run.name.startswith(name)), None)
        metric = 'R2' if task != 'species' else 'MAP'
        geographic_test_metric = run.summary_metrics.get(f'Geographic test {metric}') if run else np.nan
        data[architecture] = geographic_test_metric

    # Create DataFrame with architectures as rows and single column for geographic test metric
    df = pd.DataFrame.from_dict(data, orient='index', columns=[f'Geographic test {metric}'])
    df.index.name = 'Architecture'

    # Use the same rounding for selection and display
    display_decimals = 2
    df_disp = df.round(display_decimals)

    # --- highlight best value (after rounding) ---
    mask = pd.DataFrame(False, index=df_disp.index, columns=df_disp.columns, dtype=bool)
    best = df_disp.iloc[:, 0].max(skipna=True)
    eq = df_disp.iloc[:, 0].eq(best).fillna(False)
    mask.iloc[:, 0] = eq

    # Format from the rounded values
    df_fmt = df_disp.apply(lambda col: col.map(lambda x: '--' if pd.isna(x) else f'{x:.{display_decimals}f}'))
    bold = '\\textbf{' + df_fmt + '}'
    df_fmt = df_fmt.where(~mask, bold)

    # Create LaTeX table
    cols = df_fmt.columns.tolist()
    header_line = ' & '.join(['\\textbf{Architecture}'] + [f'\\textbf{{{c}}}' for c in cols]) + r' \\'
    latex = df_fmt.to_latex(index=True,
                            header=False,
                            index_names=False,
                            escape=False,
                            column_format='lr')
    latex = latex.replace('\\toprule', '\\toprule\n' + header_line + '\n\\midrule', 1)

    caption_metric = r"R$^2$" if task != 'species' else "MAP"
    latex = ("\\begin{table}[ht]\n\\centering\n" +
            latex +
            f"\\caption{{{task.replace('_', ' ').capitalize().replace('ph', 'pH')} geographic test {caption_metric} by architecture}}\n" +
            f"\\label{{tab:{task}_rq2}}\n" +
            "\\end{table}\n")

    return latex

def tabulate_results_RQ3_task(task):
    adaptation_mode = 'FT'
    train_percent = 100  # Use full training data for RQ3
    architectures = ['TerraMindS2', 'TerraMind', 'CopernicusFMS2', 'CopernicusFM']

    # Create data structure: {architecture: {metric_type: metric_value}}
    data = {architecture: {'Random test': np.nan, 'Geographic test': np.nan} for architecture in architectures}

    for architecture in architectures:
        name = '_'.join([task, architecture, adaptation_mode, str(train_percent)]) + '_'
        run = next((run for run in runs if run.name.startswith(name)), None)
        metric = 'R2' if task != 'species' else 'MAP'

        if run:
            random_test_metric = run.summary_metrics.get(f'Random test {metric}', np.nan)
            geographic_test_metric = run.summary_metrics.get(f'Geographic test {metric}', np.nan)
        else:
            random_test_metric = np.nan
            geographic_test_metric = np.nan

        data[architecture]['Random test'] = random_test_metric
        data[architecture]['Geographic test'] = geographic_test_metric

    # Create DataFrame with architectures as rows and test types as columns
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index.name = 'Architecture'

    # Use the same rounding for selection and display
    display_decimals = 2
    df_disp = df.round(display_decimals)

    # --- highlight best per column (after rounding) ---
    mask = pd.DataFrame(False, index=df_disp.index, columns=df_disp.columns, dtype=bool)
    for col in df_disp.columns:
        best = df_disp[col].max(skipna=True)
        eq = df_disp[col].eq(best).fillna(False)
        mask[col] = eq

    # Format from the rounded values
    df_fmt = df_disp.apply(lambda col: col.map(lambda x: '--' if pd.isna(x) else f'{x:.{display_decimals}f}'))
    bold = '\\textbf{' + df_fmt + '}'
    df_fmt = df_fmt.where(~mask, bold)

    # Create LaTeX table
    cols = df_fmt.columns.tolist()
    header_line = ' & '.join(['\\textbf{Architecture}'] + [f'\\textbf{{{c}}}' for c in cols]) + r' \\'
    latex = df_fmt.to_latex(index=True,
                            header=False,
                            index_names=False,
                            escape=False,
                            column_format='l' + 'r'*len(cols))
    latex = latex.replace('\\toprule', '\\toprule\n' + header_line + '\n\\midrule', 1)

    caption_metric = r"R$^2$" if task != 'species' else "MAP"
    latex = ("\\begin{table}[ht]\n\\centering\n" +
            latex +
            f"\\caption{{{task.replace('_', ' ').capitalize().replace('ph', 'pH')} test {caption_metric} by architecture and test split}}\n" +
            f"\\label{{tab:{task}_rq3}}\n" +
            "\\end{table}\n")

    return latex

# def tabulate_results_RQ1_task(task):
#     adaptation_modes = ['FT']
#     splits = ['Random']
#     data = {split: {mode: {architecture: np.nan for architecture in architectures} for mode in adaptation_modes} for split in splits}

#     for architecture in architectures:
#         for adaptation_mode in adaptation_modes:
#             name = '_'.join([task, architecture, adaptation_mode])
#             run = next((run for run in runs if run.name.startswith(name)), None)
#             metric = 'R2' if task != 'species' else 'MAP'
#             random_test_metric = run.summary_metrics.get(f'Random test {metric}') if run else np.nan
#             geographic_test_metric = run.summary_metrics.get(f'Geographic test {metric}') if run else np.nan
#             data['Random'][adaptation_mode][architecture] = random_test_metric
#             data['Geographic'][adaptation_mode][architecture] = geographic_test_metric

#     random_df = pd.DataFrame.from_dict(data['Random'], orient='index')[architectures].reindex(adaptation_modes)
#     geographic_df = pd.DataFrame.from_dict(data['Geographic'], orient='index')[architectures].reindex(adaptation_modes)
#     df = pd.concat({'Random': random_df, 'Geographic': geographic_df}, axis=0)
#     df.index.set_names(['Split', 'Adaptation mode'], inplace=True)
#     df = df.rename(index=lambda s: s.capitalize().replace('_', ' ')
#                    .replace('Standard', 'FT').replace('Lp', 'LP').replace('Ttt', 'TTT').replace('Mt3', 'MT3').replace('mjt', 'MJT').replace('jt', 'JT').replace('mt3', 'MT3').replace('Sln', 'SLN'),
#                    level='Adaptation mode')

#     # Use the same rounding for selection and display
#     display_decimals = 2
#     df_disp = df.round(display_decimals)

#     # --- highlight best per column within each split (after rounding) ---
#     mask = pd.DataFrame(False, index=df_disp.index, columns=df_disp.columns, dtype=bool)
#     for split in ['Random', 'Geographic']:
#         sub = df_disp.loc[split]
#         best = sub.max(axis=0, skipna=True)
#         eq = sub.eq(best).fillna(False)
#         mask.loc[(split, slice(None)), :] = eq.to_numpy()

#     # Format from the rounded values
#     df_fmt = df_disp.apply(lambda col: col.map(lambda x: '--' if pd.isna(x) else f'{x:.{display_decimals}f}'))
#     bold = '\\textbf{' + df_fmt + '}'
#     df_fmt = df_fmt.where(~mask, bold)
#     cols = df_fmt.columns.tolist()
#     header_line = ' & '.join(['\\textbf{Split}', '\\textbf{Adaptation mode}'] +
#                              [f'\\textbf{{{c}}}' for c in cols]) + r' \\'
#     latex = df_fmt.to_latex(index=True,
#                             header=False,
#                             index_names=False,
#                             multirow=True,
#                             multicolumn=False,
#                             escape=False,
#                             column_format='cl' + 'r'*len(cols))
#     latex = latex.replace('\\toprule', '\\toprule\n' + header_line + '\n\\midrule', 1)
#     latex = latex.replace(r'\multirow[t]{', r'\multirow[c]{')
#     caption_metric = r"R$^2$" if task != 'species' else "MAP"
#     latex = ("\\begin{table}[ht]\n\\centering\n" +
#             latex +
#             f"\\caption{{{task.replace('_', ' ').capitalize().replace('ph', 'pH')} test {caption_metric}}}\n" +
#             f"\\label{{tab:{task}_r2}}\n" +
#             "\\end{table}\n")

#     return latex

def plot_rq1_performance():
    adaptation_mode = 'FT'
    train_percents = [5, 50, 100]

    plt.figure(figsize=(10, 6)) # sets figure size
    colors = plt.cm.tab10(np.linspace(0, 1, len(architectures))) # defines colors for each architecture

    # Collect data for all tasks
    all_data = []

    for task in tasks:
        metric = 'R2' if task != 'species' else 'MAP'
        metric_name = f'Random test {metric}'

        for architecture in architectures:
            for train_percent in train_percents:
                run = next((run for run in runs if run.name.startswith(f'{task}_{architecture}_{adaptation_mode}_{train_percent}_')), None)
                random_test_metric = run.summary_metrics.get(metric_name) if run else np.nan

                if random_test_metric is not None and not np.isnan(random_test_metric): # checks if the metric is not None and not NaN
                    all_data.append({'task': task, 'architecture': architecture, 'train_percent': train_percent, 'metric': random_test_metric})

    df = pd.DataFrame(all_data) # converts the data to a DataFrame

    # Calculate relative performance (delta) using ConvNeXtV2A as baseline
    df_delta = df.copy()

    for task in tasks:
        for train_percent in train_percents:
            baseline_data = df[(df['task'] == task) & (df['architecture'] == 'ConvNeXtV2A') & (df['train_percent'] == train_percent)] # gets the baseline performance for the task and train percentage

            if not baseline_data.empty:
                baseline_metric = baseline_data['metric'].iloc[0] # value of the test metric for the baseline architecture
                mask = (df_delta['task'] == task) & (df_delta['train_percent'] == train_percent)
                df_delta.loc[mask, 'metric'] = df_delta.loc[mask, 'metric'] - baseline_metric # subtracts the baseline metric from the test metric

    # Calculate overall y-axis range across all tasks
    all_metrics = df_delta['metric'].dropna()
    y_min = all_metrics.min()
    y_max = all_metrics.max()
    y_margin = (y_max - y_min) * 0.1  # Add 10% margin
    y_range = [y_min - y_margin, y_max + y_margin]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5)) # creates a figure with 1 row and 5 columns, one per task

    for i, task in enumerate(tasks):
        ax = axes[i] # gets the axis for the current task
        task_data = df_delta[df_delta['task'] == task]

        # Plot each architecture
        for j, architecture in enumerate(architectures):
            architecture_data = task_data[task_data['architecture'] == architecture]

            if not architecture_data.empty:
                architecture_data_sorted = architecture_data.sort_values('train_percent') # sort by train_percent to ensure proper line connection

                # Set style based on architecture
                if architecture == 'ConvNeXtV2A':
                    color = 'black'
                    linestyle = '-'
                    marker = 'o'
                elif architecture == 'MPMAE':
                    color = colors[j]
                    linestyle = '--'
                    marker = 's'  # square
                elif architecture == 'TerraMind':
                    color = colors[j]
                    linestyle = '--'
                    marker = '^'  # triangle up
                elif architecture == 'CopernicusFM':
                    color = colors[j]
                    linestyle = '--'
                    marker = 'D'  # diamond
                else:
                    color = colors[j]
                    linestyle = '--'
                    marker = 'o'  # circle

                ax.plot(architecture_data_sorted['train_percent'], architecture_data_sorted['metric'],
                        marker, color=color, linestyle=linestyle, label=architecture, markersize=6, linewidth=2, alpha=0.8)

        if i == 2:
            ax.set_xlabel('Training Data %')

        metric = 'R2' if task != 'species' else 'MAP'
        ax.set_ylabel(f'Δ {metric}')
        ax.set_title(f'{task.replace("_", " ").capitalize().replace("ph", "pH")}')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(train_percents)
        ax.set_ylim(y_range)  # Set consistent y-axis range across all tasks

    fig.suptitle('Change in Random Test Metric from Randomly Initialized ConvNeXtV2A', fontsize=16, y=0.98) # adds main title
    fig.legend(architectures, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(architectures), fontsize=10) # adds legend below all subplots
    plt.tight_layout()
    plt.subplots_adjust(top=0.85) # makes room for the main title
    plt.savefig(f'RQ1_plot_{tag}.png', dpi=300, bbox_inches='tight')

def plot_rq1_absolute_performance():
    adaptation_mode = 'FT'
    train_percents = [5, 50, 100]

    plt.figure(figsize=(10, 6)) # sets figure size
    colors = plt.cm.tab10(np.linspace(0, 1, len(architectures))) # defines colors for each architecture

    # Collect data for all tasks
    all_data = []

    for task in tasks:
        metric = 'R2' if task != 'species' else 'MAP'
        metric_name = f'Random test {metric}'

        for architecture in architectures:
            for train_percent in train_percents:
                run = next((run for run in runs if run.name.startswith(f'{task}_{architecture}_{adaptation_mode}_{train_percent}_')), None)
                random_test_metric = run.summary_metrics.get(metric_name) if run else np.nan

                if random_test_metric is not None and not np.isnan(random_test_metric): # checks if the metric is not None and not NaN
                    all_data.append({'task': task, 'architecture': architecture, 'train_percent': train_percent, 'metric': random_test_metric})

    df = pd.DataFrame(all_data) # converts the data to a DataFrame

    # Calculate overall y-axis range across all tasks
    all_metrics = df['metric'].dropna()
    y_min = all_metrics.min()
    y_max = all_metrics.max()
    y_margin = (y_max - y_min) * 0.1  # Add 10% margin
    y_range = [y_min - y_margin, 1.05]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5)) # creates a figure with 1 row and 5 columns, one per task

    for i, task in enumerate(tasks):
        ax = axes[i] # gets the axis for the current task
        task_data = df[df['task'] == task]

        # Plot each architecture
        for j, architecture in enumerate(architectures):
            architecture_data = task_data[task_data['architecture'] == architecture]

            if not architecture_data.empty:
                architecture_data_sorted = architecture_data.sort_values('train_percent') # sort by train_percent to ensure proper line connection

                # Set style based on architecture
                if architecture == 'ConvNeXtV2A':
                    color = 'black'
                    linestyle = '-'
                    marker = 'o'
                elif architecture == 'MPMAE':
                    color = colors[j]
                    linestyle = '--'
                    marker = 's'  # square
                elif architecture == 'TerraMind':
                    color = colors[j]
                    linestyle = '--'
                    marker = '^'  # triangle up
                elif architecture == 'CopernicusFM':
                    color = colors[j]
                    linestyle = '--'
                    marker = 'D'  # diamond
                else:
                    color = colors[j]
                    linestyle = '--'
                    marker = 'o'  # circle

                ax.plot(architecture_data_sorted['train_percent'], architecture_data_sorted['metric'],
                        marker, color=color, linestyle=linestyle, label=architecture, markersize=6, linewidth=2, alpha=0.8)

        if i == 2:
            ax.set_xlabel('Training Data %')

        metric = 'R2' if task != 'species' else 'MAP'
        ax.set_ylabel(f'{metric}')
        ax.set_title(f'{task.replace("_", " ").capitalize().replace("ph", "pH")}')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(train_percents)

        # Set y-axis range: 0-1 for species, consistent range for others
        if task == 'species':
            ax.set_ylim([0, 1.05])
        else:
            ax.set_ylim(y_range)  # Set consistent y-axis range across other tasks

    fig.suptitle('Random Test Metric Performance by Architecture and Training Data Percentage', fontsize=16, y=0.98) # adds main title
    fig.legend(architectures, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(architectures), fontsize=10) # adds legend below all subplots
    plt.tight_layout()
    plt.subplots_adjust(top=0.85) # makes room for the main title
    plt.savefig(f'RQ1_absolute_plot_{tag}.png', dpi=300, bbox_inches='tight')

def tabulate_results(rq_number):
    with open(f'latex_{tag}_RQ{rq_number}.tex', 'w') as file:
        file.write('\n'.join([globals()[f'tabulate_results_RQ{rq_number}_task'](task) for task in tasks]))

if __name__ == '__main__':
    tabulate_results(1)
    tabulate_results(2)
    tabulate_results(3)
    plot_rq1_performance()
    plot_rq1_absolute_performance()

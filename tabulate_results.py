import numpy as np
import pandas as pd
import utils
import wandb

entity = utils.read_yaml('config-user.yml')['entity']
project = utils.read_yaml('config-user.yml')['project']
runs = wandb.Api().runs(f'{entity}/{project}', filters={'tags': {'$in': ['kappa']}})
architectures = ['DINOv3Web', 'DINOv3Sat', 'MPMAE', 'TerraMind']
adaptation_modes = ['standard', 'multimodal', 'joint_training', 'multimodal_joint_training', 'ttt', 'mt3']
splits = ['Random', 'Geographic']

def tabulate_results_task(task):
    data = {split: {mode: {architecture: np.nan for architecture in architectures} for mode in adaptation_modes} for split in splits}

    for architecture in architectures:
        for adaptation_mode in adaptation_modes:
            name = '_'.join([task, architecture, adaptation_mode])
            run = next((run for run in runs if run.name.startswith(name)), None)
            metric = 'RMSE' if task != 'species' else 'MAP'
            random_test_metric = run.summary_metrics.get(f'Random test {metric}') if run else np.nan
            geographic_test_metric = run.summary_metrics.get(f'Geographic test {metric}') if run else np.nan
            data['Random'][adaptation_mode][architecture] = random_test_metric
            data['Geographic'][adaptation_mode][architecture] = geographic_test_metric

    random_df = pd.DataFrame.from_dict(data['Random'], orient='index')[architectures].reindex(adaptation_modes)
    geographic_df = pd.DataFrame.from_dict(data['Geographic'], orient='index')[architectures].reindex(adaptation_modes)

    df = pd.concat({'Random': random_df, 'Geographic': geographic_df}, axis=0)
    df.index.set_names(['Split', 'Adaptation mode'], inplace=True)
    df = df.rename(index=lambda str: str.capitalize().replace('_', ' ').replace('Ttt', 'TTT').replace('Mt3', 'MT3'), level='Adaptation mode')
    cols = df.columns.tolist()
    header_line = ' & '.join(['\\textbf{Split}', '\\textbf{Adaptation mode}'] + [f'\\textbf{{{c}}}' for c in cols]) + r' \\'

    latex = df.to_latex(na_rep='--',
                        float_format=lambda x: f'{x:.2f}',
                        index=True,
                        header=False,          # we provide our own header
                        index_names=False,     # <-- prevent pandas from adding the "Split  Adaptation mode" row
                        multirow=True,         # merged "Random/Geographic"
                        multicolumn=False,     # no "Architecture" banner
                        escape=False,
                        column_format='cl' + 'r'*len(cols))

    latex = latex.replace('\\toprule', '\\toprule\n' + header_line + '\n\\midrule', 1)
    latex = latex.replace(r'\multirow[t]{', r'\multirow[c]{')
    latex = ("\\begin{table}[ht]\n\\centering\n" +
            latex +
            f"\\caption{{{task.replace('_', ' ').capitalize().replace('ph', 'pH')} test RMSE}}\n" +
            f"\\label{{tab:{task}_rmse}}\n" +
            "\\end{table}\n")

    return latex

def tabulate_results():
    tasks = ['biomass', 'soil_nitrogen', 'soil_organic_carbon', 'soil_pH', 'species']
    latex = '\n'.join([tabulate_results_task(task) for task in tasks])

    with open('latex.tex', 'w') as file:
        file.write(latex)

if __name__ == '__main__':
    tabulate_results()

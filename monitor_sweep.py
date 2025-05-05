# ============================================== IMPORTS ============================================== #

from sys import argv
import pandas as pd
import wandb

# ============================================== FUNCTIONS ============================================== #

def monitor_sweep(entity, project, sweep_id):
    wandb_api = wandb.Api()
    project = wandb_api.project(project, entity=entity)
    sweeps = project.sweeps()

    for sweep in sweeps:
        if sweep.id == sweep_id:
            print(sweep.name)
            runs = sweep.runs
            sweep_config = sweep.config
            hyperparameter_data = {key: {'max': max([float(num) for num in value['values'] if num != 0]), 'min': min([float(num) for num in value['values'] if num != 0])} for key, value in sweep_config['parameters'].items()}
            print(hyperparameter_data)
            hyperparameters = hyperparameter_data.keys()
            all_run_data = []

            for run in runs:
                if run.state != 'finished':
                    print('Not all runs are finished')
                    exit()

                run_data = {'name': run.name,
                            'state': run.state,
                            'Val RMSE': round(run.history(keys=['Val RMSE'])['Val RMSE'].min(), 5),
                            'Test RMSE': round(run.summary_metrics['Test RMSE'], 5)}
                run_config = run.config
                run_hyperparameter_data = {hyperparameter: run_config[hyperparameter] for hyperparameter in hyperparameters}
                run_data = {**run_data, **run_hyperparameter_data}
                all_run_data.append(run_data)

            df = pd.DataFrame(all_run_data).sort_values(by='Val RMSE')
            best_run_hyperpameter_data = {hyperparameter: df.iloc[0][hyperparameter] for hyperparameter in hyperparameters}
            print(df.iloc[0])

            for hyperparameter in hyperparameters:
                hyperparameter_fine = True

                if best_run_hyperpameter_data[hyperparameter] == hyperparameter_data[hyperparameter]['max']:
                    print(f'{hyperparameter} is at max value')
                    hyperparameter_fine = False
                elif best_run_hyperpameter_data[hyperparameter] == hyperparameter_data[hyperparameter]['min']:
                    print(f'{hyperparameter} is at min value')
                    hyperparameter_fine = False

                if not hyperparameter_fine:
                    print(f'Use range {[float(best_run_hyperpameter_data[hyperparameter] * i) for i in [0.1, 1, 10]]}')

            break

if __name__ == '__main__':
    monitor_sweep(*argv[1].split('/'))

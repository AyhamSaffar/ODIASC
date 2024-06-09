import concurrent.futures as cf
import numpy as np
import pandas as pd
import pathlib as pl
from sklearn.metrics import precision_recall_curve, auc, recall_score, precision_score, f1_score
from dbscan import DBSCAN
from utils.simulators import Steel_APT_Dataset
from utils.models import models, LUNOT, LUNOT_finetuned
import time
from pythresh.thresholds.karch import KARCH

DATA_PATH = pl.Path(__file__).resolve().parent.joinpath(r'Analysis\Sensitivity Results.xlsx')

def find_optimal_DBScan_params(no_cluster_dataset: pd.DataFrame, max_min_samples: int = 30, eps_resolution: float = 0.05) -> dict[int: float]:
    '''
    finds the maximum eps with zero DBScan clusters for each min_samples under max_min_samples
    returns a dict of {min sample: optimal eps}
    '''
    clusterless_datapoints = no_cluster_dataset[['X', 'Y', 'Z']]
    min_samples_values = np.arange(2, max_min_samples+1)
    eps = eps_resolution
    max_eps = dict()
    for min_samples in min_samples_values:
        while True:
            labels, _ = DBSCAN(clusterless_datapoints, eps=eps, min_samples=min_samples)
            num_clusters = max(labels) + 1
            if num_clusters == 0:
                eps += eps_resolution
            else:
                eps = max(eps_resolution, eps-eps_resolution)
                max_eps[min_samples] = np.round(eps, 2)
                break
    return max_eps

def find_DBScan_stats(cluster_dataset: pd.DataFrame, min_samples_to_eps: dict[int: float]) -> pd.DataFrame:
    '''
    Returns precision & recall arrays from DBSCAN clustering of cluster_dataset with min_samples_to_eps parameters
    Both arrays are sorted in ascending order of recall array
    '''
    results = pd.DataFrame(index=list(min_samples_to_eps.keys()))
    results['eps'] = list(min_samples_to_eps.values())
    for min_samples, eps in min_samples_to_eps.items():
        numbered_labels, _ = DBSCAN(cluster_dataset[['X', 'Y', 'Z']], eps=eps, min_samples=min_samples)
        is_cluster_labels = numbered_labels != -1
        results.loc[min_samples, 'cluster count'] = numbered_labels.max() + 1
        results.loc[min_samples, 'precision'] = precision_score(cluster_dataset['is cluster'], is_cluster_labels, zero_division=0.0)
        results.loc[min_samples, 'recall'] = recall_score(cluster_dataset['is cluster'], is_cluster_labels)
    return results.sort_values(by='recall')

def find_model_PRs(cluster_dataset: pd.DataFrame, model_labels: np.ndarray) -> list[np.ndarray, np.ndarray]:
    precisions, recalls, thresholds = precision_recall_curve(cluster_dataset['is cluster'], model_labels)
    return precisions, recalls

def find_AUPRC(precisions: np.ndarray, recalls: np.ndarray) -> float:
    return auc(x=recalls, y=precisions)

def find_P80R(precisions: np.ndarray, recalls: np.ndarray) -> float:
    if np.max(recalls) < 0.8:
        return 0
    index_80_recall = np.abs(recalls-0.8).argmin()
    return precisions[index_80_recall]

def find_best_guess_DBSCAN_f1_score(DBScan_stats: pd.DataFrame) -> float:
    precisions = DBScan_stats['precision'].to_numpy()
    recalls = DBScan_stats['recall'].to_numpy()
    F_scores = 2*precisions*recalls/(precisions+recalls+1E-5)
    return np.sort(F_scores)[-5:].mean()

def find_DBSCAN_max_clusters_f1_score(DBScan_stats: pd.DataFrame, cluster_dataset: pd.DataFrame) -> float:
    indexes = np.nonzero(DBScan_stats['cluster count'].to_numpy() == DBScan_stats['cluster count'].max())[0]
    f1_scores = []
    for i in indexes:
        numbered_labels, _ = DBSCAN(cluster_dataset[['X', 'Y', 'Z']], eps=DBScan_stats['eps'].iloc[i], min_samples=DBScan_stats.index[i])
        is_cluster_labels = numbered_labels != -1
        f1_scores.append(f1_score(cluster_dataset['is cluster'], is_cluster_labels))
    return sum(f1_scores)/len(f1_scores)

def find_DBSCAN_mode_clusters_f1_score(DBScan_stats: pd.DataFrame, cluster_dataset: pd.DataFrame) -> float:
    indexes = np.nonzero(DBScan_stats['cluster count'].to_numpy() == DBScan_stats['cluster count'].mode()[0])[0]
    f1_scores = []
    for i in indexes:
        numbered_labels, _ = DBSCAN(cluster_dataset[['X', 'Y', 'Z']], eps=DBScan_stats['eps'].iloc[i], min_samples=DBScan_stats.index[i])
        is_cluster_labels = numbered_labels != -1
        f1_scores.append(f1_score(cluster_dataset['is cluster'], is_cluster_labels))
    return sum(f1_scores)/len(f1_scores)

def find_best_guess_model_f1_score(cluster_dataset: pd.DataFrame, model_labels: np.ndarray, thresholder) -> float:
    normalized_model_labels = model_labels - model_labels.min() / (model_labels.max() - model_labels.min())
    predicted_model_labels = thresholder().eval(normalized_model_labels)
    return f1_score(y_true=cluster_dataset['is cluster'], y_pred=predicted_model_labels)

def get_predictivities(cluster_density: int, cluster_size: int, optimal_DBSCAN_params: dict[int: float]) -> list[int, int, dict[str: float]]:
    print(f'Calculating results for cluster density of {cluster_density: <3} with {cluster_size: <2} atom clusters')

    results = dict()

    dataset = Steel_APT_Dataset(
        unit_cells_per_side=50,
        cluster_relative_density=cluster_density,
        cluster_atom_counts=np.full(shape=500//cluster_size, fill_value=cluster_size),
    ) #~250 cluster points & 2500 non cluster points
    cluster_data = dataset.data[dataset.data['Element']!='Fe']
    clusterless_data = dataset.DBScan_baseline

    DBSCAN_stats = find_DBScan_stats(cluster_data, optimal_DBSCAN_params)
    results['DBSCAN_P80R'] = find_P80R(DBSCAN_stats['precision'].to_numpy(), DBSCAN_stats['recall'].to_numpy())
    results['DBSCAN_best_guess_f1'] = find_best_guess_DBSCAN_f1_score(DBSCAN_stats)
    results['DBSCAN_max_clusters_f1'] = find_DBSCAN_max_clusters_f1_score(DBSCAN_stats, cluster_data)
    results['DBSCAN_mode_clusters_f1'] = find_DBSCAN_mode_clusters_f1_score(DBSCAN_stats, cluster_data)

    n_neighbours = 10
    for model in [model for model in models if model.short_name != 'LUNAR']:
        model_name = model.short_name if isinstance(model, (LUNOT, LUNOT_finetuned)) else f'{model.short_name}_{n_neighbours}K'
        model_labels = model.get_outputs(cluster_data, clusterless_data, n_neighbours)
        precisions, recalls = find_model_PRs(cluster_data, model_labels)
        results[f'{model_name}_AUPRC'] = find_AUPRC(precisions, recalls)
        results[f'{model_name}_P80R'] = find_P80R(precisions, recalls)
        results[f'{model_name}_KARCH_f1'] = find_best_guess_model_f1_score(cluster_data, model_labels, thresholder=KARCH)

    return cluster_density, cluster_size, results

if __name__ == '__main__':
    
    clusterless_data = Steel_APT_Dataset(unit_cells_per_side=50, cluster_relative_density=10,
                            cluster_atom_counts=np.full(shape=10, fill_value=50)).DBScan_baseline
    optimal_DBSCAN_params = find_optimal_DBScan_params(clusterless_data, eps_resolution=0.01)

    _, _, results = get_predictivities(cluster_density=1, cluster_size=1, optimal_DBSCAN_params=optimal_DBSCAN_params) #temporary to check which measurements are returned
    result_dataframes = {measurement: pd.DataFrame() for measurement in results.keys()}

    cluster_densities, cluster_sizes = np.meshgrid(range(50, 801, 50), range(1, 51, 1), indexing='ij')
    cluster_densities, cluster_sizes = cluster_densities.flatten(), cluster_sizes.flatten()

    start = time.perf_counter()

    futures = []
    with cf.ProcessPoolExecutor() as executor:
        for density, size in np.column_stack([cluster_densities, cluster_sizes]):
            futures.append(executor.submit(get_predictivities, density, size, optimal_DBSCAN_params))

        for future in cf.as_completed(futures):
            cluster_density, cluster_size, results = future.result()
            for measurement in results.keys():
                result_dataframes[measurement].loc[cluster_size, cluster_density] = results[measurement]    

    end = time.perf_counter()
    print(f'##### {len(cluster_sizes):.0f} datasets tested in {(end-start)//60:.0f}m {(end-start)%60:.0f}s #####')

    kwargs = {'mode': 'a', 'if_sheet_exists': 'replace'} if DATA_PATH.is_file() else {'mode': 'w'}
    with pd.ExcelWriter(DATA_PATH, **kwargs) as writer:
        for measurement, dataframe in result_dataframes.items():
            dataframe.sort_index().to_excel(writer, sheet_name=measurement)
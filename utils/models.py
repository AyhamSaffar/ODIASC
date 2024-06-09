import pandas as pd
import numpy as np
import pathlib as pl
import torch
from sklearn.neighbors import NearestNeighbors
from utils.my_lunar import LUNAR
from abc import ABC, abstractmethod, abstractproperty

file_path = pl.Path(__file__).resolve().parent
LUNOT_torch_model = torch.jit.load(file_path.joinpath('LUNOT')).eval()
LUNOT_finetuned_torch_model = torch.jit.load(file_path.joinpath('LUNOT_finetuned')).eval()

class Model(ABC):
    '''template for io of cluster detecting models'''

    @abstractmethod
    def get_outputs(cluster_data: pd.DataFrame, clusterless_data: pd.DataFrame, n_neighbours: int) -> np.ndarray:
        ...

    @abstractproperty
    def name(self) -> str:
        ...
    
    @abstractproperty
    def short_name(self) -> str:
        ...

class KNN(Model):
    '''returns the mean distance to the K nearest neighbours'''
    def get_outputs(self, cluster_data: pd.DataFrame, clusterless_data: pd.DataFrame, n_neighbours: int) -> np.ndarray:
        neighbour_finder = NearestNeighbors(n_neighbors=n_neighbours+1).fit(cluster_data[['X', 'Y', 'Z']])
        distances, _ = neighbour_finder.kneighbors(cluster_data[['X', 'Y', 'Z']])
        distances = distances[:, 1:]  # 1st nieghbour is always the point itself so should be skipped
        return -distances.mean(axis=1) # negative so clusters have higher scores

    @property
    def name(self) -> str:
        return 'KNN Clustering'
    
    @property
    def short_name(self) -> str:
        return 'KNN'
    

class LUNOT(Model):
    '''supervised graph neural network trained to detect small low density clusters'''
    def get_outputs(self, cluster_data: pd.DataFrame, clusterless_data: pd.DataFrame, n_neighbours: int) -> np.ndarray:
        neighbour_finder = NearestNeighbors(n_neighbors=21).fit(cluster_data[["X", "Y", "Z"]])
        distances, _ = neighbour_finder.kneighbors(cluster_data[["X", "Y", "Z"]])
        distances = distances[:, 1:]  # 1st nieghbour is the point itself
        return LUNOT_torch_model(torch.tensor(distances, dtype=torch.float32)).detach().numpy()
    
    @property
    def name(self) -> str:
        return 'LUNOT Classifier'
    
    @property
    def short_name(self) -> str:
        return 'LUNOT'


class LUNOT_finetuned(Model):
    '''finetuned supervised graph neural network trained to detect a range of sizes of low density clusters'''
    def get_outputs(self, cluster_data: pd.DataFrame, clusterless_data: pd.DataFrame, n_neighbours: int) -> np.ndarray:
        neighbour_finder = NearestNeighbors(n_neighbors=51).fit(cluster_data[["X", "Y", "Z"]])
        distances, _ = neighbour_finder.kneighbors(cluster_data[["X", "Y", "Z"]])
        distances = distances[:, 1:]  # 1st nieghbour is the point itself
        return LUNOT_finetuned_torch_model(torch.tensor(distances, dtype=torch.float32)).detach().numpy()
    
    @property
    def name(self) -> str:
        return 'Finetuned LUNOT Classifier'

    @property
    def short_name(self) -> str:
        return 'LUNOT_finetuned'

class LUNAR_Ryan(Model):
    '''a confusing implementation of LUNAR to detect clusters. Doesn't really makes sense and consistently performs worse than KNN'''
    def get_outputs(self, cluster_data: pd.DataFrame, clusterless_data: pd.DataFrame, n_neighbours: int) -> np.ndarray:
        model = LUNAR(model_type='RYAN_WEIGHT', n_neighbours=n_neighbours, negative_sampling='MIXED',
                        val_size=0.1, epsilon=0.3, proportion=0.1, n_epochs=50, lr=4E-4, wd=0.1)
        model.fit(cluster_data[['X', 'Y', 'Z']])
        return -model.decision_scores_ #scores are timesed by -1 as lower scores are given to clusters
    
    @property
    def name(self) -> str:
        return 'LUNAR Clustering'

    @property
    def short_name(self) -> str:
        return 'LUNAR_Ryan'

class LUNAR_APT(Model):
    '''an implementation of LUNAR finetuned to detect APT clustering. Still can't really differentatiate clusters from bulk'''
    def get_outputs(self, cluster_data: pd.DataFrame, clusterless_data: pd.DataFrame, n_neighbours: int) -> np.ndarray:
        model = LUNAR(model_type='FINETUNED_SCORE', n_neighbours=n_neighbours, negative_sampling='UNIFORM',
                    val_size=0.1, epsilon=0.3, proportion=0.1, n_epochs=50, lr=4E-4, wd=0.1)
        model.fit(clusterless_data[['X', 'Y', 'Z']])
        return model.decision_function(cluster_data[['X', 'Y', 'Z']])
    
    @property
    def name(self) -> str:
        return 'LUNAR Anomaly Detector'
    
    @property
    def short_name(self) -> str:
        return 'LUNAR'

models = [KNN(), LUNOT(), LUNOT_finetuned(), LUNAR_Ryan(), LUNAR_APT()]
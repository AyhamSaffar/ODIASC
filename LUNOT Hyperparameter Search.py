import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pathlib as pl
import concurrent.futures as cf
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_curve, auc
from utils.simulators import Steel_APT_Dataset
from line_profiler import LineProfiler
import time

device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_inputs(data: pd.DataFrame, k: int = 10) -> np.ndarray:
    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(data[["X", "Y", "Z"]])
    distances, indexes = knn.kneighbors(data[["X", "Y", "Z"]])
    return distances[:, 1:]  # 1st nieghbour is the point itself

class Training_Dataset(Dataset):

    def __init__(self, preprocessed_inputs: np.ndarray, cluster_labels: np.ndarray):
        self.X = preprocessed_inputs
        self.Y = cluster_labels
        self.n_examples = self.Y.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.n_examples

def get_auprc(model: torch.nn.Sequential, X: torch.Tensor, Y: torch.Tensor) -> float:
    Y_pred = model(X).detach().cpu().numpy()
    Y = Y.cpu().numpy()
    precision, recall, _ = precision_recall_curve(Y, Y_pred)
    return auc(x=recall, y=precision)

class LUNOT(nn.Module):
    def __init__(self, k: int, layer_size: int = 128, layer_count: int = 8, dropout: float = 0.0):
        super(LUNOT, self).__init__()

        layers = [nn.Linear(k, layer_size), nn.Dropout(dropout), nn.GELU()]
        for i in range(layer_count - 2):
            layers += [nn.Linear(layer_size, layer_size), nn.Dropout(dropout), nn.GELU()]
            if i % 2 == 0:
                layers += [nn.BatchNorm1d(layer_size, affine=False)]
        layers += [nn.Linear(layer_size, 1), nn.Sigmoid()]
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return torch.squeeze(out, 1)

def test_model(train_data: pd.DataFrame, val_data: pd.DataFrame, lr=0.001, n_neighbours=10, layer_size=32, batch_size=64, layer_count=3, lr_schedule_k=1.0) -> list[list[float], list[float], list[float]]:
    '''
    convenience wrapper for LUNOT hyperparameter search
    returns [losses, train_scores, val_scores]
    '''
    hyperparameters = {key: value for key, value in locals().items() if key not in ['train_data', 'val_data']}

    X_val = preprocess_inputs(val_data, k=n_neighbours)
    Y_val = val_data['is cluster'].to_numpy()
    X_train = preprocess_inputs(train_data, k=n_neighbours)
    Y_train = train_data['is cluster'].to_numpy()
    
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val = torch.tensor(Y_val, dtype=torch.float32).to(device)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)

    training_dataset = Training_Dataset(X_train, Y_train)
    train_data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    model = LUNOT(k=n_neighbours, layer_size=layer_size, layer_count=layer_count).to(device, dtype=torch.float32)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_schedule_k**epoch)

    losses, train_scores, val_scores = [], [], []

    for epoch in range(1, 21):

        model.train()
        for X, Y in train_data_loader:
            optimizer.zero_grad()
            Y_pred = model(X)
            loss = criterion(Y_pred, Y)
            loss.backward()
            optimizer.step()
        
        losses.append(loss.item())
        if lr_schedule_k != False:
            scheduler.step()
        model.eval()
        with torch.no_grad():
            train_scores.append(get_auprc(model, X_train, Y_train))
            val_scores.append(get_auprc(model, X_val, Y_val))

        if epoch >= 3 and np.std(val_scores[-3:]) < 0.01:
            # print(f'stopping at epoch {epoch} due to low variance in validation scores')
            break
    
    return losses, train_scores, val_scores, hyperparameters

if __name__ == '__main__':

    print('simulating train data')

    train_dataset = Steel_APT_Dataset(
        unit_cells_per_side=150,
        cluster_relative_density=200,
        cluster_atom_counts=np.random.randint(low=10, high=50, size=400),
    ) #~65,000 non cluster atoms and 6,500 cluster atoms
    train_data = train_dataset.data[train_dataset.data['Element']!='Fe']

    print('simulating validation data')

    val_dataset = Steel_APT_Dataset(
        unit_cells_per_side=100,
        cluster_relative_density=200,
        cluster_atom_counts=np.random.randint(low=10, high=50, size=120),
    ) #~20,000 non cluster atoms and 2,000 cluster atoms
    val_data = val_dataset.data[val_dataset.data['Element']!='Fe']

    print('running model')

    results = pd.DataFrame(columns=['max_val_score', 'corresponding_train_score', 'lr', 'n_neighbours', 'layer_size', 'batch_size', 'layer_count', 'lr_schedule_k'])
    n_test_points = 200
    layer_counts = np.random.randint(low=2, high=10, size=n_test_points)
    lr_schedule_ks = np.random.uniform(low=0.01, high=1.0, size=n_test_points)

    start = time.perf_counter()

    futures = []
    with cf.ProcessPoolExecutor() as executor:
        for layer_count, lr_schedule_k in np.column_stack([layer_counts, lr_schedule_ks]):
            futures.append(
                executor.submit(test_model, train_data, val_data, lr=0.02, n_neighbours=50, layer_size=150,
                                 batch_size=300, layer_count=int(layer_count), lr_schedule_k=lr_schedule_k)
                )
    
        for i, future in enumerate(cf.as_completed(futures)):
            losses, train_scores, val_scores, hyperparameters = future.result()
            max_val_score_index = max(range(len(val_scores)), key=lambda i: val_scores[i])
            results.loc[i] = {'max_val_score': max(val_scores), 'corresponding_train_score': train_scores[max_val_score_index], **hyperparameters}
            print(f'iteration {i:<3} : max val score = {max(val_scores):.2%} overfitting = {train_scores[max_val_score_index] - max(val_scores):.2%}')

    end = time.perf_counter()

    print(f'{n_test_points} test points tested in {end-start:.2f}s')
    print(f'{(end-start)/n_test_points:.3f}s taken per test point')

    path = pl.Path(__file__).resolve().parent.joinpath(r'Analysis/LUNOT Hyperparameter Search.xlsx')
    kwargs = {'mode': 'a', 'if_sheet_exists': 'replace'} if path.is_file() else {'mode': 'w'}
    with pd.ExcelWriter(path, **kwargs) as writer:
        results.to_excel(writer, sheet_name='Round 4')
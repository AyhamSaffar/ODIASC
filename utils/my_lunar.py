from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted
from IPython.display import clear_output
import matplotlib.pyplot as plt


from pyod.models.base import BaseDetector

def generate_negative_samples(x, sample_type, proportion, epsilon):
    n_samples = int(proportion * len(x))
    n_dim = x.shape[-1]

    # uniform samples in range [x.min(),x.max()]
    rand_unif = x.min() + (x.max() - x.min()) * np.random.rand(n_samples, n_dim).astype('float32')
    # subspace perturbation samples
    x_temp = x[np.random.choice(np.arange(len(x)), size=n_samples)]
    randmat = np.random.rand(n_samples, n_dim) < 0.3
    rand_sub = x_temp + randmat * (epsilon * np.random.randn(n_samples, n_dim)).astype('float32')

    if sample_type == 'UNIFORM':
        neg_x = rand_unif
    if sample_type == 'SUBSPACE':
        neg_x = rand_sub
    if sample_type == 'MIXED':
        # randomly sample from uniform and gaussian negative samples
        neg_x = np.concatenate((rand_unif, rand_sub), 0)
        neg_x = neg_x[np.random.choice(np.arange(len(neg_x)), size=n_samples)]

    neg_y = np.ones(len(neg_x))

    return neg_x.astype('float32'), neg_y.astype('float32')

class FINETUNED_SCORE(nn.Module):
    def __init__(self, k):
        super(FINETUNED_SCORE, self).__init__()
        self.hidden_size = 128
        self.dropout = 0.8
        self.hidden_layers = 8

        layers = [nn.Linear(k, self.hidden_size), nn.Dropout(self.dropout), nn.GELU()]
        for i in range(self.hidden_layers - 1):
            layers += [nn.Linear(self.hidden_size, self.hidden_size), nn.Dropout(self.dropout), nn.GELU()]
            if i % 2 == 0:
                layers += [nn.BatchNorm1d(self.hidden_size, affine=False)]
        layers += [nn.Linear(self.hidden_size, 1), nn.Sigmoid()]
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return torch.squeeze(out, 1)

class RYAN_SCORE(nn.Module):
    def __init__(self, k):
        super(RYAN_SCORE, self).__init__()
        self.hidden_size = 256
        self.dropout = 0.8
        self.network = nn.Sequential(
            nn.Linear(k, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear(self.hidden_size, (int(self.hidden_size * 1))),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear((int(self.hidden_size * 1)), (int(self.hidden_size * 1))),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear((int(self.hidden_size * 1)), (int(self.hidden_size * 1))),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear((int(self.hidden_size * 1)), (int(self.hidden_size * 1))),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear((int(self.hidden_size * 1)), (int(self.hidden_size * 1))),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear((int(self.hidden_size * 1)), (int(self.hidden_size * 1))),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear((int(self.hidden_size * 1)), (int(self.hidden_size * 1))),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear((int(self.hidden_size * 1)), (int(self.hidden_size * 1))),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, x):
        out = self.network(x)
        out = torch.squeeze(out, 1)
        return out

class FINETUNED_WEIGHT(nn.Module):
    def __init__(self, k):
        super(FINETUNED_WEIGHT, self).__init__()
        self.hidden_size = 32
        self.dropout = 0.8
        self.hidden_layers = 15
        
        layers = [nn.Linear(k, self.hidden_size), nn.Dropout(self.dropout), nn.GELU()]
        for i in range(self.hidden_layers - 1):
            layers += [nn.Linear(self.hidden_size, self.hidden_size), nn.Dropout(self.dropout), nn.GELU()]
            if i % 2 == 0:
                layers += [nn.BatchNorm1d(self.hidden_size, affine=False)]
        layers += [nn.Linear(self.hidden_size, k), nn.GELU()]

        self.network = nn.Sequential(*layers)
        self.final_norm = nn.Sequential(nn.Sigmoid())

    def forward(self, x):
        alpha = self.network(x)
        alpha = F.softmax(alpha, dim=1)
        out = torch.sum(alpha * x, dim=1, keepdim=True)
        out = self.final_norm(out)
        return torch.squeeze(out, 1) 

class RYAN_WEIGHT(nn.Module):
    def __init__(self, k):
        super(RYAN_WEIGHT, self).__init__()
        self.hidden_size= 32
        self.dropout = 0.8
        self.network = nn.Sequential(
            nn.Linear(k, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear(self.hidden_size, (int(self.hidden_size * 1))),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear((int(self.hidden_size * 1)), (int(self.hidden_size * 1))),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear((int(self.hidden_size * 1)), (int(self.hidden_size * 1))),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear((int(self.hidden_size * 1)), (int(self.hidden_size * 1))),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear((int(self.hidden_size * 1)), (int(self.hidden_size * 1))),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear((int(self.hidden_size * 1)), (int(self.hidden_size * 1))),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear((int(self.hidden_size * 1)), (int(self.hidden_size * 1))),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear((int(self.hidden_size * 1)), (int(self.hidden_size * 1))),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, k),
            nn.Sigmoid()
        )
        self.final_norm = nn.BatchNorm1d(1)

    def forward(self, x):
        alpha = self.network(x)
        # get weights > 0 and sum to 1.0
        alpha = F.softmax(alpha, dim=1)
        # multiply weights by each distance in input vector
        out = torch.sum(alpha * x, dim=1, keepdim=True)
        # batch norm
        out = self.final_norm(out)
        out = torch.squeeze(out, 1)
        return out

class LUNAR(BaseDetector):

    def __init__(self, model_type="finetuned weight", n_neighbours=5, negative_sampling="MIXED",
                 val_size=0.1, epsilon=0.1, proportion=1.0, n_epochs=200, lr=0.001, wd=0.1, verbose=0):
        super(LUNAR, self).__init__()

        self.model_type = model_type
        self.n_neighbours = n_neighbours
        self.negative_sampling = negative_sampling
        self.epsilon = epsilon
        self.proportion = proportion
        self.n_epochs = n_epochs
        self.lr = lr
        self.wd = wd
        self.val_size = val_size
        self.verbose = verbose
        self.device = torch.device('cpu')


        if model_type == "FINETUNED_WEIGHT":
            self.network = FINETUNED_WEIGHT(n_neighbours).to(self.device)
            self.scaler = None
    
        elif model_type == "FINETUNED_SCORE":
            self.network = FINETUNED_SCORE(n_neighbours).to(self.device)
            self.scaler = None

        elif model_type == "RYAN_WEIGHT":
            self.network = RYAN_WEIGHT(n_neighbours).to(self.device)
            self.scaler=MinMaxScaler()
    
        elif model_type == "RYAN_SCORE":
            self.network = RYAN_SCORE(n_neighbours).to(self.device)
            self.scaler=MinMaxScaler()

    def fit(self, X, y=None):

        X = np.array(X)
        self._set_n_classes(y)
        X = X.astype('float32')
        y = np.zeros(len(X))

        # split training and validation sets
        train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=self.val_size)

        if self.scaler != None:
            self.scaler.fit(train_x)
            train_x = self.scaler.transform(train_x)
            val_x = self.scaler.transform(val_x)

        # generate negative samples for training and validation set seperately 
        neg_train_x, neg_train_y = generate_negative_samples(train_x, self.negative_sampling, self.proportion,
                                                             self.epsilon)
        neg_val_x, neg_val_y = generate_negative_samples(val_x, self.negative_sampling, self.proportion, self.epsilon)

        train_x = np.vstack((train_x, neg_train_x))
        train_y = np.hstack((train_y, neg_train_y))
        val_x = np.vstack((val_x, neg_val_x))
        val_y = np.hstack((val_y, neg_val_y))

        self.neigh = NearestNeighbors(n_neighbors=self.n_neighbours + 1)
        self.neigh.fit(train_x)

        # nearest neighbours of training set
        train_dist, _ = self.neigh.kneighbors(train_x[train_y == 0], n_neighbors=self.n_neighbours + 1)
        neg_train_dist, _ = self.neigh.kneighbors(train_x[train_y == 1], n_neighbors=self.n_neighbours)
        # remove self loops of normal training points
        train_dist = np.vstack((train_dist[:, 1:], neg_train_dist))
        # nearest neighbours of validation set
        val_dist, _ = self.neigh.kneighbors(val_x, n_neighbors=self.n_neighbours)

        train_dist = torch.tensor(train_dist, dtype=torch.float32).to(self.device)
        train_y = torch.tensor(train_y, dtype=torch.float32).to(self.device)
        val_dist = torch.tensor(val_dist, dtype=torch.float32).to(self.device)
        val_y = torch.tensor(val_y, dtype=torch.float32).to(self.device)
        
        criterion = nn.MSELoss(reduction='none')
        # criterion = nn.BCELoss()
        #optimizer = optim.SGD(self.network.parameters(), lr=self.lr, weight_decay=self.wd, momentum=0.9)
        optimizer = optim.AdamW(self.network.parameters(), lr=self.lr, weight_decay=self.wd)

        best_val_score = 0
        average_losses = []
        train_scores = []
        val_scores = []
        for epoch in range(self.n_epochs):

            # see performance of model before epoch
            with torch.no_grad():
                self.network.eval()
                out = self.network(train_dist)
                train_score = roc_auc_score(train_y.cpu(), out.cpu())
                out = self.network(val_dist)
                val_score = roc_auc_score(val_y.cpu(), out.cpu())

                if val_score >= best_val_score:
                    best_dict = {'epoch': epoch,
                                 'model_state_dict': deepcopy(self.network.state_dict()),
                                 'optimizer_state_dict': deepcopy(optimizer.state_dict()),
                                 'train_score': train_score,
                                 'val_score': val_score,
                                 }

                    best_val_score = val_score

            if self.verbose == 1 and epoch != 0:
                print(f"Epoch {epoch} \t Train Loss {average_loss: .4f} \t Train Score {train_score: .4f} \t Val Score {val_score: .4f}")

            # training loop
            self.network.train()
            optimizer.zero_grad()
            out = self.network(train_dist)
            losses = criterion(out, train_y)
            average_loss = losses.mean()
            loss = losses.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1)
            optimizer.step()
                             
            average_losses.append(torch.clone(average_loss).detach().numpy())
            train_scores.append(train_score)
            val_scores.append(val_score)

        if self.verbose == 1:
            clear_output()
            print(f"Finished training...\nBest Model: Epoch {best_dict['epoch']} \t Train Score {best_dict['train_score']: .4f} \t Val Score {best_dict['val_score']: .4f}")
            fig, ax = plt.subplots()
            ax.plot(range(self.n_epochs), average_losses, label='losses')
            ax.plot(range(self.n_epochs), train_scores, label='train scores')
            ax.plot(range(self.n_epochs), val_scores, label='val scores')
            ax.set_xlabel('epochs')
            ax.hlines(y=0.5, xmin=0, xmax=self.n_epochs, colors='black', linestyles='dashed')
            ax.legend()

            
        # load best model after training
        self.network.load_state_dict(best_dict['model_state_dict'])

        # Determine outlier scores for train set
        # scale data if scaler has been passed
        if (self.scaler == None):
            X_norm = np.copy(X)
        else:
            X_norm = self.scaler.transform(X)

        # nearest neighbour search
        dist, _ = self.neigh.kneighbors(X_norm, self.n_neighbours)
        dist = torch.tensor(dist, dtype=torch.float32).to(self.device)
        # forward pass
        with torch.no_grad():
            self.network.eval()
            anomaly_scores = self.network(dist)

        self.decision_scores_ = anomaly_scores.cpu().detach().numpy().ravel()
        self._process_decision_scores()

        return self

    def decision_function(self, X):

        check_is_fitted(self, ['decision_scores_'])
        # X = check_array(X)
        X = X.astype('float32')

        # scale data
        if (self.scaler == None):
            pass
        else:
            X = self.scaler.transform(X)

        # nearest neighbour search
        dist, _ = self.neigh.kneighbors(X, self.n_neighbours)
        dist = torch.tensor(dist, dtype=torch.float32).to(self.device)
        # forward pass
        with torch.no_grad():
            self.network.eval()
            anomaly_scores = self.network(dist)

        scores = anomaly_scores.cpu().detach().numpy().ravel()

        return scores
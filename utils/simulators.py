import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.neighbors import NearestNeighbors

class Steel_APT_Dataset():
    '''
    Factory class to simulate APT data of an irradiated steel crystal. All measurements in nm.
    '''

    def __init__(self, unit_cells_per_side: int, cluster_relative_density: float, cluster_atom_counts: np.ndarray) -> None:
        self.composition = {'Fe': 0.98, 'Ni': 0.009, 'Mn': 0.01, 'Cu': 0.001}
        self.lattice_parameter = 0.28664 #steel rtp 
        self.length = unit_cells_per_side * self.lattice_parameter
        self.cluster_atom_counts = cluster_atom_counts
        self.cluster_relative_density = cluster_relative_density
        self.bulk_atom_density = 2 / (self.lattice_parameter**3)
        self.bulk_solute_density = self.bulk_atom_density * (1 - self.composition['Fe'])
        self.cluster_volumes = cluster_atom_counts / (self.bulk_solute_density*cluster_relative_density)
        self.cluster_radii = (3/4 * self.cluster_volumes/np.pi)**(1/3)

        lattice = self._generate_steel_lattice()
        DBScan_baseline = lattice.sample(frac=1-self.composition['Fe'])
        self.DBScan_baseline = self._add_artifacts(DBScan_baseline)
        data = self._add_clusters(lattice)
        data = self._assign_elements(data)
        self.data = self._add_artifacts(data)

    def display(self, without_fe=True) -> None:
        data = self.data[self.data['Element'] != 'Fe'] if without_fe else self.data
        fig = px.scatter_3d(data, x='X', y='Y', z='Z', color='Element')
        fig.update_traces(marker={'size': 1.5})
        fig.show(renderer='vscode')

    def parameters(self) -> dict:
        return {key: value for key, value in vars(self).items() if key!='data'}

    def _generate_steel_lattice(self) -> pd.DataFrame:
        lattice_parameter, length = self.lattice_parameter, self.length
        x, y, z = np.mgrid[0:length:lattice_parameter, 0:length:lattice_parameter, 0:length:lattice_parameter]
        corner_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        body_points = corner_points.copy() + lattice_parameter/2
        bcc_points = np.vstack([corner_points, body_points])
        lattice = pd.DataFrame(bcc_points, columns=['X', 'Y', 'Z'])
        lattice['is cluster'] = np.zeros(shape=lattice.shape[0], dtype=int)
        return lattice

    def _add_clusters(self, data: pd.DataFrame) -> pd.DataFrame:
        num_clusters = self.cluster_atom_counts.shape[0]
        self.cluster_positions = np.random.uniform(low=0, high=self.length, size=[num_clusters,3])
        for i in range(num_clusters):
            centre_pos = self.cluster_positions[i]
            radius, atom_count = self.cluster_radii[i], self.cluster_atom_counts[i]
            cluster_points = np.random.normal(loc=centre_pos, scale=radius, size=[atom_count, 3])
            cluster_points = pd.DataFrame(cluster_points, columns=['X', 'Y', 'Z'])
            cluster_points['is cluster'] = np.ones(shape=atom_count, dtype=int)
            data = pd.concat([data, cluster_points])

        data['around cluster'] = np.zeros(shape=data.shape[0])
        knn = NearestNeighbors(n_neighbors=max(1, int(max(self.cluster_volumes)*self.bulk_atom_density)))
        knn.fit(data[['X', 'Y', 'Z']].to_numpy())
        for i in range(num_clusters):
            position, radius = self.cluster_positions[i], self.cluster_radii[i]
            distances, indexes = knn.kneighbors([position])
            data.iloc[indexes[distances<radius], data.columns.get_loc('around cluster')] = 1
        return data
    
    def _assign_elements(self, data: pd.DataFrame) -> pd.DataFrame:
        data['Element'] = np.full(shape=data.shape[0], fill_value='Fe')
        atoms_left = {element: int(frac*data.shape[0]) for element, frac in self.composition.items() if element!='Fe'}
        mask = data['is cluster']==1
        data.loc[mask, 'Element'] = np.random.choice(list(atoms_left.keys()), size=data.loc[mask, 'Element'].shape)
        for solute in atoms_left.keys():
            atoms_left[solute] -= sum(data['Element']==solute)
            assert atoms_left[solute]>=0, "More clusters requested than solutes available. Consider increasing unit cells per side"
        
        mask = (data['around cluster'] != 1) & (data['is cluster'] != 1)
        element_array_list = [np.full(shape=count, fill_value=element) for element, count in atoms_left.items()]
        element_array = np.hstack(element_array_list)
        indexes = np.random.choice(data[mask].index, size=element_array.shape[0], replace=False)
        data.iloc[indexes, data.columns.get_loc('Element')] = element_array
        return data

    def _add_artifacts(self, data:pd.DataFrame) -> pd.DataFrame:
        data = data.sample(frac=0.55)
        x_noise = np.random.normal(loc=0, scale=0.5, size=data.shape[0])
        y_noise = np.random.normal(loc=0, scale=0.5, size=data.shape[0])
        z_noise = np.random.normal(loc=0, scale=0.05, size=data.shape[0])
        noise = np.column_stack([x_noise, y_noise, z_noise])
        data.loc[:, ['X', 'Y', 'Z']] += noise
        return data
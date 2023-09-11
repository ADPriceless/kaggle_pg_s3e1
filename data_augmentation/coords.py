import pickle

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


class NearestCluster(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters: int) -> None:
        super().__init__()
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            n_init='auto'
        )

    def fit(self, X, y=None):
        self.kmeans.fit(X.loc[:, 'Latitude':'Longitude'])
        return self

    def transform(self, X):
        predictions = self.kmeans.predict(X.loc[:, 'Latitude':'Longitude'])
        # the line below may be slow for larger datasets
        centers = np.array([self.kmeans.cluster_centers_[pred] for pred in predictions])
        X['NearestClustLat'] = centers[:, 0]
        X['NearestClustLon'] = centers[:, 1]
        return X


class DistToCoastLoader(BaseEstimator, TransformerMixin):
    def __init__(self, filepath: str, extra_data: bool) -> None:
        super().__init__()
        self.filepath = filepath
        self.extra_data = extra_data

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        with open(self.filepath, 'rb') as pckl:
            dist_to_coast = pickle.load(pckl)
        if self.extra_data:
            X['DistToCoast'] = dist_to_coast
        else:
            X['DistToCoast'] = dist_to_coast[:X.shape[0]]
        return X


class DistToCity(BaseEstimator, TransformerMixin):
    def __init__(self, city_coords: list[np.array]) -> None:
        super().__init__()
        self.city_coords = city_coords
        self.dist_to_city = None

    def fit(self, X, y=None):
        distances = np.zeros([X.shape[0], len(self.city_coords)])
        for i, city_coord in enumerate(self.city_coords):
            house_coord = X.loc[:, ['Longitude', 'Latitude']].to_numpy()
            manhattan = house_coord - np.expand_dims(city_coord, axis=0)
            distances[:, i] = np.linalg.norm(manhattan, ord=2, axis=1, keepdims=True).reshape(-1)
        for i in range(len(self.city_coords) - 1):
            self.dist_to_city = np.where(
                distances[:, i] < distances[:, i + 1],
                distances[:, i],
                distances[:, i + 1]
            )
        return self

    def transform(self, X):
        X['DistToMajorCity'] = self.dist_to_city
        return X

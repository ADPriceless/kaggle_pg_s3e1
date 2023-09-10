import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


class NearestCluster(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters) -> None:
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

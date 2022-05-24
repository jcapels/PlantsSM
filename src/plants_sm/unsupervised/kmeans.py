import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


class SklearnKMeans:

    def __init__(self, **kwargs):
        self._fitted = False
        self.kwargs = kwargs
        self.kmeans = KMeans(**self.kwargs)

    @property
    def fitted(self):
        return self._fitted

    @fitted.setter
    def fitted(self, value: bool):
        self._fitted = value

    def fit(self, x):
        self.kmeans.fit(x)
        self.fitted = True  # add decorator
        return self

    def transform(self, x) -> pd.DataFrame:
        if self.fitted:
            return self.kmeans.transform(x)
        else:
            raise ValueError("The model is not fitted")

    def predict(self, x):
        if self.fitted:
            return self.kmeans.predict(x)
        else:
            raise ValueError("The model is not fitted")

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def fit_predict(self, x):
        return self.fit(x).predict(x)

    def fit_predict_generate_scatter_plot(self, x):
        labels = self.kmeans.fit_predict(x)
        u_labels = np.unique(labels)
        plt.subplots(1, figsize=(25, 15))
        # plotting the results:
        centroids = self.kmeans.cluster_centers_
        for i in u_labels:
            plt.scatter(x[labels == i, 0], x[labels == i, 1], label=i, s=40)
        plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
        plt.title('KMeans')
        plt.legend()

    def generate_distortion_graph(self, x):
        distortions = []
        for i in range(1, 21):
            km = KMeans(
                **self.kwargs
            )
            km.fit(x)
            distortions.append(km.inertia_)

        # plot
        plt.plot(range(1, 21), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.title('Elbow Method')
        plt.show()

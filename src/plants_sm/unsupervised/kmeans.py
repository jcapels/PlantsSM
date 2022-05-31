import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from plants_sm.unsupervised.model import UnsupervisedModel


class SklearnKMeans(UnsupervisedModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kmeans = KMeans(**self.kwargs)

    def _fit(self, x):
        self.kmeans.fit(x)
        self.fitted = True  # add decorator
        return self

    def _transform(self, x) -> pd.DataFrame:
        if self.fitted:
            return self.kmeans.transform(x)
        else:
            raise ValueError("The model is not fitted")

    def _predict(self, x):
        if self.fitted:
            return self.kmeans.predict(x)
        else:
            raise ValueError("The model is not fitted")

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

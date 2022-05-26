import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns


class SklearnPCA:

    def __init__(self, **kwargs):
        self._fitted = None
        self.pca = PCA(**kwargs)

    @property
    def fitted(self):
        return self._fitted

    @fitted.setter
    def fitted(self, value: bool):
        self._fitted = value

    def fit(self, x):
        self.pca.fit(x)
        self.fitted = True  # add decorator
        return self

    def transform(self, x) -> pd.DataFrame:
        if self.fitted:
            return self.pca.transform(x)
        else:
            raise ValueError("The model is not fitted")

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def generate_pca_variance_plot(self):
        plt.figure()
        plt.plot(range(self.pca.n_components_), np.cumsum(self.pca.explained_variance_ratio_ * 100))
        for i in range(self.pca.n_components_):
            cumulative_sum = np.sum(self.pca.explained_variance_ratio_[:i])
            if cumulative_sum > 0.95:
                break
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)')
        plt.title('Cumulative Explained Variance')
        plt.xlim(0, self.pca.n_components_)
        plt.ylim(30, 100)
        plt.axhline(y=95, color='r')
        plt.axvline(x=i, color='g')
        plt.show()
        print(f'First 2 PC: {sum(self.pca.explained_variance_ratio_[0:2] * 100)}')
        print(f'First {i} PC: {sum(self.pca.explained_variance_ratio_[0:i] * 100)}')

    def generate_dotplot(self, data):
        plt.figure(figsize=(20, 20))
        plt.xlabel(f'PC1 = {np.round(self.pca.explained_variance_ratio_[0] * 100, 3)}% variance')
        plt.ylabel(f'PC2 = {np.round(self.pca.explained_variance_ratio_[1] * 100, 3)}% variance')
        sns.scatterplot(
            x=data[:, 0], y=data[:, 1],
            hue=None,
            palette=sns.color_palette("deep", 2),
            legend="full",
            s=25)

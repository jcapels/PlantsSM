import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.decomposition import PCA
import seaborn as sns

from plants_sm.unsupervised.model import UnsupervisedModel


class SklearnPCA(UnsupervisedModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pca = PCA(**self.kwargs)

    def _predict(self, x):
        pass

    def _fit(self, x):
        self.pca.fit(x)
        self.fitted = True
        return self

    def _transform(self, x) -> pd.DataFrame:
        if self.fitted:
            return self.pca.transform(x)
        else:
            raise ValueError("The model is not fitted")

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

    def generate_dotplot(self, data, labels):
        plt.figure(figsize=(20, 20))
        plt.xlabel(f'PC1 = {np.round(self.pca.explained_variance_ratio_[0] * 100, 3)}% variance')
        plt.ylabel(f'PC2 = {np.round(self.pca.explained_variance_ratio_[1] * 100, 3)}% variance')

        data = DataFrame(data)
        data["label"] = labels
        sns.scatterplot(
            data=data,
            x=data.columns[0], y=data.columns[1],
            hue="label",
            palette=sns.color_palette("deep", len(np.unique(data["label"]))),
            legend="full",
            s=25)

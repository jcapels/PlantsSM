import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


class SklearnTSNE:
    def __init__(self, **kwargs):
        self._fitted = None
        self.tsne = TSNE(**kwargs)

    @property
    def fitted(self):
        return self._fitted

    @fitted.setter
    def fitted(self, value: bool):
        self._fitted = value

    def fit(self, x):
        self.tsne.fit(x)
        self.fitted = True  # add decorator
        return self

    def fit_transform(self, x):
        return self.tsne.fit_transform(x)

    @staticmethod
    def generate_dotplot(x):
        plt.figure(figsize=(20, 20))
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plot = sns.scatterplot(
            x[:, 0], x[:, 1],
            hue=None,
            palette=sns.color_palette("deep", 2),
            legend="full",
            s=25)
        plt.title("TSNE")
        return plot

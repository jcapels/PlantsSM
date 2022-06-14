import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.manifold import TSNE
import seaborn as sns

from plants_sm.unsupervised.model import UnsupervisedModel


class SklearnTSNE(UnsupervisedModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tsne = TSNE(**self.kwargs)

    def _fit(self, x):
        self.tsne.fit(x)
        self.fitted = True
        return self

    def fit_transform(self, x):
        return self.tsne.fit_transform(x)

    def _transform(self, x):
        pass

    def _predict(self, x):
        pass

    @staticmethod
    def generate_dotplot(x, labels=None):
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

        data = DataFrame(x)
        data["label"] = labels
        sns.scatterplot(
            data=data,
            x=data.columns[0], y=data.columns[1],
            hue="label",
            palette=sns.color_palette("deep", len(np.unique(data["label"]))),
            legend="full",
            s=25)
        return plot

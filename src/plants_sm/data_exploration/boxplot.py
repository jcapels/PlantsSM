from typing import Tuple, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Boxplot:

    def __init__(self, title: Tuple[str, int], orientation: str, pallete: str = "Set3", **kwargs):
        """

        Parameters
        ----------
        title
        orientation
        pallete
        kwargs
        """
        self._plot = None
        self.title = title
        self.orientation = orientation
        self.pallete = pallete
        self.kwargs = kwargs

    def generate(self, dataframe: pd.DataFrame, features_name: List[str], label_name: str = None):
        """

        Parameters
        ----------
        features_name
        label_name
        dataframe

        Returns
        -------

        """
        plt.subplots(figsize=(20, 10))
        sns.set(font_scale=1.4)
        plt.title(self.title[0], fontsize=self.title[1])
        self.plot = sns.boxplot(x="variable", y="value", data=pd.melt(dataframe.loc[:, features_name], label_name),
                                hue=label_name,
                                palette=self.pallete,
                                orient=self.orientation)

    @property
    def plot(self):
        return self._plot

    @plot.setter
    def plot(self, value):
        self._plot = value


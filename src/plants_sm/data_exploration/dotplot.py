from typing import Tuple, Union, List

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class Dotplot:

    def __init__(self, title: Tuple[str, int], orientation: str, pallete: str, **kwargs):
        self.title = title
        self.orientation = orientation
        self.pallete = pallete
        self.kwargs = kwargs

    def generate(self, dataframe: Union[pd.DataFrame, List[List]], label: str = None):
        plt.figure(figsize=(20, 20))
        sns.scatterplot(
            dataframe[:, 0], dataframe[:, 1],
            hue=label,
            palette=self.pallete,
            data=dataframe,
            **self.kwargs
        )

        plt.legend(prop={'size': 10})
        plt.title(self.title[0], fontsize=self.title[1])

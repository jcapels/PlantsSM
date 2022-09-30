from plants_sm.data_structures.dataset import Dataset
from plants_sm.transformation.transformer import Transformer


class Padder(Transformer):

    pad_width: int = None
    padding_token: str

    def _fit(self, dataset: Dataset) -> 'Estimator':
        if not self.pad_width:
            self.max_length = dataset.dataframe.loc[:, dataset.representation_field].str.len().max()
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        pass

import pandas as pd
from pandas import DataFrame

from plants_sm.featurization.featurizer import FeaturesGenerator

from bio_embeddings.embed.unirep_embedder import UniRepEmbedder


class UniRepEmbeddings(FeaturesGenerator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        import logging
        logging.getLogger('numba').setLevel(logging.WARNING)

        self.unirep_embedder = UniRepEmbedder(**kwargs)

    def _featurize(self, instance: str, identifier: str, identifier_field_name: str) -> pd.DataFrame:
        """

        Parameters
        ----------
        instance
        identifier
        identifier_field_name

        Returns
        -------

        """
        embedding = self.unirep_embedder.embed(instance)
        features_names = [f"unirep_embedding_{i}" for i in range(1, len(embedding) + 1)]
        features_df = DataFrame([embedding], index=[0], columns=features_names)
        if self.features_names is None:
            self.features_names = features_names

        features_df[identifier_field_name] = [identifier]
        return features_df

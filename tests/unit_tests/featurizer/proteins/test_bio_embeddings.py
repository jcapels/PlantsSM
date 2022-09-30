import numpy as np
import pandas as pd

from plants_sm.featurization.proteins.bio_embeddings.word2vec import Word2Vec
from ...featurizer.proteins.test_protein_featurizers import TestProteinFeaturizers

from plants_sm.featurization.proteins.bio_embeddings.unirep import UniRepEmbeddings

import xarray as xr


class TestEmbeddings(TestProteinFeaturizers):

    def test_unirep_embeddings(self):
        dataset = UniRepEmbeddings().fit_transform(self.dataset)
        self.assertEqual(dataset.features_dataframe.shape, (2, 1901))
        self.assertAlmostEqual(0.016243484, dataset.features_dataframe.iloc[0, 1])

    def test_unirep_embeddings_3d(self):
        dataset = UniRepEmbeddings(output_shape_dimension=3).fit_transform(self.dataset)

        features = dataset.features_dataframe.to_numpy() \
            .reshape((len(dataset.identifiers), dataset.features_shape[1], len(dataset.features_fields)))

        self.assertEqual((2, 454, 1900), features.shape)

    # TODO: it generates 3-dimensional vectors, so it is not compatible with the current implementation
    # @unittest.skip("it generates 3-dimensional vectors, so it is not compatible with the current implementation")
    def test_word2vec_embeddings_2d(self):
        dataset = Word2Vec().fit_transform(self.dataset)
        self.assertEqual(dataset.features_dataframe.shape, (2, 513))
        self.assertAlmostEqual(-0.033814143, dataset.features_dataframe.iloc[0, 1])

    def test_word2vec_embeddings_3d(self):
        dataset = Word2Vec(output_shape_dimension=3).fit_transform(self.dataset)

        features = dataset.features_dataframe.to_numpy() \
            .reshape((len(dataset.identifiers), dataset.features_shape[1], len(dataset.features_fields)))

        self.assertEqual(dataset.dataframe.shape, (2, 2))
        self.assertEqual(features.shape, (2, 453, 512))

    # def test_word2vec_embeddings_3d_xarray(self):
        # sequences = dataset.identifiers  # Years
        # aa = np.arange(0, features.shape[1])  # Samples
        # features_names = np.array(dataset.features_fields)  # Patients
        #
        # maj_dim = 1
        # for dim in features.shape[:-1]:
        #     maj_dim = maj_dim * dim
        # new_dims = (maj_dim, features.shape[-1])
        # features = features.reshape(new_dims)
        #
        # # Create the MultiIndex from years, samples and patients.
        # midx = pd.MultiIndex.from_product([sequences, aa])
        #
        # # Create sample data for each patient, and add the MultiIndex.
        # patient_data = pd.DataFrame(data=features, index=midx, columns=features_names)
        # data_0 = patient_data.loc[["0"], :, :]
        # data_1 = patient_data.loc[["1"], :, :]
        #
        # data = pd.concat((data_0, data_1), axis=0)
        #
        # print(pd.pivot(data, index=["identifier"], columns=features_names, values=aa))
        #
        # numpy_convertion = patient_data.to_numpy().reshape((len(dataset.identifiers),
        #                                                     len(aa), len(dataset.features_fields)))
        # print(patient_data.to_numpy())
        # # view 3D DataFrame

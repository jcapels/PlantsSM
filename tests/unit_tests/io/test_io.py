import numpy as np
import pandas as pd

from plants_sm.io import read_csv, write_csv, CSVReader, CSVWriter

import os
from unittest import TestCase

from plants_sm.io.excel import read_excel, write_excel, ExcelReader, ExcelWriter
from plants_sm.io.h5 import write_h5, read_h5
from plants_sm.io.json import write_json, read_json
from plants_sm.io.pickle import write_pickle, read_pickle
from plants_sm.io.yaml import YAMLReader, YAMLWriter, read_yaml, write_yaml
from tests import TEST_DIR


class TestIO(TestCase):

    def setUp(self) -> None:
        self.test_read_csv = os.path.join(TEST_DIR, "data", "example1.csv")
        self.test_read_excel = os.path.join(TEST_DIR, "data", "drug_list.xlsx")
        self.df_path_to_write_csv = os.path.join(TEST_DIR, "data", "test.csv")
        self.df_path_to_write_xlsx = os.path.join(TEST_DIR, "data", "test.xlsx")
        self.test_read_yaml = os.path.join(TEST_DIR, "data", "defaults.yml")
        self.test_write_yaml = os.path.join(TEST_DIR, "data", "defaults_temp.yml")
        self.test_read_pickle = os.path.join(TEST_DIR, "data", "test.pkl")
        self.test_write_pickle = os.path.join(TEST_DIR, "data", "test2.pkl")
        self.test_json_reader = os.path.join(TEST_DIR, "data", "test.json")
        self.test_json_writer = os.path.join(TEST_DIR, "data", "test2.json")
        self.test_h5_reader = os.path.join(TEST_DIR, "data", "test.h5")
        self.test_h5_writer = os.path.join(TEST_DIR, "data", "test2.h5")

    def tearDown(self) -> None:

        paths_to_remove = [self.df_path_to_write_csv,
                           self.df_path_to_write_xlsx,
                           self.test_write_yaml,
                           self.test_write_pickle,
                           self.test_json_writer,
                           self.test_h5_writer]

        for path in paths_to_remove:
            if os.path.exists(path):
                os.remove(path)

    def test_read_csv(self):
        df = read_csv(self.test_read_csv)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(df.shape[1], 1026)

    def test_write_csv(self):
        df = read_csv(self.test_read_csv)

        written = write_csv(self.df_path_to_write_csv, df, index=False)

        self.assertTrue(written)
        self.assertTrue(os.path.exists(self.df_path_to_write_csv))

        df = read_csv(self.df_path_to_write_csv)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(df.shape[1], 1026)

    def test_read_excel(self):
        df = read_excel(self.test_read_excel, sheet_name="DrugInfo")
        self.assertEqual(df.shape[0], 271)
        self.assertEqual(df.shape[1], 117)

    def test_write_excel(self):
        df1 = read_excel(self.test_read_excel, sheet_name="DrugInfo")

        write_excel(self.df_path_to_write_xlsx, df1, index=False)

        df2 = read_excel(self.df_path_to_write_xlsx)

        self.assertEqual(df1.shape, df2.shape)
        self.assertEqual(df1.iloc[0, 0], df1.iloc[0, 0])

    def test_csv_reader(self):
        reader = CSVReader(self.test_read_csv)
        ddf = reader.read()
        reader.close_buffer()

        self.assertEqual(ddf.shape, (100, 1026))
        self.assertEqual(ddf.iloc[0, 0], 0.0)
        self.assertEqual(reader.file_types(), ['txt', 'csv', 'tsv'])

    def test_csv_writer(self):
        df = read_csv(self.test_read_csv)

        writer = CSVWriter(filepath_or_buffer=self.df_path_to_write_csv, index=False)
        written = writer.write(df=df)
        writer.close_buffer()

        self.assertTrue(written)
        self.assertTrue(os.path.exists(self.df_path_to_write_csv))

        df = read_csv(self.df_path_to_write_csv)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(df.shape[1], 1026)

    def test_excel_reader(self):
        reader = ExcelReader(self.test_read_excel, sheet_name="DrugInfo")
        df = reader.read()
        reader.close_buffer()

        self.assertEqual(df.shape, (271, 117))
        self.assertEqual(df.iloc[0, 0], 'ABACAVIR SULFATE')
        self.assertEqual(reader.file_types(), ["xlsx"])

    def test_excel_writer(self):
        df1 = read_excel(self.test_read_excel, sheet_name="DrugInfo")

        writer = ExcelWriter(filepath_or_buffer=self.df_path_to_write_xlsx, index=False)
        written = writer.write(df1)
        writer.close_buffer()

        self.assertTrue(written)
        self.assertTrue(os.path.exists(self.df_path_to_write_xlsx))

        df = read_excel(self.df_path_to_write_xlsx)
        self.assertEqual(df.shape[0], 271)
        self.assertEqual(df.shape[1], 117)
        self.assertEqual(df1.shape, df.shape)

    def test_yaml_reader(self):
        with YAMLReader(file_path_or_buffer=self.test_read_yaml) as reader:
            # suppress warning due to dask delayed decorator: https://github.com/dask/dask/issues/7779
            # noinspection PyUnresolvedReferences
            config = reader.read().compute()
        self.assertEqual(config["word2vec"]["model_file"],
                         "http://data.bioembeddings.com/public/embeddings/embedding_models/word2vec/word2vec.model")

        config = read_yaml(file_path_or_buffer=self.test_read_yaml)
        self.assertEqual(config["word2vec"]["model_file"],
                         "http://data.bioembeddings.com/public/embeddings/embedding_models/word2vec/word2vec.model")

        config = read_yaml(file_path_or_buffer=self.test_read_yaml, preserve_order=True)
        self.assertEqual(config["word2vec"]["model_file"],
                         "http://data.bioembeddings.com/public/embeddings/embedding_models/word2vec/word2vec.model")

    def test_yaml_writer(self):
        with YAMLReader(file_path_or_buffer=self.test_read_yaml) as reader:
            config = reader.read().compute()

        with YAMLWriter(file_path_or_buffer=self.test_write_yaml) as writer:
            flag = writer.write(data=config)
        self.assertTrue(flag)

        with YAMLReader(file_path_or_buffer=self.test_write_yaml) as reader:
            config = reader.read().compute()

        self.assertEqual(config["word2vec"]["model_file"],
                         "http://data.bioembeddings.com/public/embeddings/embedding_models/word2vec/word2vec.model")

        flag = write_yaml(file_path_or_buffer=self.test_write_yaml, data=config)
        self.assertTrue(flag)
        config = read_yaml(file_path_or_buffer=self.test_write_yaml)

        self.assertEqual(config["word2vec"]["model_file"],
                         "http://data.bioembeddings.com/public/embeddings/embedding_models/word2vec/word2vec.model")

    def test_pickle_reader(self):
        dataframe = read_pickle(self.test_read_pickle)
        self.assertTrue(isinstance(dataframe, pd.DataFrame))
        self.assertEqual(dataframe.shape, (0, 0))

    def test_pickle_writer(self):
        write_pickle(self.test_write_pickle, pd.DataFrame())

        dataframe = read_pickle(self.test_write_pickle)
        self.assertTrue(isinstance(dataframe, pd.DataFrame))
        self.assertEqual(dataframe.shape, (0, 0))

    def test_json_reader(self):
        config = read_json(self.test_json_reader)
        self.assertTrue(isinstance(config, dict))
        self.assertEqual(config["test"], "test")

    def test_json_writer(self):
        write_json(self.test_json_reader, {"test": "test"})

        config = read_json(self.test_json_reader)
        self.assertTrue(isinstance(config, dict))
        self.assertEqual(config["test"], "test")

    def test_h5_reader(self):
        h5_file = read_h5(self.test_h5_reader)
        self.assertTrue(isinstance(h5_file, np.ndarray))
        self.assertEqual(h5_file.shape, (0,))

    def test_h5_writer(self):
        write_h5(np.array([]), self.test_h5_writer)

        h5_file = read_h5(self.test_h5_reader)
        self.assertTrue(isinstance(h5_file, np.ndarray))
        self.assertEqual(h5_file.shape, (0,))

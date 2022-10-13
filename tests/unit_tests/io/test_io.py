from plants_sm.io import read_csv, write_csv, CSVReader, CSVWriter

import os
from unittest import TestCase

from plants_sm.io.excel import read_excel, write_excel, ExcelReader, ExcelWriter
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

    def tearDown(self) -> None:

        paths_to_remove = [self.df_path_to_write_csv,
                           self.df_path_to_write_xlsx,
                           self.test_write_yaml]

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
        df = ddf.compute()
        reader.close_buffer()

        self.assertEqual(df.shape, (100, 1026))
        self.assertEqual(df.iloc[0, 0], 0.0)
        self.assertEqual(reader.file_types, ['txt', 'csv', 'tsv'])

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
        self.assertEqual(reader.file_types, ["xlsx"])

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